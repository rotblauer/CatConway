use wgpu::util::DeviceExt;

use crate::grid::SimParams;

/// Manages the GPU compute pipeline that advances the Game of Life simulation.
///
/// Uses double-buffered storage buffers: each step reads from one and writes
/// to the other, then they swap.
pub struct Simulation {
    compute_pipeline: wgpu::ComputePipeline,
    bind_groups: [wgpu::BindGroup; 2],
    storage_buffers: [wgpu::Buffer; 2],
    params_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    /// Which buffer index is currently the "input" (0 or 1).
    current: usize,
    pub generation: u64,
}

impl Simulation {
    pub fn new(
        device: &wgpu::Device,
        initial_cells: &[u32],
        params: SimParams,
    ) -> Self {
        let width = params.width;
        let height = params.height;
        let buf_size = (width * height) as usize * std::mem::size_of::<u32>();

        // Create storage buffers
        let storage_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Buffer A"),
            contents: bytemuck::cast_slice(initial_cells),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let storage_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid Buffer B"),
            size: buf_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Params uniform
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sim Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/compute.wgsl").into(),
            ),
        });

        // Bind group layout
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Two bind groups for ping-pong
        let bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute BG A→B"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: storage_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: storage_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });

        let bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute BG B→A"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: storage_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: storage_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            compute_pipeline,
            bind_groups: [bind_group_a, bind_group_b],
            storage_buffers: [storage_a, storage_b],
            params_buffer,
            width,
            height,
            current: 0,
            generation: 0,
        }
    }

    /// Advance the simulation by one step. Encodes a compute pass into `encoder`.
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let workgroups_x = (self.width + 15) / 16;
        let workgroups_y = (self.height + 15) / 16;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Game of Life Step"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &self.bind_groups[self.current], &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.current = 1 - self.current;
        self.generation += 1;
    }

    /// Returns the storage buffer that currently holds the latest grid state.
    pub fn current_buffer(&self) -> &wgpu::Buffer {
        // After a step, `current` points to the buffer that was just written.
        // The output buffer from the last step is at index (1 - current) before swap,
        // but we already swapped, so current is now the output.
        &self.storage_buffers[1 - self.current]
    }

    /// Upload new cell data to the grid (resets to generation 0).
    pub fn upload(&mut self, queue: &wgpu::Queue, cells: &[u32], params: SimParams) {
        queue.write_buffer(&self.storage_buffers[0], 0, bytemuck::cast_slice(cells));
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        self.current = 0;
        self.generation = 0;
    }

    /// Update just the rules (birth/survival bitmasks).
    pub fn update_rules(&self, queue: &wgpu::Queue, params: SimParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}
