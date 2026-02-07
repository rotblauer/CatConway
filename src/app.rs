use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::camera::Camera;
use crate::grid::{
    Grid, Rules, pattern_acorn, pattern_glider, pattern_gosper_gun, pattern_lwss,
    pattern_r_pentomino,
};
use crate::renderer::Renderer;
use crate::simulation::Simulation;

/// Default grid dimension (square).
const GRID_SIZE: u32 = 1024;

/// Initial random fill density.
const INITIAL_DENSITY: f64 = 0.25;

/// Target simulation steps per second at speed=1.
const BASE_SPEED: f64 = 15.0;

/// Application state managing the simulation, rendering, and user interaction.
pub struct App {
    /// GPU resources (initialized after window creation).
    gpu: Option<GpuState>,
    /// Current camera.
    camera: Camera,
    /// Grid state (CPU side, used for initialization).
    grid: Grid,
    /// Whether the simulation is running.
    running: bool,
    /// Simulation speed multiplier.
    speed: f64,
    /// Time accumulator for stepping.
    step_accumulator: f64,
    last_frame: Instant,
    /// Current rule set.
    current_rule_idx: usize,
    /// Mouse drag state.
    dragging: bool,
    last_mouse_pos: Option<(f64, f64)>,
}

struct GpuState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    simulation: Simulation,
    renderer: Renderer,
}

/// The available rule sets for exploration.
fn rule_sets() -> Vec<(&'static str, Rules)> {
    vec![
        ("Conway B3/S23", Rules::conway()),
        ("HighLife B36/S23", Rules::highlife()),
        ("Day&Night B3678/S34678", Rules::day_and_night()),
        ("Seeds B2/S", Rules::seeds()),
        ("Life w/o Death B3/S*", Rules::life_without_death()),
    ]
}

impl App {
    pub fn new() -> Self {
        let mut grid = Grid::new(GRID_SIZE, GRID_SIZE);
        grid.randomize(INITIAL_DENSITY);

        Self {
            gpu: None,
            camera: Camera::new(),
            grid,
            running: true,
            speed: 1.0,
            step_accumulator: 0.0,
            last_frame: Instant::now(),
            current_rule_idx: 0,
            dragging: false,
            last_mouse_pos: None,
        }
    }

    fn initialize_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).expect("Failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("No suitable GPU adapter found");

        log::info!("GPU adapter: {:?}", adapter.get_info().name);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
            None,
        ))
        .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        self.camera.aspect = config.width as f32 / config.height as f32;

        let simulation = Simulation::new(&device, &self.grid.cells, self.grid.sim_params());
        let renderer = Renderer::new(&device, surface_format);

        self.gpu = Some(GpuState {
            window,
            surface,
            device,
            queue,
            config,
            simulation,
            renderer,
        });
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if let Some(ref mut gpu) = self.gpu {
            if new_size.width > 0 && new_size.height > 0 {
                gpu.config.width = new_size.width;
                gpu.config.height = new_size.height;
                gpu.surface.configure(&gpu.device, &gpu.config);
                self.camera.aspect = new_size.width as f32 / new_size.height as f32;
            }
        }
    }

    fn render_frame(&mut self) {
        let Some(ref mut gpu) = self.gpu else { return };

        // Timing
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f64();
        self.last_frame = now;

        // Step simulation
        if self.running {
            self.step_accumulator += dt * BASE_SPEED * self.speed;
            let steps = self.step_accumulator as u32;
            self.step_accumulator -= steps as f64;

            if steps > 0 {
                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Compute Encoder"),
                        });
                for _ in 0..steps.min(10) {
                    gpu.simulation.step(&mut encoder);
                }
                gpu.queue.submit(std::iter::once(encoder.finish()));
            }
        }

        // Render
        let output = match gpu.surface.get_current_texture() {
            Ok(tex) => tex,
            Err(wgpu::SurfaceError::Lost) => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                return;
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                log::error!("Out of GPU memory");
                return;
            }
            Err(e) => {
                log::warn!("Surface error: {e:?}");
                return;
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Update camera uniform
        let cam_uniform = self.camera.uniform(self.grid.width, self.grid.height);
        gpu.renderer.update_camera(&gpu.queue, &cam_uniform);

        // Update render bind group to point to the current simulation buffer
        gpu.renderer
            .update_bind_group(&gpu.device, gpu.simulation.current_buffer());

        let mut encoder =
            gpu.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        gpu.renderer.render(&mut encoder, &view);
        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        // Update window title with stats
        let generation = gpu.simulation.generation;
        let rules = rule_sets();
        let rule_name = rules[self.current_rule_idx].0;
        let status = if self.running { "▶" } else { "⏸" };
        gpu.window.set_title(&format!(
            "CatConway | {status} Gen {generation} | {rule_name} | Speed: {:.1}x | {GRID_SIZE}×{GRID_SIZE}",
            self.speed,
        ));

        // Request next frame
        gpu.window.request_redraw();
    }

    fn handle_key(&mut self, event: KeyEvent) {
        if event.state != ElementState::Pressed {
            return;
        }

        match event.logical_key {
            Key::Named(NamedKey::Space) => {
                self.running = !self.running;
                log::info!("Simulation {}", if self.running { "resumed" } else { "paused" });
            }
            Key::Named(NamedKey::ArrowRight) if !self.running => {
                // Single step
                if let Some(ref mut gpu) = self.gpu {
                    let mut encoder =
                        gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Step Encoder"),
                        });
                    gpu.simulation.step(&mut encoder);
                    gpu.queue.submit(std::iter::once(encoder.finish()));
                }
            }
            Key::Named(NamedKey::ArrowUp) => {
                self.speed = (self.speed * 1.5).min(100.0);
                log::info!("Speed: {:.1}x", self.speed);
            }
            Key::Named(NamedKey::ArrowDown) => {
                self.speed = (self.speed / 1.5).max(0.1);
                log::info!("Speed: {:.1}x", self.speed);
            }
            Key::Named(NamedKey::Escape) => {
                if let Some(ref gpu) = self.gpu {
                    gpu.window.set_visible(false);
                }
                std::process::exit(0);
            }
            Key::Character(ref c) => match c.as_str() {
                "r" => {
                    self.grid.randomize(INITIAL_DENSITY);
                    self.upload_grid();
                    log::info!("Grid randomized");
                }
                "c" => {
                    self.grid.clear();
                    self.upload_grid();
                    log::info!("Grid cleared");
                }
                "h" => {
                    self.camera.reset();
                    log::info!("Camera reset");
                }
                "1" => self.load_pattern("Glider", &pattern_glider()),
                "2" => self.load_pattern("R-pentomino", &pattern_r_pentomino()),
                "3" => self.load_pattern("Acorn", &pattern_acorn()),
                "4" => self.load_pattern("Gosper Gun", &pattern_gosper_gun()),
                "5" => self.load_pattern("LWSS", &pattern_lwss()),
                "n" => {
                    let rules = rule_sets();
                    self.current_rule_idx = (self.current_rule_idx + 1) % rules.len();
                    self.grid.rules = rules[self.current_rule_idx].1;
                    if let Some(ref gpu) = self.gpu {
                        gpu.simulation.update_rules(&gpu.queue, self.grid.sim_params());
                    }
                    log::info!("Rules: {}", rules[self.current_rule_idx].0);
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn load_pattern(&mut self, name: &str, pattern: &[(i32, i32)]) {
        self.grid.clear();
        self.grid.place_pattern(pattern, None);
        self.upload_grid();
        log::info!("Loaded pattern: {name}");
    }

    fn upload_grid(&mut self) {
        if let Some(ref mut gpu) = self.gpu {
            gpu.simulation
                .upload(&gpu.queue, &self.grid.cells, self.grid.sim_params());
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            let attrs = WindowAttributes::default()
                .with_title("CatConway - GPU Accelerated Game of Life")
                .with_inner_size(PhysicalSize::new(1024u32, 768));

            let window = Arc::new(
                event_loop
                    .create_window(attrs)
                    .expect("Failed to create window"),
            );

            self.initialize_gpu(window);
            self.last_frame = Instant::now();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                self.resize(size);
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_key(event);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(pos) => pos.y / 50.0,
                };
                let factor = if scroll > 0.0 { 0.9 } else { 1.1 };
                self.camera.zoom(factor as f32);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.dragging = state == ElementState::Pressed;
                    if !self.dragging {
                        self.last_mouse_pos = None;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.dragging {
                    if let Some((lx, ly)) = self.last_mouse_pos {
                        if let Some(ref gpu) = self.gpu {
                            let w = gpu.config.width as f64;
                            let h = gpu.config.height as f64;
                            let dx = (lx - position.x) / w;
                            let dy = (ly - position.y) / h;
                            self.camera.pan(dx as f32, dy as f32);
                        }
                    }
                    self.last_mouse_pos = Some((position.x, position.y));
                }
            }
            _ => {}
        }
    }
}
