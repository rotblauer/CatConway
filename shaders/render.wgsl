// Conway's Game of Life - Grid Visualization Shader
// Renders the grid as a fullscreen pass with camera-based pan/zoom.

struct CameraUniform {
    offset_x: f32,
    offset_y: f32,
    scale: f32,
    aspect: f32,
    grid_width: f32,
    grid_height: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<storage, read> grid: array<u32>;
@group(0) @binding(1) var<uniform> camera: CameraUniform;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    // Fullscreen triangle (covers entire screen with one triangle)
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = (positions[idx] + 1.0) * 0.5;
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Map UV to world coordinates using camera
    let world_x = (in.uv.x - 0.5) * camera.scale * camera.aspect + camera.offset_x;
    let world_y = (in.uv.y - 0.5) * camera.scale + camera.offset_y;

    // Convert world coords (0..1 range) to grid coords
    let gx = i32(floor(world_x * camera.grid_width));
    let gy = i32(floor(world_y * camera.grid_height));

    // Draw grid lines at cell boundaries when zoomed in
    let cell_size_px = 1.0 / (camera.scale * camera.grid_width);
    let frac_x = fract(world_x * camera.grid_width);
    let frac_y = fract(world_y * camera.grid_height);
    let grid_line = cell_size_px > 4.0 && (frac_x < 0.05 || frac_y < 0.05);

    // Outside grid bounds
    if (gx < 0 || gx >= i32(camera.grid_width) || gy < 0 || gy >= i32(camera.grid_height)) {
        return vec4<f32>(0.04, 0.04, 0.06, 1.0);
    }

    let cell_idx = u32(gy) * u32(camera.grid_width) + u32(gx);
    let alive = grid[cell_idx];

    if (grid_line) {
        if (alive == 1u) {
            return vec4<f32>(0.55, 0.42, 0.22, 1.0);
        } else {
            return vec4<f32>(0.07, 0.07, 0.10, 1.0);
        }
    }

    if (alive == 1u) {
        return vec4<f32>(0.85, 0.65, 0.30, 1.0);
    } else {
        return vec4<f32>(0.03, 0.03, 0.05, 1.0);
    }
}
