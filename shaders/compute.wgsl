// Conway's Game of Life - GPU Compute Shader
// Supports configurable birth/survival rules via bitmasks for dynamical systems exploration.
// Each workgroup thread processes one cell.

struct SimParams {
    width: u32,
    height: u32,
    birth_rule: u32,    // bitmask: bit i set means a dead cell with i neighbors becomes alive
    survival_rule: u32, // bitmask: bit i set means an alive cell with i neighbors survives
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    let w = params.width;
    let h = params.height;
    let idx = y * w + x;

    // Count live neighbors with toroidal wrapping
    var neighbors: u32 = 0u;
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            let nx = u32((i32(x) + dx + i32(w)) % i32(w));
            let ny = u32((i32(y) + dy + i32(h)) % i32(h));
            neighbors += input[ny * w + nx];
        }
    }

    let alive = input[idx];
    let mask = 1u << neighbors;

    if (alive == 1u) {
        output[idx] = select(0u, 1u, (params.survival_rule & mask) != 0u);
    } else {
        output[idx] = select(0u, 1u, (params.birth_rule & mask) != 0u);
    }
}
