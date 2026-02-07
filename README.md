# CatConway

GPU-accelerated Conway's Game of Life with a dynamical systems focus, built in Rust.

## Features

- **GPU Compute Simulation**: Runs the Game of Life entirely on the GPU using wgpu compute shaders, enabling large grid simulations (1024×1024 default)
- **Real-time Rendering**: GPU-powered fullscreen rendering with camera-based pan/zoom exploration
- **Configurable Rule Sets**: Explore different cellular automata dynamics beyond standard Conway:
  - **Conway B3/S23** — Classic Game of Life
  - **HighLife B36/S23** — Known for replicator patterns
  - **Day & Night B3678/S34678** — Symmetric under on/off inversion
  - **Seeds B2/S** — Explosive growth with no survival
  - **Life without Death B3/S\*** — Cells never die
- **Predefined Patterns**: Load classic patterns to study dynamical behavior:
  - Glider, R-pentomino, Acorn, Gosper Glider Gun, LWSS
- **Toroidal Topology**: Grid wraps at edges for true dynamical system behavior
- **Interactive Exploration**: Pan, zoom, pause, step, and adjust speed in real time
- **Static Binary**: Compiles to a single binary with `cargo build --release`

## Building

```bash
# Debug build
cargo build

# Optimized release build (static binary with LTO)
cargo build --release
```

### System Requirements

- Rust 1.85+ (edition 2024)
- GPU with Vulkan, Metal, or DX12 support
- Linux: `libxkbcommon-dev`, `libwayland-dev` (for Wayland), X11 dev libraries

On Ubuntu/Debian:
```bash
sudo apt-get install libxkbcommon-dev libwayland-dev
```

## Running

```bash
# Run with default settings
cargo run --release

# Enable debug logging
RUST_LOG=info cargo run --release
```

## Controls

| Key | Action |
|-----|--------|
| **Space** | Pause / Resume simulation |
| **→** (Right Arrow) | Step one generation (when paused) |
| **↑ / ↓** | Speed up / slow down |
| **Mouse Drag** | Pan the view |
| **Scroll Wheel** | Zoom in / out |
| **H** | Reset camera to default view |
| **R** | Randomize grid |
| **C** | Clear grid |
| **N** | Cycle through rule sets |
| **1** | Load Glider pattern |
| **2** | Load R-pentomino (methuselah) |
| **3** | Load Acorn (methuselah) |
| **4** | Load Gosper Glider Gun |
| **5** | Load LWSS (Lightweight Spaceship) |
| **Escape** | Quit |

## Testing

Run the unit tests (including the lightweight UI/state tests) with:

```bash
cargo test --all-targets --all-features
```

## CI and Releases

- GitHub Actions runs the test suite on pushes to `main` and on pull requests.
- Tagging a commit with `v*` (for example `v0.1.0`) triggers a release workflow that builds and uploads binaries for:
  - Linux (x86_64-unknown-linux-gnu)
  - macOS (aarch64-apple-darwin)
  - Windows (x86_64-pc-windows-msvc)

Release artifacts contain the compiled `catconway` binary packaged per-platform.

## Architecture

```
src/
  main.rs        — Entry point and event loop
  app.rs         — Application state, GPU init, input handling
  simulation.rs  — GPU compute pipeline (double-buffered ping-pong)
  renderer.rs    — GPU render pipeline (fullscreen triangle)
  camera.rs      — Camera with pan/zoom for exploration
  grid.rs        — Grid state, rules, and pattern definitions
shaders/
  compute.wgsl   — WGSL compute shader for simulation
  render.wgsl    — WGSL fragment shader for visualization
```

The simulation uses double-buffered GPU storage buffers with a ping-pong strategy: each generation, the compute shader reads from one buffer and writes to the other, then they swap roles. This allows the simulation to run entirely on the GPU without CPU↔GPU data transfer during normal operation.

## Dynamical Systems Focus

This implementation treats Conway's Game of Life as a discrete dynamical system:

- **Configurable rules** allow exploration of different automata in the same outer-totalistic family
- **Large grids** (1024×1024) provide sufficient state space for complex emergent behavior
- **Toroidal boundary conditions** create a closed system suitable for long-term dynamics study
- **Methuselah patterns** (R-pentomino, Acorn) demonstrate sensitivity to initial conditions
- **Generation counter** and speed controls enable precise observation of temporal evolution
