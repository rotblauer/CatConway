# CatConway

**Systematic exploration of cellular automata rule space**, powered by GPU-accelerated simulation and real-time visualization. Built in Rust.

CatConway goes beyond simulating Conway's Game of Life. It is designed to **search across the full space of outer-totalistic cellular automata rules**, automatically identify subspaces that produce interesting dynamics, and classify discovered rules by their emergent behavior. Think of it as a telescope for exploring the universe of 2D cellular automata.

## Background and Motivation

Cellular automata (CA) define simple local rules on a grid, yet produce astonishingly varied global behavior — from static equilibria to chaotic turbulence. A natural question is: **which rules are interesting, and how do they relate to one another?**

Prior work on this question spans decades:

- **Wolfram's classification (1984)** — Stephen Wolfram systematically surveyed elementary (1D, radius-1) cellular automata and proposed four qualitative behavior classes: fixed points, periodic orbits, chaotic dynamics, and complex/undecidable behavior. This showed that even the simplest rule spaces contain rich structure waiting to be mapped.
- **Langton's λ parameter (1990)** — Chris Langton introduced a single scalar parameter (the fraction of rule-table entries that map to a non-quiescent state) and demonstrated a phase transition between ordered and chaotic dynamics near a critical λ value — the so-called *edge of chaos* where complex behavior concentrates.
- **Life-like cellular automata** — The two-state, outer-totalistic family (where a cell's fate depends only on its neighbor count) yields 2¹⁸ = 262,144 distinct rules for the standard Moore neighborhood alone. Extending to radius-2 neighborhoods (24 neighbors) expands the space to over 10¹⁵ rules. Only a handful of these (Conway's B3/S23, HighLife's B36/S23, Day & Night, etc.) have been studied in depth; the vast majority remain unexplored.
- **Automated search and metrics** — Researchers have used population statistics, entropy measures, Lyapunov exponents, and compression-based complexity to screen candidate rules. CatConway builds on this tradition by combining multiple dynamical metrics with dual-level periodicity detection and behavior classification.

CatConway's goal is to make this kind of systematic rule-space exploration **accessible and interactive**: discover, classify, visualize, and compare cellular automata rules in real time.

## What This Repo Can Do

### Systematic Rule-Space Search
- **Enumerate all possible birth/survival rule combinations** for radius-1 (262K rules) and radius-2 (over 10¹⁵ rules) neighborhoods
- **Evaluate each candidate** by running a CPU simulation and computing dynamical metrics
- **Filter out uninteresting rules** (dead, saturated, or periodic) using coefficient-of-variation thresholds and dual-level periodicity detection (both population-level and grid-state-level)
- **Persist results across sessions** — examined rules are tracked in `search_examined.txt` so work is never repeated; interesting rules are saved to `search_results.txt`

### Behavior Classification
- **Compute 10 dynamical metrics** per rule: coefficient of variation, mean/final density, density range, linear trend, lag-1 autocorrelation, Shannon entropy, dominant period, monotonic fraction, and roughness
- **Classify rules into 8 behavior classes**: Dead, Static, Periodic, Explosive, Chaotic, Complex, Growing, and Declining
- **Cluster and visualize** discovered rules using k-means clustering and UMAP dimensionality reduction, displayed as interactive scatter plots in the GUI

### GPU-Accelerated Simulation and Visualization
- **Run simulations entirely on the GPU** using wgpu compute shaders with double-buffered ping-pong architecture — no CPU↔GPU transfer during normal operation
- **Large grids** (1024×1024 default) provide sufficient state space for complex emergent behavior
- **Real-time rendering** with camera-based pan/zoom exploration
- **Toroidal topology** — grid wraps at edges, creating a closed dynamical system without boundary artifacts

### Interactive Exploration
- **8 built-in rule sets** spanning different dynamical regimes:
  - **Conway B3/S23** — Classic Game of Life (chaotic/complex)
  - **HighLife B36/S23** — Known for self-replicating patterns
  - **Day & Night B3678/S34678** — Symmetric under on/off inversion
  - **Seeds B2/S** — Explosive growth with no survival
  - **Life without Death B3/S\*** — Monotone growth (cells never die)
  - **Bugs B3-5/S4-8/R2** — Amoeba-like blob dynamics (radius-2)
  - **Globe B7-11/S7-11/R2** — Stable growing masses (radius-2)
  - **Majority B5-8/S4-10/R2** — Majority-vote organic boundaries (radius-2)
- **Predefined patterns** for studying sensitivity to initial conditions: Glider, R-pentomino, Acorn, Gosper Glider Gun, LWSS
- **Apply any discovered rule** directly to the live GPU simulation from the search results panel
- **Real-time statistics**: population and density plots, generation rate tracking
- **Favorites system**: bookmark interesting rules and export GIF animations
- **Static binary**: compiles to a single binary with `cargo build --release`

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
  search.rs      — Background rule search (CPU-based auto-discovery)
  ui.rs          — egui overlay UI (sidebar, stats, rule search controls)
  stats.rs       — Population statistics and sampling
shaders/
  compute.wgsl   — WGSL compute shader for simulation
  render.wgsl    — WGSL fragment shader for visualization
```

The simulation uses double-buffered GPU storage buffers with a ping-pong strategy: each generation, the compute shader reads from one buffer and writes to the other, then they swap roles. This allows the simulation to run entirely on the GPU without CPU↔GPU data transfer during normal operation.

## Dynamical Systems Perspective

CatConway treats cellular automata as discrete dynamical systems and provides tools to study them as such:

- **Configurable outer-totalistic rules** allow systematic exploration of the same family studied by Wolfram, Langton, and others — where a cell's fate depends only on its live neighbor count
- **Large grids** (1024×1024) provide sufficient state space for complex emergent behavior to develop without finite-size artifacts
- **Toroidal boundary conditions** create a closed system (compact phase space) suitable for long-term dynamics study, matching the standard mathematical formulation
- **Methuselah patterns** (R-pentomino: 1,103-generation transient; Acorn: 5,206-generation transient) demonstrate sensitivity to initial conditions — a hallmark of chaotic dynamics
- **Automated search and classification** maps the structure of rule space, identifying phase transitions between ordered and chaotic regimes analogous to Langton's edge-of-chaos phenomenon
- **Real-time population and density tracking** enables observation of attractors, transients, and bifurcations as rules are varied

## Rule Search

CatConway includes a background rule search engine that systematically explores the space of outer-totalistic cellular automata rules. The search runs on a separate CPU thread so the GPU simulation remains fully responsive.

### Search Methodology

Each candidate rule set (a pair of birth and survival bitmasks at a given neighborhood radius) is evaluated through the following pipeline:

1. **CPU simulation** — A short simulation is run on a 640×640 grid with 25% random initial density for 3,000 generations. This mirrors the GPU compute logic (toroidal boundaries, identical neighbor-counting) but runs on the CPU to avoid blocking the visualizer.

2. **Population trace analysis** — The population count is recorded at each generation. Statistics are computed over the **second half** of the trace (post burn-in) to focus on steady-state behavior rather than transients.

3. **Coefficient of variation (CV)** — The primary interestingness metric: `CV = σ / μ` over the post-burn-in population trace. Rules with `CV < 0.05` (configurable) are rejected as too stable or trivial.

4. **Dual-level periodicity detection** — Two complementary checks reject periodic behavior:
   - **Population-level**: detects repeating population counts with period ≤ 20
   - **Grid-state-level**: detects repeating grid configurations, which catches **complement oscillations** (alive↔dead flips that produce identical population counts but different spatial patterns)

5. **Early exit** — Rules where the population drops to zero are immediately rejected.

6. **Persistence** — Interesting rules (those passing all filters) are written to `search_results.txt`. All examined rules are recorded in `search_examined.txt` to avoid redundant evaluation across sessions.

### Using Rule Search from the GUI

1. **Start**: In the left sidebar, scroll to the **Rule Search** section and click **🔍 Start Search**. The search begins iterating over all possible birth/survival rule combinations.
2. **Monitor progress**: While running, the sidebar shows:
   - **Status** — Running, Paused, or Complete
   - **Rules tested** — total number of rule sets evaluated so far
   - **Interesting** — how many rule sets passed the interestingness filter
3. **Pause / Resume**: Click **⏸ Pause** to temporarily halt the search (the thread sleeps until resumed). Click **▶ Resume** to continue where it left off.
4. **Stop**: Click **⏹ Stop** to terminate the search. You can start a new search at any time.
5. **Apply a discovered rule**: Interesting rules appear in a scrollable list below the controls. Click **Apply** next to any rule label (e.g., `B36/S23`) to immediately load that rule set into the live GPU simulation.

### Rule Space Coverage

| Radius | Neighbors | Possible Rules | Description |
|--------|-----------|----------------|-------------|
| 1 | 8 (Moore) | 2⁹ × 2⁹ = 262,144 | Standard Life-like automata |
| 2 | 24 | 2²⁵ × 2²⁵ ≈ 1.1 × 10¹⁵ | Extended neighborhood ("Larger than Life") |

## Behavior Classification

Beyond binary interesting/uninteresting filtering, CatConway can classify discovered rules into **8 behavior classes** based on 10 dynamical metrics:

### Dynamical Metrics (per rule)

| Metric | Description |
|--------|-------------|
| Coefficient of variation | Population variability (σ/μ) over steady state |
| Mean density | Average fraction of live cells |
| Final density | Density at end of evaluation |
| Density range | Max − min density over the observation window |
| Trend | Linear slope of density (growing vs. declining) |
| Autocorrelation | Lag-1 temporal correlation (memory/inertia) |
| Shannon entropy | Information content of population histogram |
| Dominant period | Detected cycle length (0 = aperiodic) |
| Monotonic fraction | Fraction of steps with consistent direction of change |
| Roughness | Mean consecutive density difference (temporal texture) |

### Behavior Classes

| Class | Criteria | Examples |
|-------|----------|---------|
| **Dead** | Final density < 0.1% | Rules where all cells die out |
| **Static** | Very low variation, near-zero trend | Still lifes, stable configurations |
| **Periodic** | Dominant period detected, low variation | Blinkers, oscillators |
| **Explosive** | Mean density > 90% | Rules that fill the grid |
| **Chaotic** | High variation (>10%), aperiodic | Rules with turbulent, unpredictable dynamics |
| **Complex** | Moderate variation, no dominant period | Edge-of-chaos rules (most interesting) |
| **Growing** | Positive density trend, not yet explosive | Rules with monotone population increase |
| **Declining** | Negative density trend | Rules with gradual population decay |

The classification system uses k-means clustering and UMAP dimensionality reduction to visualize the distribution of rules across behavior space as an interactive scatter plot in the GUI.
