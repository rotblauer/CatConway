# Metrics Guide: Finding Interesting Groups of Rules

This guide provides recommendations on which metrics to plot when exploring and classifying cellular automaton rules in CatConway. The goal is to identify the most useful metric combinations for discovering interesting behavioral groups.

## Quick Start: Recommended Metric Pairs

For **initial exploration**, these metric pairs provide the best separation of interesting rule groups:

1. **variation × mean_density** (default) — Best overall separation
   - X-axis: `mean_density` (average fraction of live cells)
   - Y-axis: `variation` (coefficient of variation)
   - Why: Separates chaotic (high variation) from stable (low variation) rules, while density distinguishes sparse vs. dense dynamics

2. **langton_lambda × variation** — Theory-driven classification
   - X-axis: `langton_lambda` (fraction of rule-table entries mapping to alive)
   - Y-axis: `variation`
   - Why: Langton's λ parameter predicts phase transitions; complex behavior concentrates near λ ≈ 0.3

3. **spatial_entropy × activity** — Spatial vs. temporal complexity
   - X-axis: `activity` (mean fraction of cells changing state)
   - Y-axis: `spatial_entropy` (Shannon entropy of 2×2 block patterns)
   - Why: Separates spatially ordered (low entropy) from disordered (high entropy) dynamics

## All Available Metrics

CatConway computes **14 behavioral metrics** for each rule:

| Metric | Type | Range | Description | Best For |
|--------|------|-------|-------------|----------|
| **variation** | Temporal | 0.0–∞ | Coefficient of variation (σ/μ) of population | Identifying chaotic vs. stable dynamics |
| **mean_density** | Spatial | 0.0–1.0 | Average fraction of live cells | Distinguishing sparse vs. dense rules |
| **final_density** | Spatial | 0.0–1.0 | Density at end of simulation | Detecting convergence behavior |
| **density_range** | Temporal | 0.0–1.0 | Max − min density over time | Measuring amplitude of oscillations |
| **trend** | Temporal | −∞–∞ | Linear slope of density (growing vs. declining) | Finding monotonic growth/decay |
| **autocorrelation** | Temporal | −1.0–1.0 | Lag-1 temporal correlation | Measuring short-term memory |
| **entropy** | Statistical | 0.0–∞ | Shannon entropy of population histogram | Quantifying unpredictability |
| **dominant_period** | Temporal | 0–max | Detected cycle length (0 = aperiodic) | Identifying oscillators |
| **monotonic_fraction** | Temporal | 0.0–1.0 | Fraction of steps with consistent direction | Detecting directional trends |
| **roughness** | Temporal | 0.0–∞ | Mean absolute consecutive density difference | Measuring temporal texture |
| **langton_lambda** | Theoretical | 0.0–1.0 | Fraction of rule entries → alive state | Predicting complexity (edge of chaos) |
| **activity** | Temporal | 0.0–1.0 | Mean fraction of cells changing state/gen | Measuring dynamism |
| **spatial_entropy** | Spatial | 0.0–∞ | Shannon entropy of 2×2 block patterns | Quantifying spatial disorder |
| **damage_spreading** | Sensitivity | −∞–∞ | Hamming distance growth from perturbation | Detecting chaotic sensitivity |

## Metric Pair Recommendations by Goal

### Goal: Separate Wolfram's Four Classes

Wolfram classified CA into: fixed points, periodic orbits, chaotic dynamics, and complex/undecidable behavior.

**Recommended pairs:**
- `variation × dominant_period` — Separates periodic (period > 0) from aperiodic (period = 0)
- `activity × variation` — High activity + high variation = chaos; low activity = fixed/static

### Goal: Find Edge-of-Chaos Rules (Langton's Hypothesis)

Langton proposed that complex computation emerges near a critical λ value (~0.3) between order and chaos.

**Recommended pairs:**
- `langton_lambda × variation` — Look for high variation near λ ≈ 0.3
- `langton_lambda × spatial_entropy` — Complex rules have moderate λ and high spatial entropy
- `langton_lambda × damage_spreading` — Edge of chaos shows moderate damage spreading

### Goal: Distinguish Spatial vs. Temporal Complexity

Some rules are spatially complex but temporally periodic, and vice versa.

**Recommended pairs:**
- `spatial_entropy × variation` — High both = spatio-temporal chaos; high spatial only = complex patterns
- `spatial_entropy × dominant_period` — Separates periodic spatial patterns from aperiodic
- `roughness × spatial_entropy` — Temporal texture vs. spatial disorder

### Goal: Identify Self-Organizing/Growing Rules

Rules that exhibit monotonic growth or organized expansion.

**Recommended pairs:**
- `trend × mean_density` — Positive trend + increasing density = growth
- `monotonic_fraction × trend` — High monotonic + positive trend = consistent growth
- `trend × activity` — Active growth vs. passive convergence

### Goal: Detect Sensitive Dependence (Chaos)

Chaotic rules amplify small perturbations exponentially.

**Recommended pairs:**
- `damage_spreading × variation` — Both high = chaos
- `damage_spreading × autocorrelation` — Chaos has high spreading + low autocorrelation
- `entropy × damage_spreading` — Unpredictability + sensitivity = chaos

## Three-Metric Exploration (Advanced)

While the UI currently supports 2D scatter plots, here are recommended **triples** for future 3D visualization or sequential filtering:

1. **λ × variation × spatial_entropy** — Complete Langton hypothesis + spatial structure
2. **activity × spatial_entropy × damage_spreading** — Temporal, spatial, and chaotic dimensions
3. **mean_density × variation × trend** — Density, stability, and direction
4. **autocorrelation × roughness × variation** — Memory, texture, and chaos
5. **dominant_period × variation × activity** — Periodicity, stability, and dynamism

## Clustering Recommendations

When using k-means clustering on the metric space:

**Suggested k values:**
- **k=8** — Matches the 8 behavior classes (Dead, Static, Periodic, Explosive, Chaotic, Complex, Declining, Growing)
- **k=4** — Rough correspondence to Wolfram's classification
- **k=16** — Finer-grained sub-clusters within behavior classes

**Most discriminative metrics for clustering:**
1. `variation` — Primary separator of dynamics
2. `mean_density` — Primary separator of spatial occupancy
3. `langton_lambda` — Theoretical predictor
4. `spatial_entropy` — Spatial structure
5. `activity` — Temporal dynamism

**Feature scaling:** All metrics are automatically normalized to [0,1] range before clustering to prevent dominance by high-magnitude features.

## UMAP Dimensionality Reduction

UMAP projects the 14D metric space to 2D while preserving cluster structure. It often reveals natural groupings not visible in 2D metric scatter plots.

**Best practice:**
1. First explore with **recommended 2D metric pairs** to understand individual metric relationships
2. Then use **UMAP projection** to discover hidden clusters in the full 14D space
3. Cross-reference UMAP clusters with specific metric pairs to interpret what makes each cluster unique

## Practical Workflow

1. **Start broad:** Use `variation × mean_density` to get an overview
2. **Apply theory:** Check `langton_lambda × variation` to test edge-of-chaos hypothesis
3. **Zoom in:** Filter by behavior class, then explore class-specific metric pairs
4. **Discover patterns:** Use UMAP to find unexpected clusters
5. **Interpret:** Return to specific 2D metric pairs to understand why UMAP grouped certain rules

## References

- **Wolfram, S.** (1984). *Universality and complexity in cellular automata.* Physica D.
- **Langton, C.G.** (1990). *Computation at the edge of chaos.* Physica D.
- **Bagnoli, F., et al.** (1992). *Damage spreading and Lyapunov exponents in cellular automata.* Physics Letters A.
- **Crutchfield, J.P. & Hanson, J.E.** (1993). *Turbulent pattern bases for cellular automata.* Physica D.

---

**Tip:** The classification system is imperfect — use it as a starting point, not ground truth. Manually inspect rules that fall on class boundaries or cluster centroids; these often exhibit the most interesting hybrid behaviors.
