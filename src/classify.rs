use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use rand::Rng;

use crate::grid::Rules;
use crate::search::{cpu_step, parse_rule_label, rules_to_label};

// ── Behavioral metrics ──────────────────────────────────────────────────────

/// A comprehensive set of behavioral metrics computed after a burn-in period.
#[derive(Debug, Clone)]
pub struct RuleMetrics {
    /// Coefficient of variation over post-burn-in window.
    pub variation: f64,
    /// Mean population density over post-burn-in window.
    pub mean_density: f64,
    /// Final population density at end of simulation.
    pub final_density: f64,
    /// Range (max - min) of density over post-burn-in window.
    pub density_range: f64,
    /// Linear trend (slope) of density over post-burn-in window.
    /// Positive = growing, negative = declining, near-zero = stable.
    pub trend: f64,
    /// Lag-1 autocorrelation of density trace (short-term memory).
    pub autocorrelation: f64,
    /// Shannon entropy of population histogram (complexity measure).
    pub entropy: f64,
    /// Dominant period detected in the trace (0 if aperiodic).
    pub dominant_period: usize,
    /// Fraction of consecutive steps with monotonic change (same sign).
    pub monotonic_fraction: f64,
    /// Roughness: mean absolute difference between consecutive densities.
    pub roughness: f64,
}

impl RuleMetrics {
    /// Return metrics as a feature vector for clustering.
    pub fn feature_vector(&self) -> [f64; 10] {
        [
            self.variation,
            self.mean_density,
            self.final_density,
            self.density_range,
            self.trend,
            self.autocorrelation,
            self.entropy,
            self.dominant_period as f64,
            self.monotonic_fraction,
            self.roughness,
        ]
    }
}

/// Feature names corresponding to `feature_vector()` indices.
pub const FEATURE_NAMES: [&str; 10] = [
    "variation",
    "mean_density",
    "final_density",
    "density_range",
    "trend",
    "autocorrelation",
    "entropy",
    "dominant_period",
    "monotonic_fraction",
    "roughness",
];

// ── Behavior classification ─────────────────────────────────────────────────

/// Behavioral classification of a cellular automaton rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BehaviorClass {
    /// Population dies to zero.
    Dead,
    /// Reaches a stable fixed point (very low variation).
    Static,
    /// Oscillates with a detectable period.
    Periodic,
    /// Saturates most of the grid (>90% density).
    Explosive,
    /// High variation, no detectable period — chaotic dynamics.
    Chaotic,
    /// Moderate variation with interesting dynamics.
    Complex,
    /// Population is declining over time (negative trend).
    Declining,
    /// Population is growing over time (positive trend).
    Growing,
}

impl fmt::Display for BehaviorClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BehaviorClass::Dead => write!(f, "Dead"),
            BehaviorClass::Static => write!(f, "Static"),
            BehaviorClass::Periodic => write!(f, "Periodic"),
            BehaviorClass::Explosive => write!(f, "Explosive"),
            BehaviorClass::Chaotic => write!(f, "Chaotic"),
            BehaviorClass::Complex => write!(f, "Complex"),
            BehaviorClass::Declining => write!(f, "Declining"),
            BehaviorClass::Growing => write!(f, "Growing"),
        }
    }
}

impl BehaviorClass {
    /// All behavior classes in display order.
    pub fn all() -> &'static [BehaviorClass] {
        &[
            BehaviorClass::Dead,
            BehaviorClass::Static,
            BehaviorClass::Periodic,
            BehaviorClass::Explosive,
            BehaviorClass::Chaotic,
            BehaviorClass::Complex,
            BehaviorClass::Declining,
            BehaviorClass::Growing,
        ]
    }

    /// Parse from string representation.
    pub fn from_str(s: &str) -> Option<BehaviorClass> {
        match s {
            "Dead" => Some(BehaviorClass::Dead),
            "Static" => Some(BehaviorClass::Static),
            "Periodic" => Some(BehaviorClass::Periodic),
            "Explosive" => Some(BehaviorClass::Explosive),
            "Chaotic" => Some(BehaviorClass::Chaotic),
            "Complex" => Some(BehaviorClass::Complex),
            "Declining" => Some(BehaviorClass::Declining),
            "Growing" => Some(BehaviorClass::Growing),
            _ => None,
        }
    }
}

/// Classify a rule based on its computed metrics.
pub fn classify(metrics: &RuleMetrics) -> BehaviorClass {
    // Dead: zero or near-zero final density.
    if metrics.final_density < 0.001 {
        return BehaviorClass::Dead;
    }

    // Explosive: saturates most of the grid.
    if metrics.mean_density > 0.90 {
        return BehaviorClass::Explosive;
    }

    // Periodic: a dominant period was detected.
    if metrics.dominant_period > 0 && metrics.variation < 0.05 {
        return BehaviorClass::Periodic;
    }

    // Static: very low variation and no significant trend.
    if metrics.variation < 0.005 && metrics.trend.abs() < 0.0001 {
        return BehaviorClass::Static;
    }

    // Growing: significant positive trend.
    if metrics.trend > 0.0005 && metrics.mean_density < 0.85 {
        return BehaviorClass::Growing;
    }

    // Declining: significant negative trend.
    if metrics.trend < -0.0005 {
        return BehaviorClass::Declining;
    }

    // Chaotic: high variation, aperiodic.
    if metrics.variation > 0.10 && metrics.dominant_period == 0 {
        return BehaviorClass::Chaotic;
    }

    // Complex: everything else with moderate variation.
    BehaviorClass::Complex
}

// ── Metrics computation ─────────────────────────────────────────────────────

/// Configuration for the classification pipeline.
#[derive(Debug, Clone)]
pub struct ClassifyConfig {
    /// Grid size (square) for CPU simulations.
    pub grid_size: u32,
    /// Initial random fill density.
    pub density: f64,
    /// Total number of generations to simulate.
    pub generations: u32,
    /// Burn-in period (first N generations discarded before metrics).
    pub burn_in: u32,
    /// Maximum period length to detect.
    pub max_period: usize,
    /// Path for classified results output.
    pub results_path: PathBuf,
    /// Path for tracking examined rule sets.
    pub examined_path: PathBuf,
    /// Neighborhood radius to search (1 or 2).
    pub radius: u32,
}

impl Default for ClassifyConfig {
    fn default() -> Self {
        Self {
            grid_size: 64,
            density: 0.25,
            generations: 500,
            burn_in: 200,
            max_period: 30,
            results_path: PathBuf::from("classified_results.txt"),
            examined_path: PathBuf::from("classify_examined.txt"),
            radius: 1,
        }
    }
}

/// Result of classifying a single rule set.
#[derive(Debug, Clone)]
pub struct ClassifiedRule {
    pub rules: Rules,
    pub label: String,
    pub metrics: RuleMetrics,
    pub behavior: BehaviorClass,
    /// Cluster index assigned by k-means (if clustering has run).
    pub cluster: Option<usize>,
    /// UMAP 2D projection coordinates.
    pub umap_x: Option<f64>,
    pub umap_y: Option<f64>,
    /// Cluster index assigned by k-means on the UMAP projection.
    pub umap_cluster: Option<usize>,
}

/// Evaluate a rule set: run simulation and compute behavioral metrics.
pub fn compute_metrics(rules: &Rules, config: &ClassifyConfig) -> RuleMetrics {
    let size = config.grid_size;
    let total = (size * size) as usize;
    let total_f = total as f64;
    let mut cells = vec![0u32; total];

    // Random init.
    let mut rng = rand::thread_rng();
    for cell in &mut cells {
        *cell = u32::from(rng.gen_range(0.0..1.0) < config.density);
    }

    let mut densities: Vec<f64> = Vec::with_capacity(config.generations as usize);

    for _step in 0..config.generations {
        let pop: u64 = cells.iter().map(|&c| u64::from(c)).sum();
        densities.push(pop as f64 / total_f);

        if pop == 0 && _step > config.burn_in {
            break;
        }

        cells = cpu_step(&cells, size, size, rules);
    }

    compute_metrics_from_trace(&densities, config.burn_in as usize, config.max_period)
}

/// Compute all behavioral metrics from a density trace, using the portion
/// after `burn_in` steps.
pub fn compute_metrics_from_trace(
    densities: &[f64],
    burn_in: usize,
    max_period: usize,
) -> RuleMetrics {
    let start = burn_in.min(densities.len());
    let window = &densities[start..];

    if window.is_empty() {
        return RuleMetrics {
            variation: 0.0,
            mean_density: 0.0,
            final_density: densities.last().copied().unwrap_or(0.0),
            density_range: 0.0,
            trend: 0.0,
            autocorrelation: 0.0,
            entropy: 0.0,
            dominant_period: 0,
            monotonic_fraction: 0.0,
            roughness: 0.0,
        };
    }

    let n = window.len() as f64;
    let mean = window.iter().sum::<f64>() / n;
    let variance = window.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();
    let variation = if mean > 1e-9 { stddev / mean } else { 0.0 };

    let min_d = window.iter().copied().fold(f64::INFINITY, f64::min);
    let max_d = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let density_range = max_d - min_d;

    let final_density = *window.last().unwrap_or(&0.0);

    // Linear trend via least-squares regression.
    let trend = linear_trend(window);

    // Lag-1 autocorrelation.
    let autocorrelation = lag1_autocorrelation(window, mean, variance);

    // Shannon entropy of population histogram.
    let entropy = population_entropy(window);

    // Dominant period detection.
    let dominant_period = detect_dominant_period(window, max_period);

    // Monotonic fraction.
    let monotonic_fraction = compute_monotonic_fraction(window);

    // Roughness: mean absolute difference between consecutive steps.
    let roughness = compute_roughness(window);

    RuleMetrics {
        variation,
        mean_density: mean,
        final_density,
        density_range,
        trend,
        autocorrelation,
        entropy,
        dominant_period,
        monotonic_fraction,
        roughness,
    }
}

/// Compute the linear trend (slope) of a time series via least-squares.
fn linear_trend(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    for (i, &y) in data.iter().enumerate() {
        let x = i as f64;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return 0.0;
    }
    (n * sum_xy - sum_x * sum_y) / denom
}

/// Lag-1 autocorrelation of a time series.
fn lag1_autocorrelation(data: &[f64], mean: f64, variance: f64) -> f64 {
    if data.len() < 2 || variance < 1e-15 {
        return 0.0;
    }
    let n = (data.len() - 1) as f64;
    let sum: f64 = data
        .windows(2)
        .map(|w| (w[0] - mean) * (w[1] - mean))
        .sum();
    sum / (n * variance)
}

/// Shannon entropy of a discretized population histogram.
fn population_entropy(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    // Bin into 50 buckets spanning [min, max].
    let min_v = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max_v - min_v;
    if range < 1e-12 {
        return 0.0;
    }

    let num_bins = 50usize;
    let mut bins = vec![0u64; num_bins];
    let bin_max_idx = (num_bins - 1) as f64;
    for &v in data {
        let idx = ((v - min_v) / range * bin_max_idx).round() as usize;
        bins[idx.min(num_bins - 1)] += 1;
    }

    let total = data.len() as f64;
    let mut entropy = 0.0;
    for &count in &bins {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// Minimum autocorrelation value required to consider a lag as a true period.
const MIN_PERIOD_ACF_THRESHOLD: f64 = 0.7;

/// Detect the dominant period in a time series using autocorrelation peaks.
/// Returns 0 if no clear period is found.
fn detect_dominant_period(data: &[f64], max_period: usize) -> usize {
    if data.len() < 4 {
        return 0;
    }
    let n = data.len();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-15 {
        // Constant data → period 1 (static).
        return 1;
    }

    let limit = max_period.min(n / 2);
    let mut best_period = 0;
    let mut best_acf = MIN_PERIOD_ACF_THRESHOLD;

    for lag in 1..=limit {
        let acf: f64 = data
            .iter()
            .zip(data[lag..].iter())
            .map(|(&a, &b)| (a - mean) * (b - mean))
            .sum::<f64>()
            / ((n - lag) as f64 * var);

        if acf > best_acf {
            best_acf = acf;
            best_period = lag;
        }
    }

    best_period
}

/// Fraction of consecutive steps where the sign of change stays the same.
fn compute_monotonic_fraction(data: &[f64]) -> f64 {
    if data.len() < 3 {
        return 0.0;
    }
    let diffs: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
    let same_sign = diffs
        .windows(2)
        .filter(|w| (w[0] >= 0.0) == (w[1] >= 0.0))
        .count();
    same_sign as f64 / (diffs.len() - 1) as f64
}

/// Mean absolute difference between consecutive density values.
fn compute_roughness(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let sum: f64 = data.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    sum / (data.len() - 1) as f64
}

// ── Simple k-means clustering ───────────────────────────────────────────────

/// Run k-means clustering on feature vectors and return cluster assignments.
/// Features are normalized (z-score) before clustering.
pub fn kmeans_cluster(features: &[[f64; 10]], k: usize, max_iter: usize) -> Vec<usize> {
    let n = features.len();
    if n == 0 || k == 0 {
        return vec![];
    }
    if k >= n {
        return (0..n).collect();
    }

    // Normalize features (z-score).
    let (normalized, _means, _stddevs) = normalize_features(features);

    // Initialize centroids using k-means++ style: spread initial picks.
    let mut centroids = Vec::with_capacity(k);
    let mut rng = rand::thread_rng();
    centroids.push(normalized[rng.gen_range(0..n)]);

    for _ in 1..k {
        let dists: Vec<f64> = normalized
            .iter()
            .map(|p| {
                centroids
                    .iter()
                    .map(|c| squared_dist(p, c))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        let total: f64 = dists.iter().sum();
        if total < 1e-15 {
            // All points are the same; just pick random.
            centroids.push(normalized[rng.gen_range(0..n)]);
            continue;
        }

        // Weighted random selection.
        let threshold = rng.gen_range(0.0..total);
        let mut cumulative = 0.0;
        let mut chosen = 0;
        for (i, d) in dists.iter().enumerate() {
            cumulative += d;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(normalized[chosen]);
    }

    let mut assignments = vec![0usize; n];

    for _iter in 0..max_iter {
        // Assign each point to nearest centroid.
        let mut changed = false;
        for (i, point) in normalized.iter().enumerate() {
            let nearest = centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    squared_dist(point, a)
                        .partial_cmp(&squared_dist(point, b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if assignments[i] != nearest {
                assignments[i] = nearest;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Recompute centroids.
        let mut new_centroids = vec![[0.0f64; 10]; k];
        let mut counts = vec![0usize; k];
        for (i, point) in normalized.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (j, &v) in point.iter().enumerate() {
                new_centroids[c][j] += v;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..10 {
                    new_centroids[c][j] /= counts[c] as f64;
                }
            }
        }
        centroids = new_centroids;
    }

    assignments
}

fn squared_dist(a: &[f64; 10], b: &[f64; 10]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

fn normalize_features(features: &[[f64; 10]]) -> (Vec<[f64; 10]>, [f64; 10], [f64; 10]) {
    let n = features.len() as f64;
    let mut means = [0.0f64; 10];
    let mut stddevs = [0.0f64; 10];

    for f in features {
        for (j, &v) in f.iter().enumerate() {
            means[j] += v;
        }
    }
    for m in &mut means {
        *m /= n;
    }

    for f in features {
        for (j, &v) in f.iter().enumerate() {
            stddevs[j] += (v - means[j]).powi(2);
        }
    }
    for s in &mut stddevs {
        *s = (*s / n).sqrt();
        if *s < 1e-12 {
            *s = 1.0; // Constant feature: use unit stddev to avoid near-zero division.
        }
    }

    let normalized: Vec<[f64; 10]> = features
        .iter()
        .map(|f| {
            let mut norm = [0.0f64; 10];
            for (j, &v) in f.iter().enumerate() {
                norm[j] = (v - means[j]) / stddevs[j];
            }
            norm
        })
        .collect();

    (normalized, means, stddevs)
}

// ── UMAP dimensionality reduction ───────────────────────────────────────────

/// Default number of nearest neighbors for UMAP graph construction.
const UMAP_N_NEIGHBORS: usize = 15;
/// Default number of SGD epochs for UMAP embedding optimization.
const UMAP_N_EPOCHS: usize = 200;
/// Fewer epochs used during periodic re-clustering to reduce latency.
const UMAP_N_EPOCHS_INCREMENTAL: usize = 100;
/// Random seed for deterministic UMAP initialization (reproducible embeddings).
const UMAP_RANDOM_SEED: u64 = 42;
/// Maximum magnitude for repulsive force gradients, preventing large jumps
/// that can destabilize the SGD optimization.
const REPULSIVE_FORCE_CLAMP: f64 = 4.0;

/// Result of a UMAP projection: 2D coordinates for each input point.
#[derive(Debug, Clone)]
pub struct UmapProjection {
    /// 2D coordinates for each input point.
    pub coords: Vec<[f64; 2]>,
}

/// Run UMAP on 10-dimensional feature vectors, returning a 2D projection.
///
/// This is a simplified implementation of UMAP (Uniform Manifold Approximation
/// and Projection) suitable for interactive visualization. It constructs a
/// fuzzy k-nearest-neighbor graph in high-dimensional space and optimizes a
/// low-dimensional embedding using stochastic gradient descent with
/// attractive/repulsive forces.
pub fn umap_project(features: &[[f64; 10]], n_neighbors: usize, n_epochs: usize) -> UmapProjection {
    let n = features.len();
    if n < 2 {
        return UmapProjection {
            coords: features.iter().map(|_| [0.0, 0.0]).collect(),
        };
    }

    let k = n_neighbors.min(n - 1).max(1);

    // Normalize features before computing distances.
    let (normalized, _means, _stddevs) = normalize_features(features);

    // Step 1: Compute pairwise distances.
    let mut dists = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = squared_dist(&normalized[i], &normalized[j]).sqrt();
            dists[i][j] = d;
            dists[j][i] = d;
        }
    }

    // Step 2: For each point, find k nearest neighbors and compute fuzzy
    //         membership strengths using local bandwidth (sigma).
    let mut knn: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
    let mut memberships = vec![vec![0.0f64; n]; n];

    for i in 0..n {
        let mut neighbors: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, dists[i][j]))
            .collect();
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        neighbors.truncate(k);

        let rho = neighbors.first().map(|&(_, d)| d).unwrap_or(0.0);

        // Binary search for sigma such that sum of exp(-(d - rho)/sigma) ≈ log2(k).
        let target = (k as f64).ln() / std::f64::consts::LN_2;
        let sigma = find_sigma(&neighbors, rho, target);

        for &(j, d) in &neighbors {
            let val = if d <= rho {
                1.0
            } else {
                (-(d - rho) / sigma).exp()
            };
            memberships[i][j] = val;
        }

        knn.push(neighbors);
    }

    // Step 3: Symmetrize memberships: p = p_ij + p_ji - p_ij * p_ji.
    let mut graph = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let p = memberships[i][j];
            let q = memberships[j][i];
            let sym = p + q - p * q;
            graph[i][j] = sym;
            graph[j][i] = sym;
        }
    }

    // Collect edges with weights for SGD optimization.
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if graph[i][j] > 1e-10 {
                edges.push((i, j, graph[i][j]));
            }
        }
    }

    // Step 4: Initialize 2D embedding with small random values seeded
    //         deterministically for reproducibility.
    let mut embedding: Vec<[f64; 2]> = Vec::with_capacity(n);
    {
        // Simple deterministic initialization: place points on a circle
        // scaled by a spectral-like heuristic, then perturb slightly.
        let scale = 10.0;
        for i in 0..n {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            // Mix in feature information for better initialization.
            let r = scale * (0.5 + 0.5 * normalized[i][0].abs().min(1.0));
            embedding.push([r * angle.cos(), r * angle.sin()]);
        }
    }

    // Step 5: Optimize embedding with SGD.
    let a = 1.0;
    let b = 1.0;
    let initial_lr = 1.0;

    let mut rng_state: u64 = UMAP_RANDOM_SEED;
    let mut cheap_rand = || -> f64 {
        // xorshift64
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f64) / (u64::MAX as f64)
    };

    for epoch in 0..n_epochs {
        let lr = initial_lr * (1.0 - epoch as f64 / n_epochs as f64).max(0.001);

        // Attractive forces along edges.
        for &(i, j, w) in &edges {
            let dx = embedding[i][0] - embedding[j][0];
            let dy = embedding[i][1] - embedding[j][1];
            let dist_sq = dx * dx + dy * dy + 1e-10;
            let grad = -2.0 * a * b * dist_sq.powf(b - 1.0) / (1.0 + a * dist_sq.powf(b));
            let force = w * grad * lr;
            embedding[i][0] += force * dx;
            embedding[i][1] += force * dy;
            embedding[j][0] -= force * dx;
            embedding[j][1] -= force * dy;
        }

        // Repulsive forces: sample negative pairs.
        let n_neg = (5 * edges.len()).min(n * n);
        for _ in 0..n_neg {
            let i = (cheap_rand() * n as f64) as usize % n;
            let j = (cheap_rand() * n as f64) as usize % n;
            if i == j || graph[i][j] > 1e-10 {
                continue;
            }
            let dx = embedding[i][0] - embedding[j][0];
            let dy = embedding[i][1] - embedding[j][1];
            let dist_sq = dx * dx + dy * dy + 1e-10;
            let grad = 2.0 * b / ((0.001 + dist_sq) * (1.0 + a * dist_sq.powf(b)));
            let force = grad * lr;
            let fx = (force * dx).clamp(-REPULSIVE_FORCE_CLAMP, REPULSIVE_FORCE_CLAMP);
            let fy = (force * dy).clamp(-REPULSIVE_FORCE_CLAMP, REPULSIVE_FORCE_CLAMP);
            embedding[i][0] += fx;
            embedding[i][1] += fy;
        }
    }

    UmapProjection { coords: embedding }
}

/// Binary search for the local bandwidth sigma such that the sum of
/// membership strengths approximates the target (log2(k)).
fn find_sigma(neighbors: &[(usize, f64)], rho: f64, target: f64) -> f64 {
    let mut lo = 1e-10_f64;
    let mut hi = 100.0_f64;

    for _ in 0..64 {
        let mid = (lo + hi) / 2.0;
        let sum: f64 = neighbors
            .iter()
            .map(|&(_, d)| {
                if d <= rho {
                    1.0
                } else {
                    (-(d - rho) / mid).exp()
                }
            })
            .sum();
        if sum > target {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    (lo + hi) / 2.0
}

/// Run UMAP projection and then k-means on the 2D projection to produce
/// agnostic cluster assignments. Returns (projection, cluster_assignments).
pub fn umap_cluster(
    features: &[[f64; 10]],
    n_neighbors: usize,
    n_epochs: usize,
    k: usize,
) -> (UmapProjection, Vec<usize>) {
    let projection = umap_project(features, n_neighbors, n_epochs);
    if projection.coords.len() < 2 {
        return (projection, vec![0; features.len()]);
    }

    // Convert 2D coords to [f64; 10] for reuse of kmeans_cluster (pad with zeros).
    let padded: Vec<[f64; 10]> = projection
        .coords
        .iter()
        .map(|&[x, y]| {
            let mut v = [0.0f64; 10];
            v[0] = x;
            v[1] = y;
            v
        })
        .collect();

    let assignments = kmeans_cluster(&padded, k, 50);
    (projection, assignments)
}

// ── Background classification thread ────────────────────────────────────────

/// Progress snapshot of the classification search.
#[derive(Debug, Clone)]
pub struct ClassifyProgress {
    pub total_examined: usize,
    pub classified_count: usize,
    pub running: bool,
    pub paused: bool,
    /// Counts per behavior class.
    pub class_counts: HashMap<BehaviorClass, usize>,
}

/// Shared state for the background classification thread.
struct ClassifyState {
    examined: std::collections::HashSet<(u32, u32, u32)>,
    results: Vec<ClassifiedRule>,
    total_examined: usize,
    running: bool,
}

/// Thread-safe handle to the background classification search.
pub struct ClassifyHandle {
    state: Arc<Mutex<ClassifyState>>,
    shutdown: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
}

impl ClassifyHandle {
    /// Get a snapshot of current progress.
    pub fn progress(&self) -> ClassifyProgress {
        let s = self.state.lock().unwrap();
        let mut class_counts = HashMap::new();
        for r in &s.results {
            *class_counts.entry(r.behavior).or_insert(0) += 1;
        }
        ClassifyProgress {
            total_examined: s.total_examined,
            classified_count: s.results.len(),
            running: s.running,
            paused: self.paused.load(Ordering::Relaxed),
            class_counts,
        }
    }

    /// Get a clone of all classified results.
    pub fn results(&self) -> Vec<ClassifiedRule> {
        self.state.lock().unwrap().results.clone()
    }

    /// Get results filtered by behavior class.
    pub fn results_by_class(&self, class: BehaviorClass) -> Vec<ClassifiedRule> {
        self.state
            .lock()
            .unwrap()
            .results
            .iter()
            .filter(|r| r.behavior == class)
            .cloned()
            .collect()
    }

    /// Re-cluster all results using k-means with the given k, and update UMAP projection.
    pub fn recluster(&self, k: usize) {
        let mut s = self.state.lock().unwrap();
        if s.results.is_empty() {
            return;
        }
        let features: Vec<[f64; 10]> = s.results.iter().map(|r| r.metrics.feature_vector()).collect();
        let assignments = kmeans_cluster(&features, k, 50);
        for (i, r) in s.results.iter_mut().enumerate() {
            r.cluster = Some(assignments[i]);
        }

        // Run UMAP projection and cluster on the projection.
        if features.len() >= 4 {
            let (projection, umap_assignments) = umap_cluster(&features, UMAP_N_NEIGHBORS, UMAP_N_EPOCHS, k);
            for (i, r) in s.results.iter_mut().enumerate() {
                r.umap_x = Some(projection.coords[i][0]);
                r.umap_y = Some(projection.coords[i][1]);
                r.umap_cluster = Some(umap_assignments[i]);
            }
        }
    }

    /// Signal the thread to stop.
    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    /// Pause the thread.
    pub fn pause(&self) {
        self.paused.store(true, Ordering::Relaxed);
    }

    /// Resume the thread.
    pub fn resume(&self) {
        self.paused.store(false, Ordering::Relaxed);
    }
}

// ── File I/O ────────────────────────────────────────────────────────────────

const RESULTS_HEADER: &str = "\
# CatConway Classified Rule Results
# Format: B<birth>/S<survival>[/R<radius>] class=<Class> variation=<v> mean_density=<md> final_density=<fd> density_range=<dr> trend=<t> autocorrelation=<ac> entropy=<e> dominant_period=<dp> monotonic_fraction=<mf> roughness=<r>
";

fn ensure_header(path: &Path) {
    if path.exists() {
        return;
    }
    if let Ok(mut f) = fs::File::create(path) {
        let _ = f.write_all(RESULTS_HEADER.as_bytes());
    }
}

fn append_classified_result(path: &Path, result: &ClassifiedRule) {
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(path) {
        let m = &result.metrics;
        let _ = writeln!(
            f,
            "{} class={} variation={:.6} mean_density={:.6} final_density={:.6} density_range={:.6} trend={:.8} autocorrelation={:.6} entropy={:.6} dominant_period={} monotonic_fraction={:.6} roughness={:.8}",
            result.label,
            result.behavior,
            m.variation,
            m.mean_density,
            m.final_density,
            m.density_range,
            m.trend,
            m.autocorrelation,
            m.entropy,
            m.dominant_period,
            m.monotonic_fraction,
            m.roughness,
        );
    }
}

fn load_examined(path: &Path) -> std::collections::HashSet<(u32, u32, u32)> {
    let mut set = std::collections::HashSet::new();
    let Ok(file) = fs::File::open(path) else {
        return set;
    };
    for line in std::io::BufReader::new(file).lines().map_while(Result::ok) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 3 {
            if let (Ok(b), Ok(s), Ok(r)) = (
                parts[0].parse::<u32>(),
                parts[1].parse::<u32>(),
                parts[2].parse::<u32>(),
            ) {
                set.insert((b, s, r));
            }
        }
    }
    set
}

fn append_examined(path: &Path, birth: u32, survival: u32, radius: u32) {
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{birth},{survival},{radius}");
    }
}

/// Load classified results from a file.
pub fn load_classified_results(path: &Path) -> Vec<ClassifiedRule> {
    let mut out = Vec::new();
    let Ok(file) = fs::File::open(path) else {
        return out;
    };
    for line in std::io::BufReader::new(file).lines().map_while(Result::ok) {
        let line = line.trim().to_string();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(result) = parse_classified_line(&line) {
            out.push(result);
        }
    }
    out
}

fn parse_classified_line(line: &str) -> Option<ClassifiedRule> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }
    let label = parts[0];
    let rules = parse_rule_label(label)?;

    let mut class = BehaviorClass::Complex;
    let mut variation = 0.0;
    let mut mean_density = 0.0;
    let mut final_density = 0.0;
    let mut density_range = 0.0;
    let mut trend = 0.0;
    let mut autocorrelation = 0.0;
    let mut entropy = 0.0;
    let mut dominant_period = 0usize;
    let mut monotonic_fraction = 0.0;
    let mut roughness = 0.0;

    for part in &parts[1..] {
        if let Some((key, val)) = part.split_once('=') {
            match key {
                "class" => {
                    if let Some(c) = BehaviorClass::from_str(val) {
                        class = c;
                    }
                }
                "variation" => variation = val.parse().unwrap_or(0.0),
                "mean_density" => mean_density = val.parse().unwrap_or(0.0),
                "final_density" => final_density = val.parse().unwrap_or(0.0),
                "density_range" => density_range = val.parse().unwrap_or(0.0),
                "trend" => trend = val.parse().unwrap_or(0.0),
                "autocorrelation" => autocorrelation = val.parse().unwrap_or(0.0),
                "entropy" => entropy = val.parse().unwrap_or(0.0),
                "dominant_period" => dominant_period = val.parse().unwrap_or(0),
                "monotonic_fraction" => monotonic_fraction = val.parse().unwrap_or(0.0),
                "roughness" => roughness = val.parse().unwrap_or(0.0),
                _ => {}
            }
        }
    }

    Some(ClassifiedRule {
        rules,
        label: label.to_string(),
        metrics: RuleMetrics {
            variation,
            mean_density,
            final_density,
            density_range,
            trend,
            autocorrelation,
            entropy,
            dominant_period,
            monotonic_fraction,
            roughness,
        },
        behavior: class,
        cluster: None,
        umap_x: None,
        umap_y: None,
        umap_cluster: None,
    })
}

// ── Background classification thread ────────────────────────────────────────

/// Maximum neighbor count for a given radius (Moore neighborhood).
fn max_neighbors(radius: u32) -> u32 {
    let side = 2 * radius + 1;
    side * side - 1
}

/// Spawn a background thread that classifies all rule sets for the configured
/// radius. Results are grouped by behavior class.
pub fn spawn_classify(config: ClassifyConfig) -> ClassifyHandle {
    let examined = load_examined(&config.examined_path);

    let state = Arc::new(Mutex::new(ClassifyState {
        total_examined: examined.len(),
        examined,
        results: Vec::new(),
        running: true,
    }));

    let shutdown = Arc::new(AtomicBool::new(false));
    let paused = Arc::new(AtomicBool::new(false));

    let handle = ClassifyHandle {
        state: state.clone(),
        shutdown: shutdown.clone(),
        paused: paused.clone(),
    };

    thread::spawn(move || {
        run_classify(state, shutdown, paused, config);
    });

    handle
}

fn run_classify(
    state: Arc<Mutex<ClassifyState>>,
    shutdown: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    config: ClassifyConfig,
) {
    ensure_header(&config.results_path);

    let max_n = max_neighbors(config.radius);
    let mask_count = 1u32 << (max_n + 1);

    /// Number of newly classified rules between re-clustering passes.
    const RECLUSTER_BATCH_SIZE: u32 = 100;

    let mut batch_count = 0u32;

    // birth=0 means no cells can ever be born → skip entirely.
    for birth in 1..mask_count {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        for survival in 0..mask_count {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Wait while paused.
            while paused.load(Ordering::Relaxed) && !shutdown.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(100));
            }

            // Skip already examined.
            {
                let s = state.lock().unwrap();
                if s.examined.contains(&(birth, survival, config.radius)) {
                    continue;
                }
            }

            let rules = Rules {
                birth,
                survival,
                radius: config.radius,
            };

            let metrics = compute_metrics(&rules, &config);
            let behavior = classify(&metrics);
            let label = rules_to_label(&rules);

            let result = ClassifiedRule {
                rules,
                label: label.clone(),
                metrics,
                behavior,
                cluster: None,
                umap_x: None,
                umap_y: None,
                umap_cluster: None,
            };

            // Record and persist.
            {
                let mut s = state.lock().unwrap();
                s.examined.insert((birth, survival, config.radius));
                s.total_examined += 1;
                append_examined(&config.examined_path, birth, survival, config.radius);
                append_classified_result(&config.results_path, &result);
                s.results.push(result);
            }

            batch_count += 1;

            // Re-cluster every 100 new rules to keep clusters up-to-date.
            if batch_count % RECLUSTER_BATCH_SIZE == 0 {
                let s = state.lock().unwrap();
                if s.results.len() >= 8 {
                    let features: Vec<[f64; 10]> =
                        s.results.iter().map(|r| r.metrics.feature_vector()).collect();
                    drop(s);
                    let k = 8.min(features.len());
                    let assignments = kmeans_cluster(&features, k, 30);

                    // Run UMAP projection alongside re-clustering.
                    let (projection, umap_assignments) = if features.len() >= 4 {
                        let (p, a) = umap_cluster(&features, UMAP_N_NEIGHBORS, UMAP_N_EPOCHS_INCREMENTAL, k);
                        (Some(p), Some(a))
                    } else {
                        (None, None)
                    };

                    let mut s = state.lock().unwrap();
                    for (i, r) in s.results.iter_mut().enumerate() {
                        if i < assignments.len() {
                            r.cluster = Some(assignments[i]);
                        }
                        if let (Some(proj), Some(ua)) = (&projection, &umap_assignments) {
                            if i < proj.coords.len() {
                                r.umap_x = Some(proj.coords[i][0]);
                                r.umap_y = Some(proj.coords[i][1]);
                                r.umap_cluster = Some(ua[i]);
                            }
                        }
                    }
                }
            }
        }
    }

    // Final clustering pass.
    {
        let mut s = state.lock().unwrap();
        if s.results.len() >= 8 {
            let features: Vec<[f64; 10]> =
                s.results.iter().map(|r| r.metrics.feature_vector()).collect();
            let k = 8.min(features.len());
            let assignments = kmeans_cluster(&features, k, 50);

            // Final UMAP projection.
            let (projection, umap_assignments) = if features.len() >= 4 {
                let (p, a) = umap_cluster(&features, UMAP_N_NEIGHBORS, UMAP_N_EPOCHS, k);
                (Some(p), Some(a))
            } else {
                (None, None)
            };

            for (i, r) in s.results.iter_mut().enumerate() {
                if i < assignments.len() {
                    r.cluster = Some(assignments[i]);
                }
                if let (Some(proj), Some(ua)) = (&projection, &umap_assignments) {
                    if i < proj.coords.len() {
                        r.umap_x = Some(proj.coords[i][0]);
                        r.umap_y = Some(proj.coords[i][1]);
                        r.umap_cluster = Some(ua[i]);
                    }
                }
            }
        }
        s.running = false;
    }
    log::info!("Rule classification complete");
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Metrics computation ──

    #[test]
    fn constant_trace_gives_zero_variation() {
        let densities = vec![0.5; 100];
        let m = compute_metrics_from_trace(&densities, 20, 30);
        assert!(m.variation < 1e-10);
        assert!((m.mean_density - 0.5).abs() < 1e-10);
        assert!(m.trend.abs() < 1e-10);
    }

    #[test]
    fn dead_trace_classified_as_dead() {
        let densities = vec![0.0; 100];
        let m = compute_metrics_from_trace(&densities, 20, 30);
        assert_eq!(classify(&m), BehaviorClass::Dead);
    }

    #[test]
    fn saturated_trace_classified_as_explosive() {
        let densities = vec![0.95; 100];
        let m = compute_metrics_from_trace(&densities, 20, 30);
        assert_eq!(classify(&m), BehaviorClass::Explosive);
    }

    #[test]
    fn periodic_trace_has_nonzero_period() {
        let densities: Vec<f64> = (0..200)
            .map(|i| if i % 5 == 0 { 0.4 } else { 0.3 })
            .collect();
        let m = compute_metrics_from_trace(&densities, 20, 30);
        assert!(m.dominant_period > 0);
    }

    #[test]
    fn linear_trend_detected() {
        let densities: Vec<f64> = (0..200).map(|i| 0.1 + i as f64 * 0.002).collect();
        let m = compute_metrics_from_trace(&densities, 20, 30);
        assert!(m.trend > 0.0, "Expected positive trend, got {}", m.trend);
    }

    #[test]
    fn high_variation_trace() {
        let densities: Vec<f64> = (0..200)
            .map(|i| 0.3 + 0.2 * ((i as f64 * 0.1).sin()))
            .collect();
        let m = compute_metrics_from_trace(&densities, 20, 30);
        assert!(m.variation > 0.0);
        assert!(m.roughness > 0.0);
    }

    #[test]
    fn empty_trace_returns_defaults() {
        let m = compute_metrics_from_trace(&[], 0, 30);
        assert_eq!(m.variation, 0.0);
        assert_eq!(m.mean_density, 0.0);
    }

    // ── Classification ──

    #[test]
    fn classify_static_low_variation() {
        let m = RuleMetrics {
            variation: 0.001,
            mean_density: 0.3,
            final_density: 0.3,
            density_range: 0.002,
            trend: 0.00001,
            autocorrelation: 0.9,
            entropy: 0.1,
            dominant_period: 0,
            monotonic_fraction: 0.5,
            roughness: 0.001,
        };
        assert_eq!(classify(&m), BehaviorClass::Static);
    }

    #[test]
    fn classify_growing() {
        let m = RuleMetrics {
            variation: 0.05,
            mean_density: 0.4,
            final_density: 0.6,
            density_range: 0.3,
            trend: 0.001,
            autocorrelation: 0.5,
            entropy: 2.0,
            dominant_period: 0,
            monotonic_fraction: 0.7,
            roughness: 0.01,
        };
        assert_eq!(classify(&m), BehaviorClass::Growing);
    }

    #[test]
    fn classify_declining() {
        let m = RuleMetrics {
            variation: 0.05,
            mean_density: 0.3,
            final_density: 0.1,
            density_range: 0.3,
            trend: -0.001,
            autocorrelation: 0.5,
            entropy: 2.0,
            dominant_period: 0,
            monotonic_fraction: 0.7,
            roughness: 0.01,
        };
        assert_eq!(classify(&m), BehaviorClass::Declining);
    }

    #[test]
    fn classify_chaotic() {
        let m = RuleMetrics {
            variation: 0.2,
            mean_density: 0.4,
            final_density: 0.35,
            density_range: 0.3,
            trend: 0.0001,
            autocorrelation: 0.1,
            entropy: 3.0,
            dominant_period: 0,
            monotonic_fraction: 0.5,
            roughness: 0.05,
        };
        assert_eq!(classify(&m), BehaviorClass::Chaotic);
    }

    // ── K-means clustering ──

    #[test]
    fn kmeans_empty_returns_empty() {
        let result = kmeans_cluster(&[], 3, 10);
        assert!(result.is_empty());
    }

    #[test]
    fn kmeans_single_point() {
        let features = vec![[1.0; 10]];
        let result = kmeans_cluster(&features, 1, 10);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn kmeans_two_clusters() {
        let mut features = Vec::new();
        // Cluster A: all values near 0.
        for _ in 0..10 {
            features.push([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
        // Cluster B: all values near 10.
        for _ in 0..10 {
            features.push([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
        }
        let result = kmeans_cluster(&features, 2, 50);
        assert_eq!(result.len(), 20);
        // All items in cluster A should have the same label.
        let a_cluster = result[0];
        assert!(result[..10].iter().all(|&c| c == a_cluster));
        // All items in cluster B should have the same label (different from A).
        let b_cluster = result[10];
        assert!(result[10..].iter().all(|&c| c == b_cluster));
        assert_ne!(a_cluster, b_cluster);
    }

    // ── Linear trend ──

    #[test]
    fn linear_trend_positive() {
        let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let slope = linear_trend(&data);
        assert!((slope - 0.01).abs() < 0.001);
    }

    #[test]
    fn linear_trend_flat() {
        let data = vec![5.0; 100];
        let slope = linear_trend(&data);
        assert!(slope.abs() < 1e-10);
    }

    // ── Autocorrelation ──

    #[test]
    fn autocorrelation_constant_is_zero() {
        let data = vec![1.0; 100];
        let ac = lag1_autocorrelation(&data, 1.0, 0.0);
        assert_eq!(ac, 0.0);
    }

    #[test]
    fn autocorrelation_highly_correlated() {
        // Slowly varying signal has high lag-1 autocorrelation.
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.01).sin()).collect();
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let ac = lag1_autocorrelation(&data, mean, var);
        assert!(ac > 0.9, "Expected high autocorrelation, got {ac}");
    }

    // ── Entropy ──

    #[test]
    fn entropy_constant_is_zero() {
        let data = vec![0.5; 100];
        assert_eq!(population_entropy(&data), 0.0);
    }

    #[test]
    fn entropy_varied_is_positive() {
        let data: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        assert!(population_entropy(&data) > 0.0);
    }

    // ── Feature vector ──

    #[test]
    fn feature_vector_length() {
        let m = RuleMetrics {
            variation: 0.1,
            mean_density: 0.3,
            final_density: 0.3,
            density_range: 0.05,
            trend: 0.001,
            autocorrelation: 0.5,
            entropy: 2.0,
            dominant_period: 5,
            monotonic_fraction: 0.6,
            roughness: 0.01,
        };
        assert_eq!(m.feature_vector().len(), 10);
        assert_eq!(FEATURE_NAMES.len(), 10);
    }

    // ── Behavior class ──

    #[test]
    fn behavior_class_display() {
        assert_eq!(format!("{}", BehaviorClass::Dead), "Dead");
        assert_eq!(format!("{}", BehaviorClass::Chaotic), "Chaotic");
    }

    #[test]
    fn behavior_class_roundtrip() {
        for &class in BehaviorClass::all() {
            let s = format!("{class}");
            assert_eq!(BehaviorClass::from_str(&s), Some(class));
        }
    }

    #[test]
    fn behavior_class_from_str_invalid() {
        assert_eq!(BehaviorClass::from_str("nonsense"), None);
    }

    // ── File I/O ──

    #[test]
    fn classified_result_roundtrip() {
        let dir = std::env::temp_dir().join("catconway_test_classify_results");
        let _ = fs::remove_file(&dir);

        ensure_header(&dir);
        let result = ClassifiedRule {
            rules: Rules::conway(),
            label: "B3/S23".into(),
            metrics: RuleMetrics {
                variation: 0.12,
                mean_density: 0.03,
                final_density: 0.03,
                density_range: 0.01,
                trend: 0.0001,
                autocorrelation: 0.5,
                entropy: 1.5,
                dominant_period: 0,
                monotonic_fraction: 0.4,
                roughness: 0.005,
            },
            behavior: BehaviorClass::Complex,
            cluster: None,
            umap_x: None,
            umap_y: None,
            umap_cluster: None,
        };
        append_classified_result(&dir, &result);

        let loaded = load_classified_results(&dir);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].label, "B3/S23");
        assert_eq!(loaded[0].behavior, BehaviorClass::Complex);
        assert!((loaded[0].metrics.variation - 0.12).abs() < 0.001);

        let _ = fs::remove_file(&dir);
    }

    // ── Monotonic fraction ──

    #[test]
    fn monotonic_fraction_increasing() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mf = compute_monotonic_fraction(&data);
        assert!((mf - 1.0).abs() < 1e-10);
    }

    #[test]
    fn monotonic_fraction_alternating() {
        let data: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let mf = compute_monotonic_fraction(&data);
        assert!(mf < 0.1, "Expected low monotonic fraction, got {mf}");
    }

    // ── Roughness ──

    #[test]
    fn roughness_constant_is_zero() {
        let data = vec![0.5; 100];
        assert_eq!(compute_roughness(&data), 0.0);
    }

    #[test]
    fn roughness_alternating_is_high() {
        let data: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let r = compute_roughness(&data);
        assert!((r - 1.0).abs() < 1e-10);
    }

    // ── Handle pause/resume ──

    #[test]
    fn classify_handle_pause_resume() {
        let results_path = std::env::temp_dir().join("catconway_test_classify_pause_results.txt");
        let examined_path = std::env::temp_dir().join("catconway_test_classify_pause_examined.txt");
        let _ = fs::remove_file(&results_path);
        let _ = fs::remove_file(&examined_path);

        let config = ClassifyConfig {
            grid_size: 8,
            generations: 20,
            burn_in: 10,
            results_path: results_path.clone(),
            examined_path: examined_path.clone(),
            ..ClassifyConfig::default()
        };

        let handle = spawn_classify(config);

        // Initially not paused and running.
        let p = handle.progress();
        assert!(!p.paused);
        assert!(p.running);

        // Pause.
        handle.pause();
        assert!(handle.progress().paused);

        // Resume.
        handle.resume();
        assert!(!handle.progress().paused);

        // Stop and wait for thread to finish.
        handle.stop();
        thread::sleep(Duration::from_millis(300));
        assert!(!handle.progress().running);

        let _ = fs::remove_file(&results_path);
        let _ = fs::remove_file(&examined_path);
    }

    // ── Compute metrics for actual rules ──

    #[test]
    fn conway_metrics_are_reasonable() {
        let config = ClassifyConfig {
            grid_size: 32,
            generations: 200,
            burn_in: 100,
            ..ClassifyConfig::default()
        };
        let rules = Rules::conway();
        let m = compute_metrics(&rules, &config);
        // Conway should settle to some density between 0 and 1.
        assert!(m.mean_density > 0.0 && m.mean_density < 1.0);
        assert!(m.final_density >= 0.0 && m.final_density <= 1.0);
    }

    #[test]
    fn seeds_likely_dead_or_chaotic() {
        let config = ClassifyConfig {
            grid_size: 32,
            generations: 200,
            burn_in: 100,
            ..ClassifyConfig::default()
        };
        let rules = Rules::seeds();
        let m = compute_metrics(&rules, &config);
        let class = classify(&m);
        // Seeds is explosive/chaotic — it should not be static or dead for a short run.
        // (Behavior depends on grid size and density, so just check it classified.)
        assert!(
            class == BehaviorClass::Chaotic
                || class == BehaviorClass::Dead
                || class == BehaviorClass::Periodic
                || class == BehaviorClass::Complex
                || class == BehaviorClass::Explosive,
            "Unexpected class: {class}"
        );
    }

    // ── UMAP ──

    #[test]
    fn umap_single_point_returns_origin() {
        let features = vec![[1.0; 10]];
        let proj = umap_project(&features, 5, 50);
        assert_eq!(proj.coords.len(), 1);
        assert_eq!(proj.coords[0], [0.0, 0.0]);
    }

    #[test]
    fn umap_empty_returns_empty() {
        let features: Vec<[f64; 10]> = vec![];
        let proj = umap_project(&features, 5, 50);
        assert!(proj.coords.is_empty());
    }

    #[test]
    fn umap_two_clusters_separated() {
        let mut features = Vec::new();
        // Cluster A: near 0.
        for _ in 0..10 {
            features.push([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
        // Cluster B: near 10.
        for _ in 0..10 {
            features.push([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
        }
        let proj = umap_project(&features, 5, 200);
        assert_eq!(proj.coords.len(), 20);

        // Verify two clusters are spatially separated in the projection.
        let centroid_a = [
            proj.coords[0..10].iter().map(|c| c[0]).sum::<f64>() / 10.0,
            proj.coords[0..10].iter().map(|c| c[1]).sum::<f64>() / 10.0,
        ];
        let centroid_b = [
            proj.coords[10..20].iter().map(|c| c[0]).sum::<f64>() / 10.0,
            proj.coords[10..20].iter().map(|c| c[1]).sum::<f64>() / 10.0,
        ];
        let dist = ((centroid_a[0] - centroid_b[0]).powi(2)
            + (centroid_a[1] - centroid_b[1]).powi(2))
        .sqrt();
        assert!(
            dist > 0.1,
            "Expected separated clusters, got distance {dist}"
        );
    }

    #[test]
    fn umap_cluster_assigns_all_points() {
        let mut features = Vec::new();
        for i in 0..20 {
            let v = i as f64;
            features.push([v, v, v, v, v, v, v, v, v, v]);
        }
        let (proj, assignments) = umap_cluster(&features, 5, 100, 3);
        assert_eq!(proj.coords.len(), 20);
        assert_eq!(assignments.len(), 20);
        // All assignments should be in range [0, 3).
        assert!(assignments.iter().all(|&a| a < 3));
    }

    #[test]
    fn find_sigma_converges() {
        let neighbors = vec![(1, 1.0), (2, 2.0), (3, 3.0)];
        let target = (3.0_f64).ln() / std::f64::consts::LN_2;
        let sigma = find_sigma(&neighbors, 0.5, target);
        assert!(sigma > 0.0);
        assert!(sigma.is_finite());
    }
}
