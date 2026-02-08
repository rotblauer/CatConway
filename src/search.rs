use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use rand::Rng;

use crate::grid::Rules;

/// Configuration for the background rule search.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Grid size (square) for CPU simulations.
    pub grid_size: u32,
    /// Initial random fill density.
    pub density: f64,
    /// Number of generations to simulate per rule set.
    pub generations: u32,
    /// Minimum coefficient of variation to consider a rule set "interesting".
    pub min_variation: f64,
    /// Maximum period length to detect and reject as uninteresting.
    pub max_period: usize,
    /// Path for the results output file.
    pub results_path: PathBuf,
    /// Path for tracking previously examined rule sets.
    pub examined_path: PathBuf,
    /// Neighborhood radius to search (1 or 2).
    pub radius: u32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            grid_size: 64,
            density: 0.25,
            generations: 300,
            min_variation: 0.05,
            max_period: 20,
            results_path: PathBuf::from("search_results.txt"),
            examined_path: PathBuf::from("search_examined.txt"),
            radius: 1,
        }
    }
}

/// A rule set found to be interesting, with its associated metrics.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub rules: Rules,
    pub label: String,
    pub variation: f64,
    pub final_density: f64,
}

/// Progress snapshot of the background search.
#[derive(Debug, Clone)]
pub struct SearchProgress {
    pub total_examined: usize,
    pub total_interesting: usize,
    pub running: bool,
}

/// Shared state for the background search thread.
struct SearchState {
    examined: HashSet<(u32, u32, u32)>,
    results: Vec<SearchResult>,
    total_examined: usize,
    total_interesting: usize,
    running: bool,
}

/// Thread-safe handle to the background search.
pub struct SearchHandle {
    state: Arc<Mutex<SearchState>>,
    shutdown: Arc<AtomicBool>,
}

impl SearchHandle {
    /// Get a snapshot of current search progress.
    pub fn progress(&self) -> SearchProgress {
        let s = self.state.lock().unwrap();
        SearchProgress {
            total_examined: s.total_examined,
            total_interesting: s.total_interesting,
            running: s.running,
        }
    }

    /// Get a clone of all interesting results found so far.
    pub fn results(&self) -> Vec<SearchResult> {
        self.state.lock().unwrap().results.clone()
    }

    /// Signal the search thread to stop.
    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

// ── CPU simulation (mirrors the GPU compute shader) ─────────────────────────

/// Advance the grid by one generation on the CPU using the given rules.
/// The grid uses toroidal (wrap-around) boundary conditions.
pub fn cpu_step(cells: &[u32], width: u32, height: u32, rules: &Rules) -> Vec<u32> {
    let w = width as i32;
    let h = height as i32;
    let r = rules.radius as i32;
    let mut next = vec![0u32; cells.len()];

    for y in 0..h {
        for x in 0..w {
            let mut count = 0u32;
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = ((x + dx) % w + w) % w;
                    let ny = ((y + dy) % h + h) % h;
                    count += cells[(ny * w + nx) as usize];
                }
            }

            let alive = cells[(y * w + x) as usize] == 1;
            let next_alive = if alive {
                (rules.survival >> count) & 1 == 1
            } else {
                (rules.birth >> count) & 1 == 1
            };
            next[(y * w + x) as usize] = u32::from(next_alive);
        }
    }
    next
}

// ── Interestingness metrics ─────────────────────────────────────────────────

/// Evaluate a rule set by running a CPU simulation and measuring variation.
/// Returns `(coefficient_of_variation, final_population)`.
fn evaluate_rules(rules: &Rules, config: &SearchConfig) -> (f64, u64) {
    let size = config.grid_size;
    let total = (size * size) as usize;
    let mut cells = vec![0u32; total];

    // Deterministic-ish random init with the configured density.
    let mut rng = rand::thread_rng();
    for cell in &mut cells {
        *cell = u32::from(rng.gen_range(0.0..1.0) < config.density);
    }

    let mut populations: Vec<u64> = Vec::with_capacity(config.generations as usize);

    for _ in 0..config.generations {
        let pop: u64 = cells.iter().map(|&c| u64::from(c)).sum();
        populations.push(pop);

        // Early exit if dead.
        if pop == 0 {
            break;
        }

        cells = cpu_step(&cells, size, size, rules);
    }

    let variation = compute_variation(&populations, total as u64, config.max_period);
    let final_pop = populations.last().copied().unwrap_or(0);
    (variation, final_pop)
}

/// Compute the coefficient of variation over the second half of the population
/// trace, returning 0.0 for dead, saturated, or periodic traces.
fn compute_variation(populations: &[u64], total_cells: u64, max_period: usize) -> f64 {
    if populations.len() < 20 {
        return 0.0;
    }

    // Analyse the second half (after transients have settled).
    let start = populations.len() / 2;
    let window = &populations[start..];

    if window.is_empty() {
        return 0.0;
    }

    let mean = window.iter().sum::<u64>() as f64 / window.len() as f64;

    // Dead or saturated → not interesting.
    if mean < 1.0 || mean > total_cells as f64 * 0.95 {
        return 0.0;
    }

    // Check for exact periodicity up to max_period.
    if is_periodic(window, max_period) {
        return 0.0;
    }

    let variance =
        window.iter().map(|&p| (p as f64 - mean).powi(2)).sum::<f64>() / window.len() as f64;
    let stddev = variance.sqrt();

    stddev / mean
}

/// Return `true` if `data` is periodic with period ≤ `max_period`.
fn is_periodic(data: &[u64], max_period: usize) -> bool {
    for period in 1..=max_period.min(data.len() / 2) {
        // windows(period + 1) yields [data[i]..=data[i+period]] for every i.
        // Checking w[0] == w[period] therefore verifies data[i] == data[i+period]
        // across the entire slice, which is the definition of period-P repetition.
        let periodic = data.windows(period + 1).all(|w| w[0] == w[period]);
        if periodic {
            return true;
        }
    }
    false
}

// ── B/S notation formatting & parsing ───────────────────────────────────────

/// Format a `Rules` value as a human-readable B/S label.
///
/// Radius-1 rules use concatenated digits: `B36/S23`.
/// Radius-2+ rules use comma-separated counts: `B3,4,5/S4,5,6,7,8/R2`.
pub fn rules_to_label(rules: &Rules) -> String {
    let max_n = max_neighbors(rules.radius);

    let fmt = |mask: u32| -> String {
        let digits: Vec<String> = (0..=max_n)
            .filter(|&i| (mask >> i) & 1 == 1)
            .map(|i| i.to_string())
            .collect();
        if rules.radius > 1 {
            digits.join(",")
        } else {
            digits.join("")
        }
    };

    if rules.radius > 1 {
        format!("B{}/S{}/R{}", fmt(rules.birth), fmt(rules.survival), rules.radius)
    } else {
        format!("B{}/S{}", fmt(rules.birth), fmt(rules.survival))
    }
}

/// Parse a B/S label back into a `Rules` value.
///
/// Accepted formats:
/// - `B36/S23` (radius-1)
/// - `B3,4,5/S4,5,6,7,8/R2` (extended radius)
pub fn parse_rule_label(label: &str) -> Option<Rules> {
    let parts: Vec<&str> = label.split('/').collect();
    if parts.len() < 2 {
        return None;
    }

    let birth_str = parts[0].strip_prefix('B')?;
    let survival_str = parts[1].strip_prefix('S')?;
    let radius: u32 = if parts.len() > 2 {
        parts[2].strip_prefix('R')?.parse().ok()?
    } else {
        1
    };

    let birth = parse_neighbor_digits(birth_str)?;
    let survival = parse_neighbor_digits(survival_str)?;

    Some(Rules {
        birth,
        survival,
        radius,
    })
}

/// Maximum neighbor count for the largest supported radius (radius 2 = 24).
const MAX_NEIGHBOR_COUNT: u32 = 24;

/// Parse comma-separated or plain digit strings into a bitmask.
fn parse_neighbor_digits(s: &str) -> Option<u32> {
    if s.is_empty() {
        return Some(0);
    }

    let mut mask = 0u32;
    if s.contains(',') {
        for part in s.split(',') {
            let n: u32 = part.parse().ok()?;
            if n > MAX_NEIGHBOR_COUNT {
                return None;
            }
            mask |= 1 << n;
        }
    } else {
        for ch in s.chars() {
            let n = ch.to_digit(10)?;
            mask |= 1 << n;
        }
    }
    Some(mask)
}

/// Maximum neighbor count for a given radius (Moore neighborhood).
fn max_neighbors(radius: u32) -> u32 {
    let side = 2 * radius + 1;
    side * side - 1
}

// ── File I/O ────────────────────────────────────────────────────────────────

const RESULTS_HEADER: &str = "\
# CatConway Rule Search Results
# Format: B<birth>/S<survival>[/R<radius>] variation=<cv> final_density=<d>
# Load these rule sets into the visualizer to explore them.
";

/// Ensure the results file exists and has a header.
fn ensure_results_header(path: &Path) {
    if path.exists() {
        return;
    }
    if let Ok(mut f) = fs::File::create(path) {
        let _ = f.write_all(RESULTS_HEADER.as_bytes());
    }
}

/// Append a search result to the results file.
fn append_result(path: &Path, result: &SearchResult) {
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(
            f,
            "{} variation={:.4} final_density={:.4}",
            result.label, result.variation, result.final_density
        );
    }
}

/// Load all previously examined `(birth, survival, radius)` tuples from disk.
fn load_examined(path: &Path) -> HashSet<(u32, u32, u32)> {
    let mut set = HashSet::new();
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

/// Append one examined entry to the tracking file.
fn append_examined(path: &Path, birth: u32, survival: u32, radius: u32) {
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{birth},{survival},{radius}");
    }
}

/// Load interesting rule sets from a results file so the visualizer can use them.
pub fn load_results(path: &Path) -> Vec<(String, Rules)> {
    let mut out = Vec::new();
    let Ok(file) = fs::File::open(path) else {
        return out;
    };
    for line in std::io::BufReader::new(file).lines().map_while(Result::ok) {
        let line = line.trim().to_string();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let label = line.split_whitespace().next().unwrap_or("");
        if let Some(rules) = parse_rule_label(label) {
            out.push((label.to_string(), rules));
        }
    }
    out
}

// ── Background search thread ────────────────────────────────────────────────

/// Spawn a background thread that iterates over rule sets for the configured
/// radius, evaluates each one, and writes interesting results to disk.
///
/// Returns a handle that can be used to query progress and stop the search.
pub fn spawn_search(config: SearchConfig) -> SearchHandle {
    let examined = load_examined(&config.examined_path);

    let state = Arc::new(Mutex::new(SearchState {
        total_examined: examined.len(),
        total_interesting: 0,
        examined,
        results: Vec::new(),
        running: true,
    }));

    let shutdown = Arc::new(AtomicBool::new(false));

    let handle = SearchHandle {
        state: state.clone(),
        shutdown: shutdown.clone(),
    };

    thread::spawn(move || {
        run_search(state, shutdown, config);
    });

    handle
}

/// Main search loop executed on its own thread.
fn run_search(state: Arc<Mutex<SearchState>>, shutdown: Arc<AtomicBool>, config: SearchConfig) {
    ensure_results_header(&config.results_path);

    let max_n = max_neighbors(config.radius);
    let mask_count = 1u32 << (max_n + 1);

    // birth=0 means no cells can ever be born → skip entirely.
    for birth in 1..mask_count {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        for survival in 0..mask_count {
            if shutdown.load(Ordering::Relaxed) {
                break;
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

            let (variation, final_pop) = evaluate_rules(&rules, &config);
            let total_cells = (config.grid_size as u64) * (config.grid_size as u64);
            let final_density = final_pop as f64 / total_cells as f64;

            let label = rules_to_label(&rules);

            // Record and persist.
            {
                let mut s = state.lock().unwrap();
                s.examined.insert((birth, survival, config.radius));
                s.total_examined += 1;
                append_examined(&config.examined_path, birth, survival, config.radius);

                if variation >= config.min_variation {
                    let result = SearchResult {
                        rules,
                        label: label.clone(),
                        variation,
                        final_density,
                    };
                    append_result(&config.results_path, &result);
                    s.results.push(result);
                    s.total_interesting += 1;
                    log::info!("Interesting rule set found: {label} (variation={variation:.4})");
                }
            }
        }
    }

    if let Ok(mut s) = state.lock() {
        s.running = false;
    }
    log::info!("Rule search complete");
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CPU simulation ──

    #[test]
    fn cpu_step_empty_grid_stays_empty() {
        let cells = vec![0u32; 25];
        let rules = Rules::conway();
        let next = cpu_step(&cells, 5, 5, &rules);
        assert!(next.iter().all(|&c| c == 0));
    }

    #[test]
    fn cpu_step_block_is_stable() {
        // 2×2 block at (1,1) in a 6×6 grid should be a still life under Conway.
        let mut cells = vec![0u32; 36];
        for &(x, y) in &[(1, 1), (2, 1), (1, 2), (2, 2)] {
            cells[y * 6 + x] = 1;
        }
        let rules = Rules::conway();
        let next = cpu_step(&cells, 6, 6, &rules);
        assert_eq!(cells, next);
    }

    #[test]
    fn cpu_step_blinker_oscillates() {
        // Vertical blinker at center of 5×5 grid → should become horizontal.
        let mut cells = vec![0u32; 25];
        // Vertical: (2,1), (2,2), (2,3)
        cells[1 * 5 + 2] = 1;
        cells[2 * 5 + 2] = 1;
        cells[3 * 5 + 2] = 1;
        let rules = Rules::conway();
        let next = cpu_step(&cells, 5, 5, &rules);
        // Should be horizontal: (1,2), (2,2), (3,2)
        assert_eq!(next[2 * 5 + 1], 1);
        assert_eq!(next[2 * 5 + 2], 1);
        assert_eq!(next[2 * 5 + 3], 1);
        let pop: u32 = next.iter().sum();
        assert_eq!(pop, 3);
    }

    #[test]
    fn cpu_step_wraps_toroidally() {
        // Three cells along the top edge should wrap neighbors via bottom.
        let mut cells = vec![0u32; 25];
        cells[0 * 5 + 1] = 1; // (1,0)
        cells[0 * 5 + 2] = 1; // (2,0)
        cells[0 * 5 + 3] = 1; // (3,0)
        let rules = Rules::conway();
        let next = cpu_step(&cells, 5, 5, &rules);
        // Top/bottom wrapping: expect alive at (2,4) and (2,0).
        assert_eq!(next[4 * 5 + 2], 1, "should wrap to bottom row");
        assert_eq!(next[0 * 5 + 2], 1, "center should survive");
    }

    // ── Periodicity detection ──

    #[test]
    fn detects_static_as_periodic() {
        let data = vec![100u64; 50];
        assert!(is_periodic(&data, 20));
    }

    #[test]
    fn detects_period_two() {
        let data: Vec<u64> = (0..50).map(|i| if i % 2 == 0 { 10 } else { 20 }).collect();
        assert!(is_periodic(&data, 20));
    }

    #[test]
    fn non_periodic_returns_false() {
        let data: Vec<u64> = (0..50).map(|i| i * i).collect();
        assert!(!is_periodic(&data, 20));
    }

    // ── Variation metric ──

    #[test]
    fn dead_population_gives_zero_variation() {
        let pops = vec![0u64; 30];
        assert_eq!(compute_variation(&pops, 100, 20), 0.0);
    }

    #[test]
    fn constant_population_gives_zero_variation() {
        let pops = vec![50u64; 40];
        assert_eq!(compute_variation(&pops, 100, 20), 0.0);
    }

    #[test]
    fn varying_population_gives_positive_variation() {
        // Linearly increasing → non-periodic, non-zero CV.
        let pops: Vec<u64> = (0..100).map(|i| 100 + i * 2).collect();
        let v = compute_variation(&pops, 10000, 20);
        assert!(v > 0.0, "expected positive variation, got {v}");
    }

    // ── Label formatting & parsing ──

    #[test]
    fn conway_label_roundtrip() {
        let rules = Rules::conway();
        let label = rules_to_label(&rules);
        assert_eq!(label, "B3/S23");
        let parsed = parse_rule_label(&label).unwrap();
        assert_eq!(parsed, rules);
    }

    #[test]
    fn highlife_label_roundtrip() {
        let rules = Rules::highlife();
        let label = rules_to_label(&rules);
        assert_eq!(label, "B36/S23");
        let parsed = parse_rule_label(&label).unwrap();
        assert_eq!(parsed, rules);
    }

    #[test]
    fn seeds_label_roundtrip() {
        let rules = Rules::seeds();
        let label = rules_to_label(&rules);
        assert_eq!(label, "B2/S");
        let parsed = parse_rule_label(&label).unwrap();
        assert_eq!(parsed, rules);
    }

    #[test]
    fn extended_radius_label_roundtrip() {
        let rules = Rules::bugs();
        let label = rules_to_label(&rules);
        assert_eq!(label, "B3,4,5/S4,5,6,7,8/R2");
        let parsed = parse_rule_label(&label).unwrap();
        assert_eq!(parsed, rules);
    }

    #[test]
    fn parse_invalid_label_returns_none() {
        assert!(parse_rule_label("").is_none());
        assert!(parse_rule_label("nonsense").is_none());
        assert!(parse_rule_label("X3/Y2").is_none());
    }

    // ── File I/O ──

    #[test]
    fn examined_roundtrip() {
        let dir = std::env::temp_dir().join("catconway_test_examined");
        let _ = fs::remove_file(&dir);

        append_examined(&dir, 8, 12, 1);
        append_examined(&dir, 3, 5, 2);

        let set = load_examined(&dir);
        assert!(set.contains(&(8, 12, 1)));
        assert!(set.contains(&(3, 5, 2)));
        assert_eq!(set.len(), 2);

        let _ = fs::remove_file(&dir);
    }

    #[test]
    fn results_file_roundtrip() {
        let dir = std::env::temp_dir().join("catconway_test_results");
        let _ = fs::remove_file(&dir);

        ensure_results_header(&dir);
        append_result(
            &dir,
            &SearchResult {
                rules: Rules::conway(),
                label: "B3/S23".into(),
                variation: 0.12,
                final_density: 0.03,
            },
        );

        let loaded = load_results(&dir);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "B3/S23");
        assert_eq!(loaded[0].1, Rules::conway());

        let _ = fs::remove_file(&dir);
    }

    #[test]
    fn max_neighbors_radius_1() {
        assert_eq!(max_neighbors(1), 8);
    }

    #[test]
    fn max_neighbors_radius_2() {
        assert_eq!(max_neighbors(2), 24);
    }
}
