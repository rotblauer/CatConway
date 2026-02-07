use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Maximum number of population samples retained for the history plot.
const MAX_HISTORY: usize = 512;

/// Snapshot of simulation statistics at a given generation.
#[derive(Debug, Clone)]
pub struct StatsSample {
    pub generation: u64,
    pub population: u64,
    pub density: f64,
    pub timestamp: Instant,
}

/// Thread-safe statistics store shared between the sampling thread and the UI.
#[derive(Debug, Clone)]
pub struct Stats {
    inner: Arc<Mutex<StatsInner>>,
}

#[derive(Debug)]
struct StatsInner {
    /// Ring buffer of population samples over time.
    history: VecDeque<StatsSample>,
    /// Most recent generation rate (generations per second).
    gen_rate: f64,
    /// Last generation seen (for computing rate).
    last_gen: u64,
    last_rate_time: Instant,
    /// Total cell count of the grid.
    total_cells: u64,
}

impl Stats {
    pub fn new(total_cells: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(StatsInner {
                history: VecDeque::with_capacity(MAX_HISTORY),
                gen_rate: 0.0,
                last_gen: 0,
                last_rate_time: Instant::now(),
                total_cells,
            })),
        }
    }

    /// Record a new population sample.  Called from the sampling thread.
    pub fn record(&self, generation: u64, population: u64) {
        if let Ok(mut inner) = self.inner.lock() {
            let density = if inner.total_cells > 0 {
                population as f64 / inner.total_cells as f64
            } else {
                0.0
            };

            let now = Instant::now();
            let dt = now.duration_since(inner.last_rate_time).as_secs_f64();
            if dt > 0.25 {
                let dg = generation.saturating_sub(inner.last_gen) as f64;
                inner.gen_rate = dg / dt;
                inner.last_gen = generation;
                inner.last_rate_time = now;
            }

            let sample = StatsSample {
                generation,
                population,
                density,
                timestamp: now,
            };

            if inner.history.len() >= MAX_HISTORY {
                inner.history.pop_front();
            }
            inner.history.push_back(sample);
        }
    }

    /// Update total cell count (e.g. after grid resize).
    pub fn set_total_cells(&self, total: u64) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.total_cells = total;
        }
    }

    /// Clear history (e.g. after grid reset).
    pub fn clear(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.history.clear();
            inner.gen_rate = 0.0;
            inner.last_gen = 0;
            inner.last_rate_time = Instant::now();
        }
    }

    /// Get the current generation rate.
    pub fn gen_rate(&self) -> f64 {
        self.inner.lock().map(|i| i.gen_rate).unwrap_or(0.0)
    }

    /// Get the latest population value.
    pub fn latest_population(&self) -> u64 {
        self.inner
            .lock()
            .ok()
            .and_then(|i| i.history.back().map(|s| s.population))
            .unwrap_or(0)
    }

    /// Get the latest density value.
    pub fn latest_density(&self) -> f64 {
        self.inner
            .lock()
            .ok()
            .and_then(|i| i.history.back().map(|s| s.density))
            .unwrap_or(0.0)
    }

    /// Get a snapshot of population history as (generation, population) pairs.
    pub fn population_history(&self) -> Vec<[f64; 2]> {
        self.inner
            .lock()
            .map(|i| {
                i.history
                    .iter()
                    .map(|s| [s.generation as f64, s.population as f64])
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get a snapshot of density history as (generation, density) pairs.
    pub fn density_history(&self) -> Vec<[f64; 2]> {
        self.inner
            .lock()
            .map(|i| {
                i.history
                    .iter()
                    .map(|s| [s.generation as f64, s.density])
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Shared data for the sampling thread to read from the main thread.
pub struct SamplingBridge {
    pub generation: Arc<Mutex<u64>>,
    pub population: Arc<Mutex<u64>>,
}

impl SamplingBridge {
    pub fn new() -> Self {
        Self {
            generation: Arc::new(Mutex::new(0)),
            population: Arc::new(Mutex::new(0)),
        }
    }

    /// Update the bridge values from the main thread.
    pub fn update(&self, generation: u64, population: u64) {
        if let Ok(mut g) = self.generation.lock() {
            *g = generation;
        }
        if let Ok(mut p) = self.population.lock() {
            *p = population;
        }
    }
}

/// Spawn a background thread that periodically samples stats from the bridge
/// and records them into the Stats store.
pub fn spawn_sampling_thread(
    stats: Stats,
    bridge: Arc<SamplingBridge>,
    interval: Duration,
) -> thread::JoinHandle<()> {
    thread::spawn(move || loop {
        thread::sleep(interval);

        let cur_gen = bridge
            .generation
            .lock()
            .map(|g| *g)
            .unwrap_or(0);
        let pop = bridge
            .population
            .lock()
            .map(|p| *p)
            .unwrap_or(0);

        stats.record(cur_gen, pop);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_record_and_read() {
        let stats = Stats::new(100);
        stats.record(1, 25);
        assert_eq!(stats.latest_population(), 25);
        assert!((stats.latest_density() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_history() {
        let stats = Stats::new(1000);
        for i in 0..10 {
            stats.record(i, i * 100);
        }
        let hist = stats.population_history();
        assert_eq!(hist.len(), 10);
        assert!((hist[0][0] - 0.0).abs() < f64::EPSILON);
        assert!((hist[9][1] - 900.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_clear() {
        let stats = Stats::new(100);
        stats.record(1, 50);
        stats.clear();
        assert_eq!(stats.population_history().len(), 0);
    }

    #[test]
    fn test_stats_max_history() {
        let stats = Stats::new(100);
        for i in 0..600 {
            stats.record(i, 50);
        }
        assert!(stats.population_history().len() <= MAX_HISTORY);
    }

    #[test]
    fn test_sampling_bridge() {
        let bridge = SamplingBridge::new();
        bridge.update(42, 1000);
        assert_eq!(*bridge.generation.lock().unwrap(), 42);
        assert_eq!(*bridge.population.lock().unwrap(), 1000);
    }
}
