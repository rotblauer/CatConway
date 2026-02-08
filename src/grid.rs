use rand::Rng;

/// Rules defining the dynamical system. Standard Conway is B3/S23.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rules {
    /// Bitmask: bit `i` set means a dead cell with `i` neighbors becomes alive.
    pub birth: u32,
    /// Bitmask: bit `i` set means a live cell with `i` neighbors survives.
    pub survival: u32,
    /// Neighborhood radius: 1 = Moore (8 neighbors), 2 = extended (24 neighbors).
    pub radius: u32,
}

impl Rules {
    /// Standard Conway's Game of Life: B3/S23
    pub fn conway() -> Self {
        Self {
            birth: 1 << 3,
            survival: (1 << 2) | (1 << 3),
            radius: 1,
        }
    }

    /// HighLife: B36/S23 - known for its replicator pattern
    pub fn highlife() -> Self {
        Self {
            birth: (1 << 3) | (1 << 6),
            survival: (1 << 2) | (1 << 3),
            radius: 1,
        }
    }

    /// Day & Night: B3678/S34678 - symmetric under on/off inversion
    pub fn day_and_night() -> Self {
        Self {
            birth: (1 << 3) | (1 << 6) | (1 << 7) | (1 << 8),
            survival: (1 << 3) | (1 << 4) | (1 << 6) | (1 << 7) | (1 << 8),
            radius: 1,
        }
    }

    /// Seeds: B2/S (no survival) - every cell dies, only birth
    pub fn seeds() -> Self {
        Self {
            birth: 1 << 2,
            survival: 0,
            radius: 1,
        }
    }

    /// Life without Death: B3/S012345678 - cells never die
    pub fn life_without_death() -> Self {
        Self {
            birth: 1 << 3,
            survival: 0x1FF, // bits 0-8 all set
            radius: 1,
        }
    }

    /// Bugs: B3-5/S4-8/R2 - amoeba-like blobs that move and split.
    /// Extended Moore neighborhood (radius 2, 24 neighbors).
    pub fn bugs() -> Self {
        Self {
            birth: bits(3..=5),
            survival: bits(4..=8),
            radius: 2,
        }
    }

    /// Globe: B7-11/S7-11/R2 - stable growing blob structures.
    /// Extended Moore neighborhood (radius 2, 24 neighbors).
    pub fn globe() -> Self {
        Self {
            birth: bits(7..=11),
            survival: bits(7..=11),
            radius: 2,
        }
    }

    /// Majority: B5-8/S4-10/R2 - majority-vote dynamics with organic boundaries.
    /// Extended Moore neighborhood (radius 2, 24 neighbors).
    pub fn majority() -> Self {
        Self {
            birth: bits(5..=8),
            survival: bits(4..=10),
            radius: 2,
        }
    }
}

/// Build a bitmask with bits set for each value in the inclusive range.
fn bits(range: std::ops::RangeInclusive<u32>) -> u32 {
    let mut mask = 0u32;
    for i in range {
        mask |= 1 << i;
    }
    mask
}

impl Default for Rules {
    fn default() -> Self {
        Self::conway()
    }
}

/// GPU-compatible simulation parameters (padded to 32 bytes for uniform alignment).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
    pub width: u32,
    pub height: u32,
    pub birth_rule: u32,
    pub survival_rule: u32,
    pub neighborhood_radius: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Manages the grid state on the CPU side (for initialization and pattern loading).
pub struct Grid {
    pub width: u32,
    pub height: u32,
    pub cells: Vec<u32>,
    pub rules: Rules,
}

impl Grid {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            cells: vec![0; (width * height) as usize],
            rules: Rules::conway(),
        }
    }

    /// Fill with random cells at the given density (0.0 = empty, 1.0 = full).
    pub fn randomize(&mut self, density: f64) {
        let mut rng = rand::thread_rng();
        for cell in &mut self.cells {
            *cell = if rng.gen_range(0.0..1.0) < density { 1 } else { 0 };
        }
    }

    /// Clear all cells.
    pub fn clear(&mut self) {
        self.cells.fill(0);
    }

    /// Set a single cell (with bounds wrapping).
    pub fn set(&mut self, x: i32, y: i32, alive: bool) {
        let w = self.width as i32;
        let h = self.height as i32;
        let wx = ((x % w) + w) % w;
        let wy = ((y % h) + h) % h;
        self.cells[(wy * w + wx) as usize] = if alive { 1 } else { 0 };
    }

    /// Get cell state (with bounds wrapping).
    pub fn get(&self, x: i32, y: i32) -> bool {
        let w = self.width as i32;
        let h = self.height as i32;
        let wx = ((x % w) + w) % w;
        let wy = ((y % h) + h) % h;
        self.cells[(wy * w + wx) as usize] == 1
    }

    /// Place a pattern at the given position (center of grid if None).
    pub fn place_pattern(&mut self, pattern: &[(i32, i32)], center: Option<(i32, i32)>) {
        let (cx, cy) = center.unwrap_or((self.width as i32 / 2, self.height as i32 / 2));
        for &(dx, dy) in pattern {
            self.set(cx + dx, cy + dy, true);
        }
    }

    /// Count live cells.
    pub fn population(&self) -> u64 {
        self.cells.iter().map(|&c| c as u64).sum()
    }

    /// Build GPU simulation parameters.
    pub fn sim_params(&self) -> SimParams {
        SimParams {
            width: self.width,
            height: self.height,
            birth_rule: self.rules.birth,
            survival_rule: self.rules.survival,
            neighborhood_radius: self.rules.radius,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

// ── Predefined patterns ──

/// Glider: small, moving pattern.
pub fn pattern_glider() -> Vec<(i32, i32)> {
    vec![(0, -1), (1, 0), (-1, 1), (0, 1), (1, 1)]
}

/// R-pentomino: a methuselah that runs for 1103 generations.
pub fn pattern_r_pentomino() -> Vec<(i32, i32)> {
    vec![(0, -1), (1, -1), (-1, 0), (0, 0), (0, 1)]
}

/// Acorn: a methuselah that takes 5206 generations to stabilize.
pub fn pattern_acorn() -> Vec<(i32, i32)> {
    vec![(-3, 0), (-2, 0), (-2, -2), (0, -1), (1, 0), (2, 0), (3, 0)]
}

/// Gosper glider gun: infinite growth pattern.
pub fn pattern_gosper_gun() -> Vec<(i32, i32)> {
    vec![
        // Left block
        (-18, 0), (-18, 1), (-17, 0), (-17, 1),
        // Left ship
        (-8, 0), (-8, 1), (-8, 2), (-7, -1), (-7, 3), (-6, -2), (-6, 4),
        (-5, -2), (-5, 4), (-4, 1), (-3, -1), (-3, 3), (-2, 0), (-2, 1),
        (-2, 2), (-1, 1),
        // Right ship
        (2, 0), (2, -1), (2, -2), (3, 0), (3, -1), (3, -2), (4, -3),
        (4, 1), (6, -4), (6, -3), (6, 1), (6, 2),
        // Right block
        (16, -1), (16, -2), (17, -1), (17, -2),
    ]
}

/// Lightweight spaceship (LWSS).
pub fn pattern_lwss() -> Vec<(i32, i32)> {
    vec![
        (-2, -1), (-1, -2), (0, -2), (1, -2), (2, -2),
        (2, -1), (2, 0), (1, 1), (-2, 0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_new() {
        let grid = Grid::new(100, 100);
        assert_eq!(grid.cells.len(), 10000);
        assert_eq!(grid.population(), 0);
    }

    #[test]
    fn test_grid_set_get() {
        let mut grid = Grid::new(10, 10);
        grid.set(3, 4, true);
        assert!(grid.get(3, 4));
        assert!(!grid.get(0, 0));
    }

    #[test]
    fn test_grid_wrapping() {
        let mut grid = Grid::new(10, 10);
        grid.set(-1, -1, true);
        assert!(grid.get(9, 9));
        grid.set(10, 10, true);
        assert!(grid.get(0, 0));
    }

    #[test]
    fn test_grid_randomize() {
        let mut grid = Grid::new(100, 100);
        grid.randomize(0.5);
        let pop = grid.population();
        // With 10000 cells at 50% density, population should be roughly 5000
        assert!(pop > 1000 && pop < 9000);
    }

    #[test]
    fn test_grid_clear() {
        let mut grid = Grid::new(10, 10);
        grid.randomize(1.0);
        assert!(grid.population() > 0);
        grid.clear();
        assert_eq!(grid.population(), 0);
    }

    #[test]
    fn test_place_pattern() {
        let mut grid = Grid::new(100, 100);
        grid.place_pattern(&pattern_glider(), None);
        assert_eq!(grid.population(), 5);
    }

    #[test]
    fn test_rules_conway() {
        let rules = Rules::conway();
        // Birth with exactly 3 neighbors
        assert_ne!(rules.birth & (1 << 3), 0);
        assert_eq!(rules.birth & (1 << 2), 0);
        // Survive with 2 or 3 neighbors
        assert_ne!(rules.survival & (1 << 2), 0);
        assert_ne!(rules.survival & (1 << 3), 0);
        assert_eq!(rules.survival & (1 << 4), 0);
    }

    #[test]
    fn test_sim_params() {
        let grid = Grid::new(256, 128);
        let p = grid.sim_params();
        assert_eq!(p.width, 256);
        assert_eq!(p.height, 128);
        assert_eq!(p.neighborhood_radius, 1);
    }

    #[test]
    fn test_extended_rules_radius() {
        assert_eq!(Rules::conway().radius, 1);
        assert_eq!(Rules::highlife().radius, 1);
        assert_eq!(Rules::bugs().radius, 2);
        assert_eq!(Rules::globe().radius, 2);
        assert_eq!(Rules::majority().radius, 2);
    }

    #[test]
    fn test_bits_helper() {
        let bugs = Rules::bugs();
        // B3-5: bits 3,4,5 set
        assert_ne!(bugs.birth & (1 << 3), 0);
        assert_ne!(bugs.birth & (1 << 4), 0);
        assert_ne!(bugs.birth & (1 << 5), 0);
        assert_eq!(bugs.birth & (1 << 2), 0);
        assert_eq!(bugs.birth & (1 << 6), 0);
    }
}
