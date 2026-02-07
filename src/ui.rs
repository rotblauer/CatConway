use egui_plot::{Line, PlotPoints};

use crate::stats::Stats;

/// Desired grid resolution chosen via the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridResolution {
    pub width: u32,
    pub height: u32,
}

impl GridResolution {
    pub fn new(w: u32, h: u32) -> Self {
        Self {
            width: w.max(16),
            height: h.max(16),
        }
    }
}

/// Actions that the UI can emit for the main application to handle.
#[derive(Debug, Default)]
pub struct UiActions {
    pub toggle_pause: bool,
    pub step_once: bool,
    pub randomize: bool,
    pub clear: bool,
    pub reset_camera: bool,
    pub select_rule: Option<usize>,
    pub load_pattern: Option<usize>,
    pub speed_change: Option<f64>,
    pub apply_resolution: Option<GridResolution>,
}

/// Persistent state for the egui overlay UI.
pub struct UiState {
    /// Whether the sidebar panel is visible.
    pub show_sidebar: bool,
    /// Whether the stats panel is visible.
    pub show_stats: bool,
    /// Editable width/height strings for resolution input.
    pub res_width_str: String,
    pub res_height_str: String,
}

impl UiState {
    pub fn new(default_width: u32, default_height: u32) -> Self {
        Self {
            show_sidebar: true,
            show_stats: true,
            res_width_str: default_width.to_string(),
            res_height_str: default_height.to_string(),
        }
    }
}

/// Rule set display info.
pub struct RuleInfo {
    pub name: &'static str,
    pub selected: bool,
}

/// Pattern display info.
pub struct PatternInfo {
    pub name: &'static str,
    pub key: &'static str,
}

/// Draw the egui overlay UI and return any actions the user triggered.
pub fn draw_ui(
    ctx: &egui::Context,
    state: &mut UiState,
    running: bool,
    speed: f64,
    generation: u64,
    rules: &[RuleInfo],
    patterns: &[PatternInfo],
    stats: &Stats,
    grid_width: u32,
    grid_height: u32,
) -> UiActions {
    let mut actions = UiActions::default();

    // ── Top menu bar ──
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("Simulation", |ui| {
                let label = if running { "⏸ Pause" } else { "▶ Resume" };
                if ui.button(label).clicked() {
                    actions.toggle_pause = true;
                    ui.close_menu();
                }
                if ui.add_enabled(!running, egui::Button::new("→ Step")).clicked() {
                    actions.step_once = true;
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("🎲 Randomize").clicked() {
                    actions.randomize = true;
                    ui.close_menu();
                }
                if ui.button("🗑 Clear").clicked() {
                    actions.clear = true;
                    ui.close_menu();
                }
            });

            ui.menu_button("Rules", |ui| {
                for (idx, rule) in rules.iter().enumerate() {
                    let label = if rule.selected {
                        format!("● {}", rule.name)
                    } else {
                        format!("  {}", rule.name)
                    };
                    if ui.button(label).clicked() {
                        actions.select_rule = Some(idx);
                        ui.close_menu();
                    }
                }
            });

            ui.menu_button("Patterns", |ui| {
                for (idx, pat) in patterns.iter().enumerate() {
                    if ui.button(format!("[{}] {}", pat.key, pat.name)).clicked() {
                        actions.load_pattern = Some(idx);
                        ui.close_menu();
                    }
                }
            });

            ui.menu_button("View", |ui| {
                if ui.button("🏠 Reset Camera (H)").clicked() {
                    actions.reset_camera = true;
                    ui.close_menu();
                }
                ui.separator();
                ui.checkbox(&mut state.show_sidebar, "Show Sidebar");
                ui.checkbox(&mut state.show_stats, "Show Stats Panel");
            });
        });
    });

    // ── Left sidebar panel ──
    if state.show_sidebar {
        egui::SidePanel::left("sidebar")
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.heading("CatConway");
                ui.separator();

                // ── Status ──
                ui.label(format!(
                    "Status: {}",
                    if running { "▶ Running" } else { "⏸ Paused" }
                ));
                ui.label(format!("Generation: {generation}"));
                ui.label(format!("Grid: {grid_width} × {grid_height}"));
                ui.separator();

                // ── Speed control ──
                ui.label("Speed");
                ui.horizontal(|ui| {
                    if ui.button("−").clicked() {
                        actions.speed_change = Some(speed / 1.5);
                    }
                    ui.label(format!("{speed:.1}×"));
                    if ui.button("+").clicked() {
                        actions.speed_change = Some(speed * 1.5);
                    }
                });
                ui.separator();

                // ── Grid resolution ──
                ui.label("Grid Resolution");
                ui.horizontal(|ui| {
                    ui.label("W:");
                    ui.add(egui::TextEdit::singleline(&mut state.res_width_str).desired_width(50.0));
                    ui.label("H:");
                    ui.add(egui::TextEdit::singleline(&mut state.res_height_str).desired_width(50.0));
                });
                if ui.button("Apply Resolution").clicked() {
                    if let (Ok(w), Ok(h)) = (
                        state.res_width_str.parse::<u32>(),
                        state.res_height_str.parse::<u32>(),
                    ) {
                        actions.apply_resolution =
                            Some(GridResolution::new(w.clamp(16, 4096), h.clamp(16, 4096)));
                    }
                }
                ui.separator();

                // ── Quick stats ──
                let pop = stats.latest_population();
                let density = stats.latest_density();
                let rate = stats.gen_rate();
                ui.label(format!("Population: {pop}"));
                ui.label(format!("Density: {:.1}%", density * 100.0));
                ui.label(format!("Gen/sec: {rate:.0}"));
            });
    }

    // ── Bottom stats panel with plot ──
    if state.show_stats {
        egui::TopBottomPanel::bottom("stats_panel")
            .default_height(160.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Population History");
                });

                let history = stats.population_history();
                let points = PlotPoints::new(history);
                let line = Line::new(points).name("Population");

                egui_plot::Plot::new("pop_plot")
                    .height(120.0)
                    .allow_drag(false)
                    .allow_zoom(false)
                    .allow_scroll(false)
                    .show_axes(true)
                    .show(ui, |plot_ui| {
                        plot_ui.line(line);
                    });
            });
    }

    actions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_resolution_clamps_to_minimum() {
        let res = GridResolution::new(8, 12);
        assert_eq!(res.width, 16);
        assert_eq!(res.height, 16);
    }

    #[test]
    fn ui_state_initializes_defaults() {
        let state = UiState::new(128, 64);
        assert_eq!(state.res_width_str, "128");
        assert_eq!(state.res_height_str, "64");
        assert!(state.show_sidebar);
        assert!(state.show_stats);
    }
}
