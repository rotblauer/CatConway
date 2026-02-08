use std::collections::HashMap;

use egui_plot::{Line, PlotPoints, Points};

use crate::classify::{BehaviorClass, ClassifiedRule, FEATURE_NAMES};
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
    pub start_search: bool,
    pub stop_search: bool,
    pub toggle_search_pause: bool,
    pub apply_search_result: Option<usize>,
    /// Start the classification search.
    pub start_classify: bool,
    /// Stop the classification search.
    pub stop_classify: bool,
    /// Pause/resume the classification search.
    pub toggle_classify_pause: bool,
    /// Apply a classified rule by its index in the filtered results list.
    pub apply_classified_rule: Option<usize>,
    /// Re-cluster with the specified k.
    pub recluster: Option<usize>,
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
    /// Currently selected behavior class filter for classification results.
    pub classify_filter: Option<BehaviorClass>,
    /// X-axis metric index for the scatter plot.
    pub scatter_x: usize,
    /// Y-axis metric index for the scatter plot.
    pub scatter_y: usize,
    /// Number of clusters for k-means.
    pub cluster_k_str: String,
}

impl UiState {
    pub fn new(default_width: u32, default_height: u32) -> Self {
        Self {
            show_sidebar: true,
            show_stats: true,
            res_width_str: default_width.to_string(),
            res_height_str: default_height.to_string(),
            classify_filter: None,
            scatter_x: 1, // mean_density
            scatter_y: 0, // variation
            cluster_k_str: "8".to_string(),
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

/// Information about the background rule search for UI display.
pub struct SearchInfo {
    /// Whether a search handle exists.
    pub active: bool,
    /// Whether the search thread is still running.
    pub running: bool,
    /// Whether the search is paused.
    pub paused: bool,
    /// Total rules examined so far.
    pub total_examined: usize,
    /// Total interesting rules found.
    pub total_interesting: usize,
    /// Labels of discovered interesting rules.
    pub result_labels: Vec<String>,
}

/// Information about the behavior classification search for UI display.
pub struct ClassifyInfo {
    /// Whether a classify handle exists.
    pub active: bool,
    /// Whether the thread is still running.
    pub running: bool,
    /// Whether it is paused.
    pub paused: bool,
    /// Total rules examined.
    pub total_examined: usize,
    /// Total classified.
    pub classified_count: usize,
    /// Counts per behavior class.
    pub class_counts: HashMap<BehaviorClass, usize>,
    /// All classified results (or filtered subset).
    pub results: Vec<ClassifiedRule>,
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
    search_info: &SearchInfo,
    classify_info: &ClassifyInfo,
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
            .default_width(260.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
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
                        ui.add(
                            egui::TextEdit::singleline(&mut state.res_width_str)
                                .desired_width(50.0),
                        );
                        ui.label("H:");
                        ui.add(
                            egui::TextEdit::singleline(&mut state.res_height_str)
                                .desired_width(50.0),
                        );
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

                    // ── Rule Search (legacy) ──
                    ui.separator();
                    ui.heading("Rule Search");
                    if !search_info.active {
                        if ui.button("🔍 Start Search").clicked() {
                            actions.start_search = true;
                        }
                    } else {
                        let status = if !search_info.running {
                            "Complete"
                        } else if search_info.paused {
                            "⏸ Paused"
                        } else {
                            "▶ Running"
                        };
                        ui.label(format!("Status: {status}"));
                        ui.label(format!("Rules tested: {}", search_info.total_examined));
                        ui.label(format!("Interesting: {}", search_info.total_interesting));

                        ui.horizontal(|ui| {
                            if search_info.running {
                                if search_info.paused {
                                    if ui.button("▶ Resume").clicked() {
                                        actions.toggle_search_pause = true;
                                    }
                                } else if ui.button("⏸ Pause").clicked() {
                                    actions.toggle_search_pause = true;
                                }
                                if ui.button("⏹ Stop").clicked() {
                                    actions.stop_search = true;
                                }
                            } else if ui.button("🔍 New Search").clicked() {
                                actions.start_search = true;
                            }
                        });

                        if !search_info.result_labels.is_empty() {
                            ui.separator();
                            ui.label("Discovered Rules:");
                            egui::ScrollArea::vertical()
                                .id_salt("search_results_scroll")
                                .max_height(150.0)
                                .show(ui, |ui| {
                                    for (idx, label) in
                                        search_info.result_labels.iter().enumerate()
                                    {
                                        ui.horizontal(|ui| {
                                            ui.label(label.as_str());
                                            if ui.small_button("Apply").clicked() {
                                                actions.apply_search_result = Some(idx);
                                            }
                                        });
                                    }
                                });
                        }
                    }

                    // ── Behavior Classification ──
                    ui.separator();
                    ui.heading("Behavior Classification");
                    draw_classify_section(
                        ui,
                        state,
                        classify_info,
                        &mut actions,
                    );
                });
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

    // ── Right panel: Metrics scatter plot (when classification is active) ──
    if classify_info.active && classify_info.classified_count > 0 {
        egui::SidePanel::right("metrics_panel")
            .default_width(320.0)
            .show(ctx, |ui| {
                ui.heading("Metrics Visualization");
                ui.separator();

                // Axis selectors.
                ui.horizontal(|ui| {
                    ui.label("X:");
                    egui::ComboBox::from_id_salt("scatter_x")
                        .selected_text(FEATURE_NAMES[state.scatter_x])
                        .show_ui(ui, |ui| {
                            for (i, name) in FEATURE_NAMES.iter().enumerate() {
                                ui.selectable_value(&mut state.scatter_x, i, *name);
                            }
                        });
                });
                ui.horizontal(|ui| {
                    ui.label("Y:");
                    egui::ComboBox::from_id_salt("scatter_y")
                        .selected_text(FEATURE_NAMES[state.scatter_y])
                        .show_ui(ui, |ui| {
                            for (i, name) in FEATURE_NAMES.iter().enumerate() {
                                ui.selectable_value(&mut state.scatter_y, i, *name);
                            }
                        });
                });

                ui.separator();

                // Group results by behavior class and plot each group in a different color.
                let x_idx = state.scatter_x;
                let y_idx = state.scatter_y;

                let class_colors: Vec<(BehaviorClass, egui::Color32)> = vec![
                    (BehaviorClass::Dead, egui::Color32::from_rgb(128, 128, 128)),
                    (BehaviorClass::Static, egui::Color32::from_rgb(100, 149, 237)),
                    (BehaviorClass::Periodic, egui::Color32::from_rgb(255, 165, 0)),
                    (BehaviorClass::Explosive, egui::Color32::from_rgb(255, 69, 0)),
                    (BehaviorClass::Chaotic, egui::Color32::from_rgb(220, 20, 60)),
                    (BehaviorClass::Complex, egui::Color32::from_rgb(50, 205, 50)),
                    (BehaviorClass::Declining, egui::Color32::from_rgb(138, 43, 226)),
                    (BehaviorClass::Growing, egui::Color32::from_rgb(0, 191, 255)),
                ];

                // Legend.
                for (class, color) in &class_colors {
                    let count = classify_info.class_counts.get(class).copied().unwrap_or(0);
                    if count > 0 {
                        ui.horizontal(|ui| {
                            let (rect, _) =
                                ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                            ui.painter().rect_filled(rect, 0.0, *color);
                            ui.label(format!("{class}: {count}"));
                        });
                    }
                }

                ui.separator();

                egui_plot::Plot::new("metrics_scatter")
                    .height(200.0)
                    .x_axis_label(FEATURE_NAMES[x_idx])
                    .y_axis_label(FEATURE_NAMES[y_idx])
                    .show(ui, |plot_ui| {
                        for (class, color) in &class_colors {
                            let pts: Vec<[f64; 2]> = classify_info
                                .results
                                .iter()
                                .filter(|r| r.behavior == *class)
                                .map(|r| {
                                    let fv = r.metrics.feature_vector();
                                    [fv[x_idx], fv[y_idx]]
                                })
                                .collect();
                            if !pts.is_empty() {
                                let points = Points::new(pts)
                                    .name(format!("{class}"))
                                    .color(*color)
                                    .radius(3.0);
                                plot_ui.points(points);
                            }
                        }
                    });

                // Clustering controls.
                ui.separator();
                ui.heading("Clustering");
                ui.horizontal(|ui| {
                    ui.label("k:");
                    ui.add(
                        egui::TextEdit::singleline(&mut state.cluster_k_str).desired_width(30.0),
                    );
                    if ui.button("Re-cluster").clicked() {
                        if let Ok(k) = state.cluster_k_str.parse::<usize>() {
                            actions.recluster = Some(k.clamp(2, 20));
                        }
                    }
                });
            });
    }

    actions
}

/// Draw the classification section within the sidebar.
fn draw_classify_section(
    ui: &mut egui::Ui,
    state: &mut UiState,
    info: &ClassifyInfo,
    actions: &mut UiActions,
) {
    if !info.active {
        if ui.button("🧬 Start Classification").clicked() {
            actions.start_classify = true;
        }
        return;
    }

    let status = if !info.running {
        "Complete"
    } else if info.paused {
        "⏸ Paused"
    } else {
        "▶ Running"
    };
    ui.label(format!("Status: {status}"));
    ui.label(format!("Rules tested: {}", info.total_examined));
    ui.label(format!("Classified: {}", info.classified_count));

    // Show class breakdown.
    for &class in BehaviorClass::all() {
        let count = info.class_counts.get(&class).copied().unwrap_or(0);
        if count > 0 {
            ui.label(format!("  {class}: {count}"));
        }
    }

    ui.horizontal(|ui| {
        if info.running {
            if info.paused {
                if ui.button("▶ Resume").clicked() {
                    actions.toggle_classify_pause = true;
                }
            } else if ui.button("⏸ Pause").clicked() {
                actions.toggle_classify_pause = true;
            }
            if ui.button("⏹ Stop").clicked() {
                actions.stop_classify = true;
            }
        } else if ui.button("🧬 New Classification").clicked() {
            actions.start_classify = true;
        }
    });

    // Filter selector.
    if info.classified_count > 0 {
        ui.separator();
        ui.label("Filter by behavior:");
        ui.horizontal(|ui| {
            if ui
                .selectable_label(state.classify_filter.is_none(), "All")
                .clicked()
            {
                state.classify_filter = None;
            }
        });
        for &class in BehaviorClass::all() {
            let count = info.class_counts.get(&class).copied().unwrap_or(0);
            if count > 0 {
                let selected = state.classify_filter == Some(class);
                if ui
                    .selectable_label(selected, format!("{class} ({count})"))
                    .clicked()
                {
                    state.classify_filter = Some(class);
                }
            }
        }

        // Show filtered results list with Apply buttons.
        ui.separator();
        ui.label("Classified Rules:");
        let filtered: Vec<(usize, &ClassifiedRule)> = info
            .results
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                state
                    .classify_filter
                    .map_or(true, |c| r.behavior == c)
            })
            .collect();

        egui::ScrollArea::vertical()
            .id_salt("classify_results_scroll")
            .max_height(200.0)
            .show(ui, |ui| {
                for (idx, result) in &filtered {
                    ui.horizontal(|ui| {
                        let cluster_str = result
                            .cluster
                            .map(|c| format!(" C{c}"))
                            .unwrap_or_default();
                        ui.label(format!(
                            "{} [{}{}]",
                            result.label, result.behavior, cluster_str
                        ));
                        if ui.small_button("Apply").clicked() {
                            actions.apply_classified_rule = Some(*idx);
                        }
                    });
                }
            });
    }
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
        assert!(state.classify_filter.is_none());
        assert_eq!(state.scatter_x, 1);
        assert_eq!(state.scatter_y, 0);
    }
}
