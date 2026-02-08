use std::collections::HashMap;

use egui_plot::{Line, PlotPoints, Points};

use crate::classify::{BehaviorClass, ClassifiedRule, ClusterSummary, FEATURE_NAMES};
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
    /// Toggle the current rule as a favorite.
    pub toggle_favorite: bool,
    /// Apply a favorite rule by its index.
    pub apply_favorite: Option<usize>,
    /// Export a GIF animation of the current rule.
    pub export_current: bool,
    /// Export GIF animations of all favorite rules.
    pub export_all_favorites: bool,
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
    /// Editable search grid size string.
    pub search_grid_size_str: String,
    /// Editable search generations string.
    pub search_generations_str: String,
    /// Editable export grid size string.
    pub export_grid_size_str: String,
    /// Editable export generations string.
    pub export_generations_str: String,
    /// Editable classify grid size string.
    pub classify_grid_size_str: String,
    /// Editable classify generations string.
    pub classify_generations_str: String,
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
            search_grid_size_str: "640".to_string(),
            search_generations_str: "3000".to_string(),
            export_grid_size_str: "640".to_string(),
            export_generations_str: "3000".to_string(),
            classify_grid_size_str: "64".to_string(),
            classify_generations_str: "500".to_string(),
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
    /// Per-cluster interpretation summaries from the latest UMAP run.
    pub cluster_summaries: Vec<ClusterSummary>,
}

/// Information about favorites for UI display.
pub struct FavoritesInfo {
    /// Whether the current rule is favorited.
    pub is_current_favorite: bool,
    /// Labels and indices of all favorited rules.
    pub entries: Vec<String>,
}

/// Label text for the favorite/unfavorite toggle button.
fn favorite_button_label(is_favorite: bool) -> &'static str {
    if is_favorite {
        "★ Unfavorite Current Rule"
    } else {
        "☆ Favorite Current Rule"
    }
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
    favorites_info: &FavoritesInfo,
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

            ui.menu_button("Favorites", |ui| {
                if ui.button(favorite_button_label(favorites_info.is_current_favorite)).clicked() {
                    actions.toggle_favorite = true;
                    ui.close_menu();
                }
                ui.separator();
                if favorites_info.entries.is_empty() {
                    ui.label("No favorites yet");
                } else {
                    for (idx, label) in favorites_info.entries.iter().enumerate() {
                        if ui.button(format!("★ {label}")).clicked() {
                            actions.apply_favorite = Some(idx);
                            ui.close_menu();
                        }
                    }
                    ui.separator();
                    if ui.button("📹 Export All Favorites as GIF").clicked() {
                        actions.export_all_favorites = true;
                        ui.close_menu();
                    }
                }
            });

            ui.menu_button("Export", |ui| {
                if ui.button("📹 Export Current Rule as GIF").clicked() {
                    actions.export_current = true;
                    ui.close_menu();
                }
                if !favorites_info.entries.is_empty()
                    && ui.button("📹 Export All Favorites as GIF").clicked()
                {
                    actions.export_all_favorites = true;
                    ui.close_menu();
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

                    // ── Favorites ──
                    ui.separator();
                    ui.heading("Favorites");
                    {
                        if ui.button(favorite_button_label(favorites_info.is_current_favorite)).clicked() {
                            actions.toggle_favorite = true;
                        }

                        if !favorites_info.entries.is_empty() {
                            egui::ScrollArea::vertical()
                                .id_salt("favorites_scroll")
                                .max_height(120.0)
                                .show(ui, |ui| {
                                    for (idx, label) in
                                        favorites_info.entries.iter().enumerate()
                                    {
                                        ui.horizontal(|ui| {
                                            ui.label(format!("★ {label}"));
                                            if ui.small_button("Apply").clicked() {
                                                actions.apply_favorite = Some(idx);
                                            }
                                        });
                                    }
                                });

                            if ui.button("📹 Export All Favorites").clicked() {
                                actions.export_all_favorites = true;
                            }
                        }

                        if ui.button("📹 Export Current Rule").clicked() {
                            actions.export_current = true;
                        }

                        ui.label("Export Grid Size:");
                        ui.add(
                            egui::TextEdit::singleline(&mut state.export_grid_size_str)
                                .desired_width(80.0),
                        );
                        ui.label("Export Generations:");
                        ui.add(
                            egui::TextEdit::singleline(&mut state.export_generations_str)
                                .desired_width(80.0),
                        );
                    }

                    // ── Rule Search (legacy) ──
                    ui.separator();
                    ui.heading("Rule Search");
                    ui.label("Search Grid Size:");
                    ui.add(
                        egui::TextEdit::singleline(&mut state.search_grid_size_str)
                            .desired_width(80.0),
                    );
                    ui.label("Search Generations:");
                    ui.add(
                        egui::TextEdit::singleline(&mut state.search_generations_str)
                            .desired_width(80.0),
                    );
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

    // ── Right panel: Metrics & UMAP scatter plots (when classification is active) ──
    if classify_info.active && classify_info.classified_count > 0 {
        egui::SidePanel::right("metrics_panel")
            .default_width(320.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
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
                ui.label("💡 Click a point to run that rule");

                // Metrics scatter plot with click-to-select.
                let metrics_response = egui_plot::Plot::new("metrics_scatter")
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
                        // Return the pointer coordinate if clicked.
                        if plot_ui.response().clicked() {
                            plot_ui.pointer_coordinate()
                        } else {
                            None
                        }
                    });

                // Handle click on metrics scatter: find nearest point to click position.
                if let Some(click_pos) = metrics_response.inner {
                    let mut best_idx = None;
                    let mut best_dist = f64::INFINITY;
                    for (i, r) in classify_info.results.iter().enumerate() {
                        let fv = r.metrics.feature_vector();
                        let dx = fv[x_idx] - click_pos.x;
                        let dy = fv[y_idx] - click_pos.y;
                        let d = dx * dx + dy * dy;
                        if d < best_dist {
                            best_dist = d;
                            best_idx = Some(i);
                        }
                    }
                    if let Some(idx) = best_idx {
                        actions.apply_classified_rule = Some(idx);
                    }
                }

                // ── UMAP Projection scatter plot ──
                let has_umap = classify_info
                    .results
                    .iter()
                    .any(|r| r.umap_x.is_some() && r.umap_y.is_some());
                if has_umap {
                    ui.separator();
                    ui.heading("UMAP Projection");

                    // Determine max UMAP cluster index for color assignment.
                    let max_umap_cluster = classify_info
                        .results
                        .iter()
                        .filter_map(|r| r.umap_cluster)
                        .max()
                        .unwrap_or(0);

                    // Generate distinct colors for agnostic UMAP clusters.
                    let umap_colors: Vec<egui::Color32> = (0..=max_umap_cluster)
                        .map(|c| {
                            let hue = (c as f32) / ((max_umap_cluster + 1) as f32);
                            hsv_to_rgb(hue, 0.8, 0.9)
                        })
                        .collect();

                    // Legend for UMAP clusters with interpretations.
                    for (c, color) in umap_colors.iter().enumerate() {
                        let count = classify_info
                            .results
                            .iter()
                            .filter(|r| r.umap_cluster == Some(c))
                            .count();
                        if count > 0 {
                            let interp = classify_info
                                .cluster_summaries
                                .iter()
                                .find(|s| s.cluster_id == c)
                                .map(|s| s.interpretation.as_str())
                                .unwrap_or("");
                            ui.horizontal(|ui| {
                                let (rect, _) = ui.allocate_exact_size(
                                    egui::vec2(12.0, 12.0),
                                    egui::Sense::hover(),
                                );
                                ui.painter().rect_filled(rect, 0.0, *color);
                                if interp.is_empty() {
                                    ui.label(format!("Cluster {c}: {count}"));
                                } else {
                                    ui.label(format!("Cluster {c}: {count} — {interp}"));
                                }
                            });
                        }
                    }

                    ui.label("💡 Click a point to run that rule");

                    let umap_response = egui_plot::Plot::new("umap_scatter")
                        .height(200.0)
                        .x_axis_label("UMAP 1")
                        .y_axis_label("UMAP 2")
                        .show(ui, |plot_ui| {
                            // Plot each UMAP cluster as a separate series.
                            for (c, color) in umap_colors.iter().enumerate() {
                                let pts: Vec<[f64; 2]> = classify_info
                                    .results
                                    .iter()
                                    .filter(|r| {
                                        r.umap_cluster == Some(c)
                                            && r.umap_x.is_some()
                                            && r.umap_y.is_some()
                                    })
                                    .map(|r| [r.umap_x.unwrap(), r.umap_y.unwrap()])
                                    .collect();
                                if !pts.is_empty() {
                                    let points = Points::new(pts)
                                        .name(format!("Cluster {c}"))
                                        .color(*color)
                                        .radius(3.0);
                                    plot_ui.points(points);
                                }
                            }
                            // Return the pointer coordinate if clicked.
                            if plot_ui.response().clicked() {
                                plot_ui.pointer_coordinate()
                            } else {
                                None
                            }
                        });

                    // Handle click on UMAP scatter: find nearest point.
                    if let Some(click_pos) = umap_response.inner {
                        let mut best_idx = None;
                        let mut best_dist = f64::INFINITY;
                        for (i, r) in classify_info.results.iter().enumerate() {
                            if let (Some(ux), Some(uy)) = (r.umap_x, r.umap_y) {
                                let dx = ux - click_pos.x;
                                let dy = uy - click_pos.y;
                                let d = dx * dx + dy * dy;
                                if d < best_dist {
                                    best_dist = d;
                                    best_idx = Some(i);
                                }
                            }
                        }
                        if let Some(idx) = best_idx {
                            actions.apply_classified_rule = Some(idx);
                        }
                    }
                }

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
            });
    }

    actions
}

/// Convert HSV color to an egui Color32. `h` is in [0, 1), `s` and `v` in [0, 1].
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> egui::Color32 {
    let h6 = (h % 1.0) * 6.0;
    let sector = (h6 as u32) % 6;
    let c = v * s;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match sector {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    egui::Color32::from_rgb(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// Draw the classification section within the sidebar.
fn draw_classify_section(
    ui: &mut egui::Ui,
    state: &mut UiState,
    info: &ClassifyInfo,
    actions: &mut UiActions,
) {
    if !info.active {
        ui.label("Classify Grid Size:");
        ui.add(
            egui::TextEdit::singleline(&mut state.classify_grid_size_str)
                .desired_width(80.0),
        );
        ui.label("Classify Generations:");
        ui.add(
            egui::TextEdit::singleline(&mut state.classify_generations_str)
                .desired_width(80.0),
        );
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
        assert_eq!(state.search_grid_size_str, "640");
        assert_eq!(state.search_generations_str, "3000");
        assert_eq!(state.export_grid_size_str, "640");
        assert_eq!(state.export_generations_str, "3000");
        assert_eq!(state.classify_grid_size_str, "64");
        assert_eq!(state.classify_generations_str, "500");
    }
}
