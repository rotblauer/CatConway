use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::camera::Camera;
use crate::classify::{self, ClassifyHandle};
use crate::export::{self, ExportConfig, RuleContext};
use crate::favorites::Favorites;
use crate::grid::{
    Grid, Rules, pattern_acorn, pattern_glider, pattern_gosper_gun, pattern_lwss,
    pattern_r_pentomino,
};
use crate::renderer::Renderer;
use crate::search::{self, SearchHandle};
use crate::simulation::Simulation;
use crate::stats::{SamplingBridge, Stats, spawn_sampling_thread};
use crate::ui::{self, ClassifyInfo, FavoritesInfo, PatternInfo, RuleInfo, SearchInfo, UiState};

/// Default grid dimension (square).
const DEFAULT_GRID_SIZE: u32 = 1024;

/// Initial random fill density.
const INITIAL_DENSITY: f64 = 0.25;

/// Target simulation steps per second at speed=1.
const BASE_SPEED: f64 = 15.0;

/// Application state managing the simulation, rendering, and user interaction.
pub struct App {
    /// GPU resources (initialized after window creation).
    gpu: Option<GpuState>,
    /// Current camera.
    camera: Camera,
    /// Grid state (CPU side, used for initialization).
    grid: Grid,
    /// Whether the simulation is running.
    running: bool,
    /// Simulation speed multiplier.
    speed: f64,
    /// Time accumulator for stepping.
    step_accumulator: f64,
    last_frame: Instant,
    /// Current rule set.
    current_rule_idx: usize,
    /// Mouse drag state.
    dragging: bool,
    last_mouse_pos: Option<(f64, f64)>,
    /// Statistics tracking.
    stats: Stats,
    /// Bridge for the sampling thread.
    sampling_bridge: Arc<SamplingBridge>,
    /// Shutdown flag for the sampling thread.
    _sampling_shutdown: Arc<AtomicBool>,
    /// UI state.
    ui_state: UiState,
    /// Background rule search handle.
    search_handle: Option<SearchHandle>,
    /// Background behavior classification handle.
    classify_handle: Option<ClassifyHandle>,
    /// Favorites store.
    favorites: Favorites,
}

struct GpuState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    simulation: Simulation,
    renderer: Renderer,
    /// egui integration state.
    egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

/// The available rule sets for exploration.
fn rule_sets() -> Vec<(&'static str, Rules)> {
    vec![
        ("Conway B3/S23", Rules::conway()),
        ("HighLife B36/S23", Rules::highlife()),
        ("Day&Night B3678/S34678", Rules::day_and_night()),
        ("Seeds B2/S", Rules::seeds()),
        ("Life w/o Death B3/S*", Rules::life_without_death()),
        ("Bugs B3-5/S4-8/R2", Rules::bugs()),
        ("Globe B7-11/S7-11/R2", Rules::globe()),
        ("Majority B5-8/S4-10/R2", Rules::majority()),
    ]
}

fn pattern_list() -> Vec<(&'static str, &'static str, Vec<(i32, i32)>)> {
    vec![
        ("Glider", "1", pattern_glider()),
        ("R-pentomino", "2", pattern_r_pentomino()),
        ("Acorn", "3", pattern_acorn()),
        ("Gosper Gun", "4", pattern_gosper_gun()),
        ("LWSS", "5", pattern_lwss()),
    ]
}

impl App {
    pub fn new() -> Self {
        let mut grid = Grid::new(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE);
        grid.randomize(INITIAL_DENSITY);

        let total_cells = (grid.width as u64) * (grid.height as u64);
        let stats = Stats::new(total_cells);
        let sampling_bridge = Arc::new(SamplingBridge::new());

        // Spawn the background sampling thread.
        let (_handle, sampling_shutdown) = spawn_sampling_thread(
            stats.clone(),
            sampling_bridge.clone(),
            Duration::from_millis(100),
        );

        Self {
            gpu: None,
            camera: Camera::new(),
            grid,
            running: true,
            speed: 1.0,
            step_accumulator: 0.0,
            last_frame: Instant::now(),
            current_rule_idx: 0,
            dragging: false,
            last_mouse_pos: None,
            stats,
            sampling_bridge,
            _sampling_shutdown: sampling_shutdown,
            ui_state: UiState::new(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE),
            search_handle: None,
            classify_handle: None,
            favorites: Favorites::new(),
        }
    }

    fn initialize_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("No suitable GPU adapter found");

        log::info!("GPU adapter: {:?}", adapter.get_info().name);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
            None,
        ))
        .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        self.camera.aspect = config.width as f32 / config.height as f32;

        let simulation = Simulation::new(&device, &self.grid.cells, self.grid.sim_params());
        let renderer = Renderer::new(&device, surface_format);

        // ── egui setup ──
        let egui_ctx = egui::Context::default();
        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            &window,
            Some(window.scale_factor() as f32),
            None, // theme
            None, // max_texture_side
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, surface_format, None, 1, false);

        self.gpu = Some(GpuState {
            window,
            surface,
            device,
            queue,
            config,
            simulation,
            renderer,
            egui_ctx,
            egui_winit,
            egui_renderer,
        });
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if let Some(ref mut gpu) = self.gpu {
            if new_size.width > 0 && new_size.height > 0 {
                gpu.config.width = new_size.width;
                gpu.config.height = new_size.height;
                gpu.surface.configure(&gpu.device, &gpu.config);
                self.camera.aspect = new_size.width as f32 / new_size.height as f32;
            }
        }
    }

    fn render_frame(&mut self) {
        if self.gpu.is_none() {
            return;
        }

        // Timing
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f64();
        self.last_frame = now;

        // ── Phase 1: Simulation step ──
        if self.running {
            self.step_accumulator += dt * BASE_SPEED * self.speed;
            let steps = self.step_accumulator as u32;
            self.step_accumulator -= steps as f64;

            if steps > 0 {
                let gpu = self.gpu.as_mut().unwrap();
                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Compute Encoder"),
                        });
                for _ in 0..steps.min(10) {
                    gpu.simulation.step(&mut encoder);
                }
                gpu.queue.submit(std::iter::once(encoder.finish()));
            }
        }

        // Update sampling bridge for the background stats thread.
        {
            let gpu = self.gpu.as_ref().unwrap();
            self.sampling_bridge
                .update(gpu.simulation.generation, self.grid.population());
        }

        // ── Phase 2: egui UI pass ──
        let (actions, paint_jobs, screen_descriptor, textures_delta, _pixels_per_point) = {
            let gpu = self.gpu.as_mut().unwrap();
            let raw_input = gpu.egui_winit.take_egui_input(&gpu.window);
            gpu.egui_ctx.begin_pass(raw_input);

            let rule_infos: Vec<RuleInfo> = rule_sets()
                .iter()
                .enumerate()
                .map(|(i, (name, _))| RuleInfo {
                    name,
                    selected: i == self.current_rule_idx,
                })
                .collect();

            let pat_infos: Vec<PatternInfo> = pattern_list()
                .iter()
                .map(|(name, key, _)| PatternInfo { name, key })
                .collect();

            let search_info = if let Some(ref handle) = self.search_handle {
                let progress = handle.progress();
                let results = handle.results();
                SearchInfo {
                    active: true,
                    running: progress.running,
                    paused: progress.paused,
                    total_examined: progress.total_examined,
                    total_interesting: progress.total_interesting,
                    result_labels: results.iter().map(|r| r.label.clone()).collect(),
                }
            } else {
                SearchInfo {
                    active: false,
                    running: false,
                    paused: false,
                    total_examined: 0,
                    total_interesting: 0,
                    result_labels: Vec::new(),
                }
            };

            let classify_info = if let Some(ref handle) = self.classify_handle {
                let progress = handle.progress();
                let results = handle.results();
                ClassifyInfo {
                    active: true,
                    running: progress.running,
                    paused: progress.paused,
                    total_examined: progress.total_examined,
                    classified_count: progress.classified_count,
                    class_counts: progress.class_counts,
                    results,
                    cluster_summaries: progress.cluster_summaries,
                }
            } else {
                ClassifyInfo {
                    active: false,
                    running: false,
                    paused: false,
                    total_examined: 0,
                    classified_count: 0,
                    class_counts: std::collections::HashMap::new(),
                    results: Vec::new(),
                    cluster_summaries: Vec::new(),
                }
            };

            let favorites_info = FavoritesInfo {
                is_current_favorite: self.favorites.is_favorite(&self.grid.rules),
                entries: self
                    .favorites
                    .entries()
                    .iter()
                    .map(|f| f.label.clone())
                    .collect(),
            };

            let actions = ui::draw_ui(
                &gpu.egui_ctx,
                &mut self.ui_state,
                self.running,
                self.speed,
                gpu.simulation.generation,
                &rule_infos,
                &pat_infos,
                &self.stats,
                self.grid.width,
                self.grid.height,
                &search_info,
                &classify_info,
                &favorites_info,
            );

            let egui_output = gpu.egui_ctx.end_pass();
            gpu.egui_winit
                .handle_platform_output(&gpu.window, egui_output.platform_output);

            let ppp = egui_output.pixels_per_point;
            let paint_jobs = gpu.egui_ctx.tessellate(egui_output.shapes, ppp);

            let screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [gpu.config.width, gpu.config.height],
                pixels_per_point: gpu.window.scale_factor() as f32,
            };

            for (id, delta) in &egui_output.textures_delta.set {
                gpu.egui_renderer
                    .update_texture(&gpu.device, &gpu.queue, *id, delta);
            }

            let mut buf_encoder =
                gpu.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            gpu.egui_renderer.update_buffers(
                &gpu.device,
                &gpu.queue,
                &mut buf_encoder,
                &paint_jobs,
                &screen_descriptor,
            );
            gpu.queue.submit(std::iter::once(buf_encoder.finish()));

            (
                actions,
                paint_jobs,
                screen_descriptor,
                egui_output.textures_delta,
                ppp,
            )
        };

        // ── Phase 3: Handle UI actions (may modify grid, rules, etc.) ──
        self.handle_ui_actions(actions);

        // ── Phase 4: Render ──
        {
            let gpu = self.gpu.as_mut().unwrap();

            let output = match gpu.surface.get_current_texture() {
                Ok(tex) => tex,
                Err(wgpu::SurfaceError::Lost) => {
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    return;
                }
                Err(wgpu::SurfaceError::OutOfMemory) => {
                    log::error!("Out of GPU memory");
                    return;
                }
                Err(e) => {
                    log::warn!("Surface error: {e:?}");
                    return;
                }
            };

            let view = output
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            // Update camera uniform
            let cam_uniform = self.camera.uniform(self.grid.width, self.grid.height);
            gpu.renderer.update_camera(&gpu.queue, &cam_uniform);

            // Update render bind group to point to the current simulation buffer
            gpu.renderer
                .update_bind_group(&gpu.device, gpu.simulation.current_buffer());

            let mut encoder =
                gpu.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Render Encoder"),
                    });

            gpu.renderer.render(&mut encoder, &view);

            // Render egui on top
            {
                let mut pass = encoder
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("egui Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    })
                    .forget_lifetime();
                gpu.egui_renderer
                    .render(&mut pass, &paint_jobs, &screen_descriptor);
            }

            gpu.queue.submit(std::iter::once(encoder.finish()));
            output.present();

            for id in &textures_delta.free {
                gpu.egui_renderer.free_texture(id);
            }

            // Update window title with stats
            let generation = gpu.simulation.generation;
            let rule_list = rule_sets();
            let rule_name = if self.current_rule_idx < rule_list.len() {
                rule_list[self.current_rule_idx].0.to_string()
            } else {
                search::rules_to_label(&self.grid.rules)
            };
            let status = if self.running { "▶" } else { "⏸" };
            gpu.window.set_title(&format!(
                "CatConway | {status} Gen {generation} | {rule_name} | Speed: {:.1}x | {}×{}",
                self.speed, self.grid.width, self.grid.height,
            ));

            // Request next frame
            gpu.window.request_redraw();
        }
    }

    fn handle_ui_actions(&mut self, actions: ui::UiActions) {
        if actions.toggle_pause {
            self.running = !self.running;
        }
        if actions.step_once && !self.running {
            if let Some(ref mut gpu) = self.gpu {
                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Step Encoder"),
                        });
                gpu.simulation.step(&mut encoder);
                gpu.queue.submit(std::iter::once(encoder.finish()));
            }
        }
        if actions.randomize {
            self.grid.randomize(INITIAL_DENSITY);
            self.upload_grid();
            self.stats.clear();
        }
        if actions.clear {
            self.grid.clear();
            self.upload_grid();
            self.stats.clear();
        }
        if actions.reset_camera {
            self.camera.reset();
        }
        if let Some(idx) = actions.select_rule {
            let rules = rule_sets();
            if idx < rules.len() {
                self.current_rule_idx = idx;
                self.grid.rules = rules[idx].1;
                if let Some(ref gpu) = self.gpu {
                    gpu.simulation
                        .update_rules(&gpu.queue, self.grid.sim_params());
                }
            }
        }
        if let Some(idx) = actions.load_pattern {
            let patterns = pattern_list();
            if idx < patterns.len() {
                self.grid.clear();
                self.grid.place_pattern(&patterns[idx].2, None);
                self.upload_grid();
                self.stats.clear();
            }
        }
        if let Some(new_speed) = actions.speed_change {
            self.speed = new_speed.clamp(0.1, 100.0);
        }
        if let Some(res) = actions.apply_resolution {
            self.apply_new_resolution(res.width, res.height);
        }

        // ── Rule search actions ──
        if actions.start_search {
            if let Some(ref handle) = self.search_handle {
                handle.stop();
            }
            let mut search_config = search::SearchConfig::default();
            if let Ok(size) = self.ui_state.search_grid_size_str.parse::<u32>() {
                search_config.grid_size = size.max(16);
            }
            if let Ok(gens) = self.ui_state.search_generations_str.parse::<u32>() {
                search_config.generations = gens.max(1);
            }
            self.search_handle =
                Some(search::spawn_search(search_config));
            log::info!("Rule search started");
        }
        if actions.stop_search {
            if let Some(ref handle) = self.search_handle {
                handle.stop();
            }
            log::info!("Rule search stopped");
        }
        if actions.toggle_search_pause {
            if let Some(ref handle) = self.search_handle {
                let progress = handle.progress();
                if progress.paused {
                    handle.resume();
                    log::info!("Rule search resumed");
                } else {
                    handle.pause();
                    log::info!("Rule search paused");
                }
            }
        }
        if let Some(idx) = actions.apply_search_result {
            if let Some(ref handle) = self.search_handle {
                let results = handle.results();
                if let Some(result) = results.get(idx) {
                    self.grid.rules = result.rules;
                    self.current_rule_idx = usize::MAX;
                    self.grid.randomize(INITIAL_DENSITY);
                    self.upload_grid(); // resets generation to 0
                    self.stats.clear();
                    log::info!("Applied search result: {}", result.label);
                }
            }
        }

        // ── Classification actions ──
        if actions.start_classify {
            if let Some(ref handle) = self.classify_handle {
                handle.stop();
            }
            self.classify_handle =
                Some(classify::spawn_classify(classify::ClassifyConfig::default()));
            log::info!("Behavior classification started");
        }
        if actions.stop_classify {
            if let Some(ref handle) = self.classify_handle {
                handle.stop();
            }
            log::info!("Behavior classification stopped");
        }
        if actions.toggle_classify_pause {
            if let Some(ref handle) = self.classify_handle {
                let progress = handle.progress();
                if progress.paused {
                    handle.resume();
                    log::info!("Behavior classification resumed");
                } else {
                    handle.pause();
                    log::info!("Behavior classification paused");
                }
            }
        }
        if let Some(idx) = actions.apply_classified_rule {
            if let Some(ref handle) = self.classify_handle {
                let results = handle.results();
                if let Some(result) = results.get(idx) {
                    self.grid.rules = result.rules;
                    self.current_rule_idx = usize::MAX;
                    self.grid.randomize(INITIAL_DENSITY);
                    self.upload_grid(); // resets generation to 0
                    self.stats.clear();
                    log::info!(
                        "Applied classified rule: {} [{}]",
                        result.label,
                        result.behavior
                    );
                }
            }
        }
        if let Some(k) = actions.recluster {
            if let Some(ref handle) = self.classify_handle {
                handle.recluster(k);
                log::info!("Re-clustered with k={k}");
            }
        }

        // ── Favorites actions ──
        if actions.toggle_favorite {
            let is_now = self.favorites.toggle(&self.grid.rules);
            let label = search::rules_to_label(&self.grid.rules);
            if is_now {
                log::info!("Added to favorites: {label}");
            } else {
                log::info!("Removed from favorites: {label}");
            }
        }
        if let Some(idx) = actions.apply_favorite {
            let entries = self.favorites.entries().to_vec();
            if let Some(fav) = entries.get(idx) {
                self.grid.rules = fav.rules;
                self.current_rule_idx = usize::MAX;
                self.grid.randomize(INITIAL_DENSITY);
                self.upload_grid();
                self.stats.clear();
                log::info!("Applied favorite rule: {}", fav.label);
            }
        }

        // ── Export actions ──
        if actions.export_current || actions.export_all_favorites {
            // Build classification context for the scatter plot overlay.
            let context = if let Some(ref handle) = self.classify_handle {
                RuleContext {
                    classified_rules: handle.results(),
                    scatter_x: self.ui_state.scatter_x,
                    scatter_y: self.ui_state.scatter_y,
                }
            } else {
                RuleContext::default()
            };
            let mut config = ExportConfig::default();
            if let Ok(size) = self.ui_state.export_grid_size_str.parse::<u32>() {
                config.grid_size = size.max(16);
            }
            if let Ok(gens) = self.ui_state.export_generations_str.parse::<u32>() {
                config.generations = gens.max(1);
            }

            if actions.export_current {
                match export::export_gif(&self.grid.rules, &config, &context) {
                    Ok(result) => {
                        log::info!(
                            "Exported GIF: {} ({} frames) → {}",
                            result.label,
                            result.total_frames,
                            result.path.display()
                        );
                    }
                    Err(e) => log::error!("Export failed: {e}"),
                }
            }
            if actions.export_all_favorites {
                let entries = self.favorites.entries().to_vec();
                let rules_list: Vec<(String, Rules)> = entries
                    .iter()
                    .map(|f| (f.label.clone(), f.rules))
                    .collect();
                let results = export::export_multiple(&rules_list, &config, &context);
                for result in results {
                    match result {
                        Ok(r) => log::info!(
                            "Exported GIF: {} ({} frames) → {}",
                            r.label,
                            r.total_frames,
                            r.path.display()
                        ),
                        Err(e) => log::error!("Export failed: {e}"),
                    }
                }
            }
        }
    }

    fn apply_new_resolution(&mut self, width: u32, height: u32) {
        self.grid = Grid::new(width, height);
        let rule_list = rule_sets();
        if self.current_rule_idx < rule_list.len() {
            self.grid.rules = rule_list[self.current_rule_idx].1;
        }
        self.grid.randomize(INITIAL_DENSITY);

        let total = (width as u64) * (height as u64);
        self.stats.set_total_cells(total);
        self.stats.clear();

        if let Some(ref mut gpu) = self.gpu {
            gpu.simulation =
                Simulation::new(&gpu.device, &self.grid.cells, self.grid.sim_params());
        }

        self.ui_state.res_width_str = width.to_string();
        self.ui_state.res_height_str = height.to_string();
        self.camera.reset();
        log::info!("Resolution changed to {width}×{height}");
    }

    fn handle_key(&mut self, event: KeyEvent) {
        if event.state != ElementState::Pressed {
            return;
        }

        match event.logical_key {
            Key::Named(NamedKey::Space) => {
                self.running = !self.running;
                log::info!(
                    "Simulation {}",
                    if self.running { "resumed" } else { "paused" }
                );
            }
            Key::Named(NamedKey::ArrowRight) if !self.running => {
                // Single step
                if let Some(ref mut gpu) = self.gpu {
                    let mut encoder =
                        gpu.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Step Encoder"),
                            });
                    gpu.simulation.step(&mut encoder);
                    gpu.queue.submit(std::iter::once(encoder.finish()));
                }
            }
            Key::Named(NamedKey::ArrowUp) => {
                self.speed = (self.speed * 1.5).min(100.0);
                log::info!("Speed: {:.1}x", self.speed);
            }
            Key::Named(NamedKey::ArrowDown) => {
                self.speed = (self.speed / 1.5).max(0.1);
                log::info!("Speed: {:.1}x", self.speed);
            }
            Key::Named(NamedKey::Escape) => {
                self._sampling_shutdown.store(true, Ordering::Relaxed);
                if let Some(ref handle) = self.search_handle {
                    handle.stop();
                }
                if let Some(ref handle) = self.classify_handle {
                    handle.stop();
                }
                if let Some(ref gpu) = self.gpu {
                    gpu.window.set_visible(false);
                }
                std::process::exit(0);
            }
            Key::Character(ref c) => match c.as_str() {
                "r" => {
                    self.grid.randomize(INITIAL_DENSITY);
                    self.upload_grid();
                    self.stats.clear();
                    log::info!("Grid randomized");
                }
                "c" => {
                    self.grid.clear();
                    self.upload_grid();
                    self.stats.clear();
                    log::info!("Grid cleared");
                }
                "h" => {
                    self.camera.reset();
                    log::info!("Camera reset");
                }
                "1" => self.load_pattern("Glider", &pattern_glider()),
                "2" => self.load_pattern("R-pentomino", &pattern_r_pentomino()),
                "3" => self.load_pattern("Acorn", &pattern_acorn()),
                "4" => self.load_pattern("Gosper Gun", &pattern_gosper_gun()),
                "5" => self.load_pattern("LWSS", &pattern_lwss()),
                "n" => {
                    let rules = rule_sets();
                    self.current_rule_idx = if self.current_rule_idx < rules.len() {
                        (self.current_rule_idx + 1) % rules.len()
                    } else {
                        0
                    };
                    self.grid.rules = rules[self.current_rule_idx].1;
                    if let Some(ref gpu) = self.gpu {
                        gpu.simulation
                            .update_rules(&gpu.queue, self.grid.sim_params());
                    }
                    log::info!("Rules: {}", rules[self.current_rule_idx].0);
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn load_pattern(&mut self, name: &str, pattern: &[(i32, i32)]) {
        self.grid.clear();
        self.grid.place_pattern(pattern, None);
        self.upload_grid();
        self.stats.clear();
        log::info!("Loaded pattern: {name}");
    }

    fn upload_grid(&mut self) {
        if let Some(ref mut gpu) = self.gpu {
            gpu.simulation
                .upload(&gpu.queue, &self.grid.cells, self.grid.sim_params());
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            let attrs = WindowAttributes::default()
                .with_title("CatConway - GPU Accelerated Game of Life")
                .with_inner_size(PhysicalSize::new(1280u32, 900));

            let window = Arc::new(
                event_loop
                    .create_window(attrs)
                    .expect("Failed to create window"),
            );

            self.initialize_gpu(window);
            self.last_frame = Instant::now();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Let egui handle events first.
        if let Some(ref mut gpu) = self.gpu {
            let response = gpu.egui_winit.on_window_event(&gpu.window, &event);
            if response.consumed {
                gpu.window.request_redraw();
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                self._sampling_shutdown.store(true, Ordering::Relaxed);
                if let Some(ref handle) = self.search_handle {
                    handle.stop();
                }
                if let Some(ref handle) = self.classify_handle {
                    handle.stop();
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                self.resize(size);
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_key(event);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(pos) => pos.y / 50.0,
                };
                let factor = if scroll > 0.0 { 0.9 } else { 1.1 };
                self.camera.zoom(factor as f32);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.dragging = state == ElementState::Pressed;
                    if !self.dragging {
                        self.last_mouse_pos = None;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.dragging {
                    if let Some((lx, ly)) = self.last_mouse_pos {
                        if let Some(ref gpu) = self.gpu {
                            let w = gpu.config.width as f64;
                            let h = gpu.config.height as f64;
                            let dx = (lx - position.x) / w;
                            let dy = (ly - position.y) / h;
                            self.camera.pan(dx as f32, dy as f32);
                        }
                    }
                    self.last_mouse_pos = Some((position.x, position.y));
                }
            }
            _ => {}
        }
    }
}
