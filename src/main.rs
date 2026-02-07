mod app;
mod camera;
mod grid;
mod renderer;
mod simulation;
mod stats;
mod ui;

use winit::event_loop::EventLoop;

fn main() {
    env_logger::init();

    log::info!("CatConway - GPU Accelerated Conway's Game of Life");
    log::info!("Controls:");
    log::info!("  Space       - Pause / Resume");
    log::info!("  Right Arrow - Step (when paused)");
    log::info!("  Up/Down     - Speed up / slow down");
    log::info!("  Mouse Drag  - Pan");
    log::info!("  Scroll      - Zoom");
    log::info!("  H           - Reset camera");
    log::info!("  R           - Randomize grid");
    log::info!("  C           - Clear grid");
    log::info!("  N           - Next rule set");
    log::info!("  1-8         - Load pattern / rule");
    log::info!("  Escape      - Quit");
    log::info!("  Use the menu bar and sidebar for additional controls.");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = app::App::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}
