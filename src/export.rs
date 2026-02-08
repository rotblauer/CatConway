use std::path::PathBuf;

use rand::Rng;

use crate::classify::ClassifiedRule;
use crate::grid::Rules;
use crate::search::{cpu_step, rules_to_label};

/// Configuration for GIF animation export.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Grid size (square) for the CPU simulation.
    pub grid_size: u32,
    /// Initial random fill density.
    pub density: f64,
    /// Number of generations to animate.
    pub generations: u32,
    /// Output directory for generated GIF files.
    pub output_dir: PathBuf,
    /// Delay between frames in hundredths of a second (e.g. 10 = 100ms).
    pub frame_delay: u16,
    /// Pixel scale: each cell is rendered as scale×scale pixels.
    pub cell_scale: u32,
    /// Height in pixels of the text overlay area at the top.
    pub overlay_height: u32,
    /// Sample every Nth generation to keep file size reasonable.
    pub frame_step: u32,
    /// Size in pixels (square) for the mini scatter plots in the overlay.
    pub plot_size: u32,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            grid_size: 64,
            density: 0.25,
            generations: 300,
            output_dir: PathBuf::from("animations"),
            frame_delay: 8,
            cell_scale: 4,
            overlay_height: 60,
            frame_step: 2,
            plot_size: 50,
        }
    }
}

/// Context about the rule's position in classification space.
/// When available, mini scatter plots are rendered in the overlay to show
/// where this rule sits among all classified rules.
#[derive(Debug, Clone, Default)]
pub struct RuleContext {
    /// All classified rules for rendering the scatter background.
    pub classified_rules: Vec<ClassifiedRule>,
    /// The metrics scatter X-axis feature index.
    pub scatter_x: usize,
    /// The metrics scatter Y-axis feature index.
    pub scatter_y: usize,
}

/// Export result containing the path and summary statistics.
#[derive(Debug, Clone)]
pub struct ExportResult {
    pub path: PathBuf,
    pub label: String,
    pub total_frames: u32,
    pub final_population: u64,
}

/// Export a GIF animation for the given rule set.
///
/// The animation shows the grid evolving over time with an overlay
/// displaying the rule label, generation counter, population stats,
/// and (when classification context is available) mini scatter plots
/// showing the rule's position in metrics space and UMAP space.
pub fn export_gif(
    rules: &Rules,
    config: &ExportConfig,
    context: &RuleContext,
) -> Result<ExportResult, String> {
    let label = rules_to_label(rules);
    let size = config.grid_size;
    let total_cells = (size * size) as usize;

    // Initialize grid.
    let mut cells = vec![0u32; total_cells];
    let mut rng = rand::thread_rng();
    for cell in &mut cells {
        *cell = u32::from(rng.gen_range(0.0..1.0) < config.density);
    }

    // Ensure output directory exists.
    std::fs::create_dir_all(&config.output_dir)
        .map_err(|e| format!("Failed to create output dir: {e}"))?;

    // Build output filename from the rule label.
    let safe_name: String = label
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' })
        .collect();
    let path = config.output_dir.join(format!("{safe_name}.gif"));

    let img_width = size * config.cell_scale;
    let img_height = size * config.cell_scale + config.overlay_height;

    let file =
        std::fs::File::create(&path).map_err(|e| format!("Failed to create file: {e}"))?;

    let mut encoder = gif::Encoder::new(file, img_width as u16, img_height as u16, &[])
        .map_err(|e| format!("Failed to create GIF encoder: {e}"))?;

    encoder
        .set_repeat(gif::Repeat::Infinite)
        .map_err(|e| format!("Failed to set repeat: {e}"))?;

    // Pre-render the static overlay portions (scatter plots don't change per frame).
    let static_overlay = render_static_overlay(
        img_width,
        config.overlay_height,
        &label,
        rules,
        context,
        config.plot_size,
    );

    let mut frame_count = 0u32;
    let mut final_pop = 0u64;

    for generation in 0..config.generations {
        let pop: u64 = cells.iter().map(|&c| u64::from(c)).sum();
        final_pop = pop;
        let density = pop as f64 / total_cells as f64;

        if generation % config.frame_step == 0 {
            let mut pixels = vec![0u8; (img_width * img_height * 4) as usize];

            // Copy static overlay then stamp dynamic text on top.
            let overlay_bytes = (img_width * config.overlay_height * 4) as usize;
            pixels[..overlay_bytes].copy_from_slice(&static_overlay[..overlay_bytes]);

            // Render dynamic stats text (generation, pop, density).
            let stats_text = format!(
                "Gen:{generation} Pop:{pop} D:{:.1}%",
                density * 100.0
            );
            render_text(&mut pixels, img_width, 2, 11, &stats_text, 0xCC, 0xCC, 0xCC);

            // Render grid cells.
            let y_offset = config.overlay_height;
            for gy in 0..size {
                for gx in 0..size {
                    let alive = cells[(gy * size + gx) as usize] == 1;
                    let (r, g, b) = if alive {
                        (0x40, 0xFF, 0x40) // bright green
                    } else {
                        (0x0A, 0x0A, 0x2E) // dark blue
                    };

                    for dy in 0..config.cell_scale {
                        for dx in 0..config.cell_scale {
                            let px = gx * config.cell_scale + dx;
                            let py = (gy * config.cell_scale + dy) + y_offset;
                            let idx = ((py * img_width + px) * 4) as usize;
                            pixels[idx] = r;
                            pixels[idx + 1] = g;
                            pixels[idx + 2] = b;
                            pixels[idx + 3] = 0xFF;
                        }
                    }
                }
            }

            // speed=1 gives best LZW compression quality.
            let mut frame = gif::Frame::from_rgba_speed(
                img_width as u16,
                img_height as u16,
                &mut pixels,
                1,
            );
            frame.delay = config.frame_delay;

            encoder
                .write_frame(&frame)
                .map_err(|e| format!("Failed to write frame: {e}"))?;

            frame_count += 1;
        }

        if pop == 0 {
            break;
        }

        cells = cpu_step(&cells, size, size, rules);
    }

    Ok(ExportResult {
        path,
        label,
        total_frames: frame_count,
        final_population: final_pop,
    })
}

/// Export GIF animations for multiple rules, sharing the same context.
pub fn export_multiple(
    rules_list: &[(String, Rules)],
    config: &ExportConfig,
    context: &RuleContext,
) -> Vec<Result<ExportResult, String>> {
    rules_list
        .iter()
        .map(|(_, rules)| export_gif(rules, config, context))
        .collect()
}

// ── Static overlay rendering ────────────────────────────────────────────────

/// Render the static portions of the overlay: rule label, behavior class,
/// and mini scatter plots (metrics + UMAP) showing the rule's position.
fn render_static_overlay(
    img_width: u32,
    overlay_height: u32,
    label: &str,
    rules: &Rules,
    context: &RuleContext,
    plot_size: u32,
) -> Vec<u8> {
    let mut pixels = vec![0u8; (img_width * overlay_height * 4) as usize];

    // Fill overlay background with dark gray.
    for y in 0..overlay_height {
        for x in 0..img_width {
            let idx = ((y * img_width + x) * 4) as usize;
            pixels[idx] = 0x1A;
            pixels[idx + 1] = 0x1A;
            pixels[idx + 2] = 0x1A;
            pixels[idx + 3] = 0xFF;
        }
    }

    // Rule label in white (row 0).
    render_text(&mut pixels, img_width, 2, 2, label, 0xFF, 0xFF, 0xFF);

    // Row 1 (y=11) is reserved for dynamic gen/pop/density text.

    // Find this rule in classified results to get behavior class.
    let this_classified = context
        .classified_rules
        .iter()
        .find(|cr| cr.rules == *rules);

    if let Some(cr) = this_classified {
        let class_str = format!("[{}]", cr.behavior);
        let text_end_x = 2 + (label.len() as u32 + 1) * 6;
        render_text(
            &mut pixels,
            img_width,
            text_end_x,
            2,
            &class_str,
            0x50, 0xC8, 0x50,
        );
    }

    // Render mini scatter plots if we have classified rules.
    if !context.classified_rules.is_empty() {
        let x_idx = context.scatter_x;
        let y_idx = context.scatter_y;

        // Metrics scatter: place at right side of overlay.
        let metrics_x = img_width.saturating_sub(plot_size * 2 + 8);
        let metrics_y = 2u32;
        if overlay_height >= plot_size + 4 {
            render_mini_scatter(
                &mut pixels,
                img_width,
                overlay_height,
                metrics_x,
                metrics_y,
                plot_size,
                &context.classified_rules,
                rules,
                |cr| {
                    let fv = cr.metrics.feature_vector();
                    (fv[x_idx], fv[y_idx])
                },
                "Metrics",
            );

            // UMAP scatter: place next to metrics scatter.
            let has_umap = context
                .classified_rules
                .iter()
                .any(|cr| cr.umap_x.is_some() && cr.umap_y.is_some());
            if has_umap {
                let umap_x = metrics_x + plot_size + 4;
                render_mini_scatter(
                    &mut pixels,
                    img_width,
                    overlay_height,
                    umap_x,
                    metrics_y,
                    plot_size,
                    &context.classified_rules,
                    rules,
                    |cr| {
                        (
                            cr.umap_x.unwrap_or(0.0),
                            cr.umap_y.unwrap_or(0.0),
                        )
                    },
                    "UMAP",
                );
            }
        }
    }

    pixels
}

/// Render a mini scatter plot into the pixel buffer at the given position.
///
/// All classified rules are plotted as dim dots; the target rule is
/// highlighted with a bright crosshair marker.
fn render_mini_scatter(
    pixels: &mut [u8],
    img_width: u32,
    img_height: u32,
    origin_x: u32,
    origin_y: u32,
    plot_size: u32,
    classified: &[ClassifiedRule],
    target_rules: &Rules,
    coord_fn: impl Fn(&ClassifiedRule) -> (f64, f64),
    title: &str,
) {
    // Collect coordinates.
    let coords: Vec<(f64, f64, bool)> = classified
        .iter()
        .map(|cr| {
            let (x, y) = coord_fn(cr);
            let is_target = cr.rules == *target_rules;
            (x, y, is_target)
        })
        .collect();

    if coords.is_empty() {
        return;
    }

    // Compute value range for normalization.
    let (mut min_x, mut max_x, mut min_y, mut max_y) =
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for &(x, y, _) in &coords {
        if x.is_finite() && y.is_finite() {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }
    let range_x = (max_x - min_x).max(1e-9);
    let range_y = (max_y - min_y).max(1e-9);

    // Draw dark plot background.
    for py in origin_y..origin_y + plot_size {
        for px in origin_x..origin_x + plot_size {
            if px < img_width && py < img_height {
                let idx = ((py * img_width + px) * 4) as usize;
                pixels[idx] = 0x10;
                pixels[idx + 1] = 0x10;
                pixels[idx + 2] = 0x18;
                pixels[idx + 3] = 0xFF;
            }
        }
    }

    // Draw border.
    for px in origin_x..origin_x + plot_size {
        set_pixel(pixels, img_width, img_height, px, origin_y, 0x40, 0x40, 0x40);
        set_pixel(
            pixels,
            img_width,
            img_height,
            px,
            origin_y + plot_size - 1,
            0x40,
            0x40,
            0x40,
        );
    }
    for py in origin_y..origin_y + plot_size {
        set_pixel(pixels, img_width, img_height, origin_x, py, 0x40, 0x40, 0x40);
        set_pixel(
            pixels,
            img_width,
            img_height,
            origin_x + plot_size - 1,
            py,
            0x40,
            0x40,
            0x40,
        );
    }

    // Plot all points as dim dots first, then overlay the target.
    let mut target_px_pos = None;
    let margin = 2u32;
    let inner = plot_size.saturating_sub(margin * 2);

    for &(x, y, is_target) in &coords {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        let nx = ((x - min_x) / range_x * inner as f64) as u32 + margin;
        // Invert Y so higher values are at the top.
        let ny = (((max_y - y) / range_y) * inner as f64) as u32 + margin;
        let px = origin_x + nx.min(plot_size - 1);
        let py = origin_y + ny.min(plot_size - 1);

        if is_target {
            target_px_pos = Some((px, py));
        } else {
            set_pixel(pixels, img_width, img_height, px, py, 0x60, 0x60, 0x80);
        }
    }

    // Draw target with a bright crosshair.
    if let Some((tx, ty)) = target_px_pos {
        // Horizontal crosshair line.
        for dx in 0..5u32 {
            let px = (tx + dx).saturating_sub(2);
            set_pixel(pixels, img_width, img_height, px, ty, 0xFF, 0x40, 0x40);
        }
        // Vertical crosshair line.
        for dy in 0..5u32 {
            let py = (ty + dy).saturating_sub(2);
            set_pixel(pixels, img_width, img_height, tx, py, 0xFF, 0x40, 0x40);
        }
        // Bright center dot.
        set_pixel(pixels, img_width, img_height, tx, ty, 0xFF, 0xFF, 0x00);
    }

    // Render title below the plot.
    let title_y = origin_y + plot_size + 1;
    if title_y + 7 < img_height {
        render_text(pixels, img_width, origin_x, title_y, title, 0x88, 0x88, 0x88);
    }
}

/// Set a single pixel in the RGBA buffer (with bounds checking).
fn set_pixel(
    pixels: &mut [u8],
    img_width: u32,
    img_height: u32,
    x: u32,
    y: u32,
    r: u8,
    g: u8,
    b: u8,
) {
    if x < img_width && y < img_height {
        let idx = ((y * img_width + x) * 4) as usize;
        pixels[idx] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
        pixels[idx + 3] = 0xFF;
    }
}

// ── Minimal 5×7 bitmap font ─────────────────────────────────────────────────

/// Render a string using the built-in 5×7 bitmap font.
fn render_text(
    pixels: &mut [u8],
    img_width: u32,
    start_x: u32,
    start_y: u32,
    text: &str,
    r: u8,
    g: u8,
    b: u8,
) {
    let mut cursor_x = start_x;
    let img_height = pixels.len() as u32 / (img_width * 4);
    for ch in text.chars() {
        let glyph = char_glyph(ch);
        for (row, &bits) in glyph.iter().enumerate() {
            for col in 0..5u32 {
                if (bits >> (4 - col)) & 1 == 1 {
                    let px = cursor_x + col;
                    let py = start_y + row as u32;
                    if px < img_width && py < img_height {
                        let idx = ((py * img_width + px) * 4) as usize;
                        pixels[idx] = r;
                        pixels[idx + 1] = g;
                        pixels[idx + 2] = b;
                        pixels[idx + 3] = 0xFF;
                    }
                }
            }
        }
        cursor_x += 6; // 5 pixels wide + 1 pixel gap
    }
}

/// Return the 5×7 bitmap for a character. Each byte represents one row,
/// with the top 5 bits encoding pixel columns (MSB = leftmost).
fn char_glyph(ch: char) -> [u8; 7] {
    match ch {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '3' => [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
        'A' | 'a' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' | 'b' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' | 'c' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' | 'd' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' | 'e' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' | 'f' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' | 'g' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'H' | 'h' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' | 'i' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' | 'j' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        'K' | 'k' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' | 'l' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' | 'm' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' | 'n' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' | 'o' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' | 'p' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' | 'q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' | 'r' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' | 's' => [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110],
        'T' | 't' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' | 'u' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' | 'v' => [0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b01010, 0b00100],
        'W' | 'w' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' | 'x' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' | 'y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' | 'z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100],
        '%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        ',' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b01000],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '*' => [0b00000, 0b00100, 0b10101, 0b01110, 0b10101, 0b00100, 0b00000],
        '[' => [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
        ']' => [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
        '(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        ')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        ' ' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        _ => [0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111], // box
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_test_config(dir: PathBuf) -> ExportConfig {
        ExportConfig {
            grid_size: 16,
            generations: 10,
            output_dir: dir,
            frame_step: 2,
            cell_scale: 2,
            overlay_height: 60,
            plot_size: 30,
            ..ExportConfig::default()
        }
    }

    #[test]
    fn export_gif_creates_file() {
        let dir = std::env::temp_dir().join("catconway_test_export");
        let _ = std::fs::remove_dir_all(&dir);

        let config = default_test_config(dir.clone());
        let ctx = RuleContext::default();

        let result = export_gif(&Rules::conway(), &config, &ctx).unwrap();
        assert!(result.path.exists());
        assert_eq!(result.label, "B3/S23");
        assert!(result.total_frames > 0);
        // Verify it's a valid GIF (starts with GIF89a or GIF87a).
        let bytes = std::fs::read(&result.path).unwrap();
        assert!(bytes.starts_with(b"GIF"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_gif_filename_from_label() {
        let dir = std::env::temp_dir().join("catconway_test_export_name");
        let _ = std::fs::remove_dir_all(&dir);

        let config = ExportConfig {
            grid_size: 8,
            generations: 5,
            cell_scale: 2,
            output_dir: dir.clone(),
            ..ExportConfig::default()
        };
        let ctx = RuleContext::default();

        let result = export_gif(&Rules::highlife(), &config, &ctx).unwrap();
        assert!(result
            .path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .contains("B36_S23"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_multiple_rules() {
        let dir = std::env::temp_dir().join("catconway_test_export_multi");
        let _ = std::fs::remove_dir_all(&dir);

        let config = ExportConfig {
            grid_size: 8,
            generations: 5,
            cell_scale: 2,
            output_dir: dir.clone(),
            ..ExportConfig::default()
        };
        let ctx = RuleContext::default();

        let rules_list = vec![
            ("Conway".to_string(), Rules::conway()),
            ("Seeds".to_string(), Rules::seeds()),
        ];

        let results = export_multiple(&rules_list, &config, &ctx);
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(r.is_ok());
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_dead_rule_terminates_early() {
        let dir = std::env::temp_dir().join("catconway_test_export_dead");
        let _ = std::fs::remove_dir_all(&dir);

        let dead_rules = Rules {
            birth: 0,
            survival: 0,
            radius: 1,
        };

        let config = ExportConfig {
            grid_size: 8,
            generations: 100,
            cell_scale: 2,
            output_dir: dir.clone(),
            ..ExportConfig::default()
        };
        let ctx = RuleContext::default();

        let result = export_gif(&dead_rules, &config, &ctx).unwrap();
        assert!(result.total_frames <= 3);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_with_classification_context() {
        use crate::classify::{BehaviorClass, ClassifiedRule, RuleMetrics};

        let dir = std::env::temp_dir().join("catconway_test_export_ctx");
        let _ = std::fs::remove_dir_all(&dir);

        let conway = Rules::conway();
        let classified = vec![
            ClassifiedRule {
                rules: conway,
                label: "B3/S23".into(),
                metrics: RuleMetrics {
                    variation: 0.05,
                    mean_density: 0.03,
                    final_density: 0.03,
                    density_range: 0.01,
                    trend: -0.0001,
                    autocorrelation: 0.8,
                    entropy: 3.5,
                    dominant_period: 0,
                    monotonic_fraction: 0.4,
                    roughness: 0.001,
                },
                behavior: BehaviorClass::Complex,
                cluster: Some(0),
                umap_x: Some(1.0),
                umap_y: Some(2.0),
                umap_cluster: Some(0),
            },
            ClassifiedRule {
                rules: Rules::seeds(),
                label: "B2/S".into(),
                metrics: RuleMetrics {
                    variation: 0.9,
                    mean_density: 0.5,
                    final_density: 0.0,
                    density_range: 0.8,
                    trend: -0.01,
                    autocorrelation: 0.1,
                    entropy: 5.0,
                    dominant_period: 0,
                    monotonic_fraction: 0.2,
                    roughness: 0.1,
                },
                behavior: BehaviorClass::Chaotic,
                cluster: Some(1),
                umap_x: Some(-1.0),
                umap_y: Some(-2.0),
                umap_cluster: Some(1),
            },
        ];

        let ctx = RuleContext {
            classified_rules: classified,
            scatter_x: 0, // variation
            scatter_y: 1, // mean_density
        };

        let config = default_test_config(dir.clone());
        let result = export_gif(&conway, &config, &ctx).unwrap();
        assert!(result.path.exists());
        let bytes = std::fs::read(&result.path).unwrap();
        assert!(bytes.starts_with(b"GIF"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mini_scatter_does_not_panic_on_empty() {
        let img_width = 100u32;
        let img_height = 60u32;
        let mut pixels = vec![0u8; (img_width * img_height * 4) as usize];

        render_mini_scatter(
            &mut pixels,
            img_width,
            img_height,
            10,
            5,
            40,
            &[],
            &Rules::conway(),
            |_cr| (0.0, 0.0),
            "Test",
        );
        // Should not panic.
    }
}
