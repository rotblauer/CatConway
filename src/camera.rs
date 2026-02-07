/// Camera state for pan/zoom exploration of the grid.
///
/// The camera maps from screen UV coordinates (0..1) to world coordinates (0..1),
/// where the grid occupies (0,0) to (1,1) in world space.
#[derive(Debug, Clone)]
pub struct Camera {
    /// Center of the view in world-space (0..1 = grid extent)
    pub center_x: f32,
    pub center_y: f32,
    /// Zoom level: 1.0 = view entire grid, smaller = zoomed in
    pub scale: f32,
    /// Window aspect ratio (width / height)
    pub aspect: f32,
}

/// Uniform data sent to the GPU for camera-based rendering.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub offset_x: f32,
    pub offset_y: f32,
    pub scale: f32,
    pub aspect: f32,
    pub grid_width: f32,
    pub grid_height: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            center_x: 0.5,
            center_y: 0.5,
            scale: 1.0,
            aspect: 1.0,
        }
    }

    /// Zoom in/out by the given factor around the current center.
    pub fn zoom(&mut self, factor: f32) {
        self.scale *= factor;
        self.scale = self.scale.clamp(0.001, 10.0);
    }

    /// Pan the camera by the given delta in screen-space fractions.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.center_x += dx * self.scale * self.aspect;
        self.center_y += dy * self.scale;
    }

    /// Reset the camera to the default view (centered, showing entire grid).
    pub fn reset(&mut self) {
        self.center_x = 0.5;
        self.center_y = 0.5;
        self.scale = 1.0;
    }

    /// Build the GPU uniform for the current camera state.
    pub fn uniform(&self, grid_width: u32, grid_height: u32) -> CameraUniform {
        CameraUniform {
            offset_x: self.center_x,
            offset_y: self.center_y,
            scale: self.scale,
            aspect: self.aspect,
            grid_width: grid_width as f32,
            grid_height: grid_height as f32,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_default() {
        let cam = Camera::new();
        assert!((cam.center_x - 0.5).abs() < f32::EPSILON);
        assert!((cam.center_y - 0.5).abs() < f32::EPSILON);
        assert!((cam.scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_camera_zoom() {
        let mut cam = Camera::new();
        cam.zoom(0.5);
        assert!((cam.scale - 0.5).abs() < f32::EPSILON);
        cam.zoom(0.5);
        assert!((cam.scale - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_camera_zoom_clamp() {
        let mut cam = Camera::new();
        // Zoom way in
        for _ in 0..100 {
            cam.zoom(0.5);
        }
        assert!(cam.scale >= 0.001);

        // Zoom way out
        for _ in 0..100 {
            cam.zoom(2.0);
        }
        assert!(cam.scale <= 10.0);
    }

    #[test]
    fn test_camera_pan() {
        let mut cam = Camera::new();
        cam.aspect = 1.0;
        let initial_x = cam.center_x;
        cam.pan(0.1, 0.0);
        assert!(cam.center_x > initial_x);
    }

    #[test]
    fn test_camera_reset() {
        let mut cam = Camera::new();
        cam.zoom(0.1);
        cam.pan(0.5, 0.5);
        cam.reset();
        assert!((cam.center_x - 0.5).abs() < f32::EPSILON);
        assert!((cam.center_y - 0.5).abs() < f32::EPSILON);
        assert!((cam.scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_camera_uniform() {
        let cam = Camera::new();
        let u = cam.uniform(512, 512);
        assert!((u.grid_width - 512.0).abs() < f32::EPSILON);
        assert!((u.grid_height - 512.0).abs() < f32::EPSILON);
        assert!((u.offset_x - 0.5).abs() < f32::EPSILON);
    }
}
