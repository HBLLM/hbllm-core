//! Frame-level change detection using perceptual hashing.
//!
//! Provides O(1) per-frame "has the scene changed?" checks to avoid
//! wasting expensive ONNX inference on static/near-static feeds.

use std::sync::Mutex;

use crate::preprocessing::{hamming_distance, perceptual_hash};
use image::DynamicImage;

/// Detects whether a new frame is visually different from the last-seen frame.
///
/// Uses 64-bit perceptual hashing with configurable Hamming distance threshold.
/// Thread-safe via internal mutex.
pub struct ChangeDetector {
    /// Maximum Hamming distance to consider "unchanged" (0–64).
    /// Default: 5 (≈ 92% similarity).
    threshold: u32,
    /// Last-seen perceptual hash.
    last_hash: Mutex<Option<u64>>,
}

impl ChangeDetector {
    /// Create a new detector with the given Hamming distance threshold.
    ///
    /// - `threshold = 0`: Only identical frames are considered unchanged.
    /// - `threshold = 5`: Default; tolerates minor JPEG artifacts and noise.
    /// - `threshold = 10`: Very lenient; only major scene changes trigger.
    pub fn new(threshold: u32) -> Self {
        Self {
            threshold,
            last_hash: Mutex::new(None),
        }
    }

    /// Check if the frame has changed from the last-seen frame.
    ///
    /// Returns `true` if the frame is new/different (inference should run),
    /// `false` if the frame is effectively identical (skip inference).
    ///
    /// Always returns `true` for the very first frame.
    pub fn is_changed(&self, img: &DynamicImage) -> bool {
        let new_hash = perceptual_hash(img);
        let mut last = self.last_hash.lock().unwrap();

        let changed = match *last {
            Some(prev_hash) => hamming_distance(prev_hash, new_hash) > self.threshold,
            None => true, // First frame is always "changed"
        };

        if changed {
            *last = Some(new_hash);
        }

        changed
    }

    /// Reset the detector state (next frame will always be "changed").
    pub fn reset(&self) {
        let mut last = self.last_hash.lock().unwrap();
        *last = None;
    }

    /// Get the current threshold.
    pub fn get_threshold(&self) -> u32 {
        self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};

    fn solid_image(r: u8, g: u8, b: u8) -> DynamicImage {
        let mut img = RgbImage::new(32, 32);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([r, g, b]);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_first_frame_always_changed() {
        let det = ChangeDetector::new(5);
        let img = solid_image(100, 100, 100);
        assert!(det.is_changed(&img));
    }

    #[test]
    fn test_identical_frames_unchanged() {
        let det = ChangeDetector::new(5);
        let img = solid_image(100, 100, 100);
        assert!(det.is_changed(&img)); // First frame
        assert!(!det.is_changed(&img)); // Same frame
        assert!(!det.is_changed(&img)); // Still same
    }

    #[test]
    fn test_different_frames_changed() {
        let det = ChangeDetector::new(5);
        // Use a solid image vs a checkerboard pattern — fundamentally different spatial structure
        let img1 = solid_image(100, 100, 100);
        // Build a checkerboard image
        let mut checker = RgbImage::new(32, 32);
        for (x, y, pixel) in checker.enumerate_pixels_mut() {
            let v = if (x / 4 + y / 4) % 2 == 0 { 255u8 } else { 0u8 };
            *pixel = image::Rgb([v, v, v]);
        }
        let img2 = DynamicImage::ImageRgb8(checker);
        assert!(det.is_changed(&img1)); // First frame
        assert!(det.is_changed(&img2)); // Different spatial pattern
    }

    #[test]
    fn test_reset() {
        let det = ChangeDetector::new(5);
        let img = solid_image(50, 50, 50);
        assert!(det.is_changed(&img));
        assert!(!det.is_changed(&img));
        det.reset();
        assert!(det.is_changed(&img)); // After reset, first frame is "changed"
    }

    #[test]
    fn test_zero_threshold_strict() {
        let det = ChangeDetector::new(0);
        let img = solid_image(100, 100, 100);
        assert!(det.is_changed(&img));
        assert!(!det.is_changed(&img)); // Exact same → unchanged
    }
}
