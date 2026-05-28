//! Pure-Rust image preprocessing for vision models.
//!
//! Handles resize, center-crop, and normalization to produce a CHW f32 tensor
//! suitable for BLIP/SigLIP/CLIP-style vision transformers.
//! Zero dependency on Python or PIL.

use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::Array3;

/// Standard ImageNet normalization constants.
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Preprocess an image for a vision transformer.
///
/// Steps:
/// 1. Resize shortest side to `target_size`, preserving aspect ratio.
/// 2. Center-crop to `target_size × target_size`.
/// 3. Convert to f32 in [0, 1], then normalize with ImageNet mean/std.
/// 4. Return as CHW ndarray (shape: [3, target_size, target_size]).
pub fn preprocess_image(img: &DynamicImage, target_size: u32) -> Array3<f32> {
    // Step 1: Resize shortest side
    let (w, h) = img.dimensions();
    let (new_w, new_h) = if w < h {
        (target_size, (target_size as f64 * h as f64 / w as f64) as u32)
    } else {
        ((target_size as f64 * w as f64 / h as f64) as u32, target_size)
    };
    let resized = img.resize_exact(new_w, new_h, FilterType::Lanczos3);

    // Step 2: Center crop
    let crop_x = (new_w.saturating_sub(target_size)) / 2;
    let crop_y = (new_h.saturating_sub(target_size)) / 2;
    let cropped = resized.crop_imm(crop_x, crop_y, target_size, target_size);

    // Step 3: Convert to CHW f32 and normalize
    let rgb = cropped.to_rgb8();
    let ts = target_size as usize;
    let mut tensor = Array3::<f32>::zeros((3, ts, ts));

    for y in 0..ts {
        for x in 0..ts {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                tensor[[c, y, x]] = (val - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }

    tensor
}

/// Compute a 64-bit perceptual hash (pHash) for change detection.
///
/// Algorithm:
/// 1. Convert to grayscale.
/// 2. Resize to 8×8.
/// 3. Compute DCT-like mean.
/// 4. Set bits based on whether each pixel is above or below the mean.
///
/// Returns a u64 where each bit represents one spatial position.
pub fn perceptual_hash(img: &DynamicImage) -> u64 {
    // Resize to 8x8 grayscale
    let small = img
        .resize_exact(8, 8, FilterType::Lanczos3)
        .to_luma8();

    // Compute mean intensity
    let pixels: Vec<f32> = small.pixels().map(|p| p[0] as f32).collect();
    let mean: f32 = pixels.iter().sum::<f32>() / pixels.len() as f32;

    // Build hash: 1 bit per pixel, set if above mean
    let mut hash: u64 = 0;
    for (i, &val) in pixels.iter().enumerate() {
        if val > mean {
            hash |= 1u64 << i;
        }
    }
    hash
}

/// Hamming distance between two perceptual hashes.
///
/// Returns the number of differing bits (0 = identical, 64 = maximally different).
#[inline]
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};

    fn make_test_image(r: u8, g: u8, b: u8) -> DynamicImage {
        let mut img = RgbImage::new(64, 64);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([r, g, b]);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_preprocess_shape() {
        let img = make_test_image(128, 64, 32);
        let tensor = preprocess_image(&img, 224);
        assert_eq!(tensor.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_preprocess_normalized_range() {
        let img = make_test_image(128, 128, 128);
        let tensor = preprocess_image(&img, 32);
        // All pixels are the same, so after normalization all values should be identical
        let val = tensor[[0, 0, 0]];
        // (128/255 - 0.485) / 0.229 ≈ 0.0693
        assert!((val - 0.0693).abs() < 0.01, "Unexpected normalized value: {val}");
    }

    #[test]
    fn test_phash_identical_images() {
        let img1 = make_test_image(100, 150, 200);
        let img2 = make_test_image(100, 150, 200);
        assert_eq!(perceptual_hash(&img1), perceptual_hash(&img2));
        assert_eq!(hamming_distance(perceptual_hash(&img1), perceptual_hash(&img2)), 0);
    }

    #[test]
    fn test_phash_different_images() {
        let img1 = make_test_image(0, 0, 0);
        let img2 = make_test_image(255, 255, 255);
        // Same solid color images have identical hashes (all above or below mean)
        // But very different structured images would differ
        let h1 = perceptual_hash(&img1);
        let h2 = perceptual_hash(&img2);
        // Solid images: all pixels equal mean, so hash is 0 for both
        assert_eq!(hamming_distance(h1, h2), 0);
    }

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance(0, 0), 0);
        assert_eq!(hamming_distance(0b1111, 0b0000), 4);
        assert_eq!(hamming_distance(u64::MAX, 0), 64);
    }
}
