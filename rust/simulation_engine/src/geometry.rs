//! High-Performance Geometric Stability and 3D Bounding-Box Physics Kernels.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FastAABB {
    pub x_min: f64,
    pub y_min: f64,
    pub z_min: f64,
    pub x_max: f64,
    pub y_max: f64,
    pub z_max: f64,
}

impl FastAABB {
    pub fn new(x_min: f64, y_min: f64, z_min: f64, x_max: f64, y_max: f64, z_max: f64) -> Self {
        Self {
            x_min: x_min.min(x_max),
            y_min: y_min.min(y_max),
            z_min: z_min.min(z_max),
            x_max: x_min.max(x_max),
            y_max: y_min.max(y_max),
            z_max: z_min.max(z_max),
        }
    }

    #[inline]
    pub fn volume(&self) -> f64 {
        (self.x_max - self.x_min) * (self.y_max - self.y_min) * (self.z_max - self.z_min)
    }

    #[inline]
    pub fn center_of_mass(&self) -> (f64, f64, f64) {
        (
            (self.x_min + self.x_max) * 0.5,
            (self.y_min + self.y_max) * 0.5,
            (self.z_min + self.z_max) * 0.5,
        )
    }

    #[inline]
    pub fn intersects(&self, other: &FastAABB) -> bool {
        self.x_min <= other.x_max
            && self.x_max >= other.x_min
            && self.y_min <= other.y_max
            && self.y_max >= other.y_min
            && self.z_min <= other.z_max
            && self.z_max >= other.z_min
    }

    #[inline]
    pub fn intersection_volume(&self, other: &FastAABB) -> f64 {
        let dx = (self.x_max.min(other.x_max) - self.x_min.max(other.x_min)).max(0.0);
        let dy = (self.y_max.min(other.y_max) - self.y_min.max(other.y_min)).max(0.0);
        let dz = (self.z_max.min(other.z_max) - self.z_min.max(other.z_min)).max(0.0);
        dx * dy * dz
    }
}

/// Evaluates whether an upper object stably rests upon a base object.
///
/// Invariants:
/// 1. Upper base elevation matches base top within tolerance (z_upper_min >= z_base_max - tol).
/// 2. Center-of-mass of upper object projects inside base XY footprint + safety margin.
pub fn evaluate_support_stability(
    upper: &FastAABB,
    base: &FastAABB,
    tolerance: f64,
) -> (bool, f64) {
    let (cm_x, cm_y, _cm_z) = upper.center_of_mass();

    // Check vertical contact
    let z_contact = (upper.z_min - base.z_max).abs() <= tolerance;
    if !z_contact && upper.z_min < base.z_max - tolerance {
        // Interpenetrating
        return (false, -1.0);
    }

    // Distance from CM to closest base edge
    let dist_x = (cm_x - base.x_min).min(base.x_max - cm_x);
    let dist_y = (cm_y - base.y_min).min(base.y_max - cm_y);
    let min_margin = dist_x.min(dist_y);

    let is_stable = z_contact && min_margin >= 0.0;
    (is_stable, min_margin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_intersection_and_volume() {
        let b1 = FastAABB::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
        let b2 = FastAABB::new(1.0, 1.0, 1.0, 3.0, 3.0, 3.0);

        assert_eq!(b1.volume(), 8.0);
        assert!(b1.intersects(&b2));
        assert_eq!(b1.intersection_volume(&b2), 1.0);
    }

    #[test]
    fn test_stability_evaluation() {
        let table = FastAABB::new(0.0, 0.0, 0.0, 10.0, 10.0, 1.0);
        let stable_cup = FastAABB::new(4.0, 4.0, 1.0, 6.0, 6.0, 3.0);
        let tippy_cup = FastAABB::new(9.5, 9.5, 1.0, 11.5, 11.5, 3.0);

        let (stable, margin) = evaluate_support_stability(&stable_cup, &table, 0.01);
        assert!(stable);
        assert!(margin > 3.0);

        let (unstable, _) = evaluate_support_stability(&tippy_cup, &table, 0.01);
        assert!(!unstable);
    }
}
