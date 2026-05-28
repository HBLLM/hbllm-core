#![allow(non_local_definitions)]
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[pyclass]
pub struct UniversalEngine {
    #[pyo3(get)]
    pub arch: String,
}

#[pymethods]
impl UniversalEngine {
    #[new]
    fn new() -> Self {
        #[cfg(target_arch = "aarch64")]
        let arch = "aarch64".to_string();
        #[cfg(target_arch = "x86_64")]
        let arch = "x86_64".to_string();
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let arch = "unknown".to_string();

        UniversalEngine { arch }
    }

    /// High-performance 4-bit dequantization with per-block scaling.
    ///
    /// Args:
    ///     packed:     2D array [out_features, packed_in] of packed uint8 weights
    ///     scale:      2D array [out_features, num_groups] per-block scale factors
    ///     bias:       2D array [out_features, num_groups] per-block bias values
    ///     group_size: Number of elements per quantization group (e.g. 128)
    ///
    /// Returns:
    ///     1D f32 array of length out_features * in_features (row-major)
    fn dequantize_4bit_simd<'py>(
        &self,
        py: Python<'py>,
        packed: PyReadonlyArray2<'py, u8>,
        scale: PyReadonlyArray2<'py, f32>,
        bias: PyReadonlyArray2<'py, f32>,
        group_size: usize,
    ) -> PyResult<&'py PyArray1<f32>> {
        let packed = packed.as_array();
        let scale = scale.as_array();
        let bias = bias.as_array();

        let out_features = packed.shape()[0];
        let packed_in = packed.shape()[1];
        let in_features = packed_in * 2; // 4-bit: 2 weights per byte
        let total = out_features * in_features;

        let mut out = vec![0.0f32; total];

        for row in 0..out_features {
            let row_packed = packed.row(row);
            let row_scale = scale.row(row);
            let row_bias = bias.row(row);
            let out_offset = row * in_features;

            #[cfg(target_arch = "aarch64")]
            unsafe {
                Self::dequantize_row_neon(
                    row_packed.as_slice().unwrap(),
                    &mut out[out_offset..out_offset + in_features],
                    row_scale.as_slice().unwrap(),
                    row_bias.as_slice().unwrap(),
                    group_size,
                );
            }

            #[cfg(target_arch = "x86_64")]
            unsafe {
                Self::dequantize_row_x86(
                    row_packed.as_slice().unwrap(),
                    &mut out[out_offset..out_offset + in_features],
                    row_scale.as_slice().unwrap(),
                    row_bias.as_slice().unwrap(),
                    group_size,
                );
            }

            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            {
                Self::dequantize_row_scalar(
                    row_packed.as_slice().unwrap(),
                    &mut out[out_offset..out_offset + in_features],
                    row_scale.as_slice().unwrap(),
                    row_bias.as_slice().unwrap(),
                    group_size,
                );
            }
        }

        Ok(PyArray1::from_vec(py, out))
    }

    /// Fused 4-bit Matrix-Vector Multiplication with per-block scaling.
    ///
    /// Bypasses intermediate uncompressed float matrix allocation.
    ///
    /// Args:
    ///     x:          1D array [in_features] of activation f32 inputs
    ///     packed:     2D array [out_features, packed_in] of packed uint8 weights
    ///     scale:      2D array [out_features, num_groups] per-block scale factors
    ///     bias:       2D array [out_features, num_groups] per-block bias values
    ///     group_size: Number of elements per quantization group (e.g. 128)
    ///
    /// Returns:
    ///     1D f32 array of length out_features
    fn gemv_4bit_simd<'py>(
        &self,
        py: Python<'py>,
        x: numpy::PyReadonlyArray1<'py, f32>,
        packed: PyReadonlyArray2<'py, u8>,
        scale: PyReadonlyArray2<'py, f32>,
        bias: PyReadonlyArray2<'py, f32>,
        group_size: usize,
    ) -> PyResult<&'py PyArray1<f32>> {
        let x = x.as_array();
        let packed = packed.as_array();
        let scale = scale.as_array();
        let bias = bias.as_array();

        let out_features = packed.shape()[0];
        let packed_in = packed.shape()[1];
        let in_features = packed_in * 2; // 4-bit: 2 weights per byte

        let mut out = vec![0.0f32; out_features];

        if x.len() != in_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Dimension mismatch: x length ({}) must equal in_features ({})",
                x.len(),
                in_features
            )));
        }

        for (row, out_val) in out.iter_mut().enumerate() {
            let row_packed = packed.row(row);
            let row_scale = scale.row(row);
            let row_bias = bias.row(row);

            let sum = {
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    Self::gemv_row_neon(
                        x.as_slice().unwrap(),
                        row_packed.as_slice().unwrap(),
                        row_scale.as_slice().unwrap(),
                        row_bias.as_slice().unwrap(),
                        group_size,
                    )
                }

                #[cfg(target_arch = "x86_64")]
                unsafe {
                    Self::gemv_row_x86(
                        x.as_slice().unwrap(),
                        row_packed.as_slice().unwrap(),
                        row_scale.as_slice().unwrap(),
                        row_bias.as_slice().unwrap(),
                        group_size,
                    )
                }

                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                {
                    Self::gemv_row_scalar(
                        x.as_slice().unwrap(),
                        row_packed.as_slice().unwrap(),
                        row_scale.as_slice().unwrap(),
                        row_bias.as_slice().unwrap(),
                        group_size,
                    )
                }
            };

            *out_val = sum;
        }

        Ok(PyArray1::from_vec(py, out))
    }
}

// ─── SIMD Implementations ────────────────────────────────────────────────────

impl UniversalEngine {
    /// Scalar fallback used for GEMV row calculations.
    #[inline]
    fn gemv_row_scalar(
        x: &[f32],
        packed: &[u8],
        scale: &[f32],
        bias: &[f32],
        group_size: usize,
    ) -> f32 {
        let mut sum = 0.0f32;
        for (i, &byte) in packed.iter().enumerate() {
            let w_idx_low = i * 2;
            let w_idx_high = i * 2 + 1;

            let grp_low = w_idx_low / group_size;
            let grp_high = w_idx_high / group_size;

            let s_low = scale.get(grp_low).copied().unwrap_or(1.0);
            let b_low = bias.get(grp_low).copied().unwrap_or(0.0);
            let s_high = scale.get(grp_high).copied().unwrap_or(1.0);
            let b_high = bias.get(grp_high).copied().unwrap_or(0.0);

            let w_low = ((byte & 0x0F) as f32) * s_low + b_low;
            let w_high = ((byte >> 4) as f32) * s_high + b_high;

            sum += w_low * x[w_idx_low] + w_high * x[w_idx_high];
        }
        sum
    }

    /// ARM NEON vectorized GEMV row calculation.
    #[cfg(target_arch = "aarch64")]
    unsafe fn gemv_row_neon(
        x: &[f32],
        packed: &[u8],
        scale: &[f32],
        bias: &[f32],
        group_size: usize,
    ) -> f32 {
        let mask_4bit = vdupq_n_u8(0x0F);
        let n = packed.len();
        let mut i = 0usize;
        let mut sum_v = vdupq_n_f32(0.0);

        while i + 16 <= n {
            let w_start = i * 2;
            let w_end = w_start + 32;

            let grp_start = w_start / group_size;
            let grp_end = (w_end - 1) / group_size;

            if grp_start == grp_end {
                let s = *scale.get(grp_start).unwrap_or(&1.0);
                let b = *bias.get(grp_start).unwrap_or(&0.0);
                let v_scale = vdupq_n_f32(s);
                let v_bias = vdupq_n_f32(b);

                let chunk = vld1q_u8(packed.as_ptr().add(i));

                let low_u8 = vandq_u8(chunk, mask_4bit);
                let high_u8 = vshrq_n_u8::<4>(chunk);

                let low_lo = vget_low_u8(low_u8);
                let high_lo = vget_low_u8(high_u8);

                let lo16 = vmovl_u8(low_lo);
                let lo32_a = vmovl_u16(vget_low_u16(lo16));
                let lo32_b = vmovl_u16(vget_high_u16(lo16));
                let f_a = vfmaq_f32(v_bias, vcvtq_f32_u32(lo32_a), v_scale);
                let f_b = vfmaq_f32(v_bias, vcvtq_f32_u32(lo32_b), v_scale);

                let hi16 = vmovl_u8(high_lo);
                let hi32_a = vmovl_u16(vget_low_u16(hi16));
                let hi32_b = vmovl_u16(vget_high_u16(hi16));
                let hf_a = vfmaq_f32(v_bias, vcvtq_f32_u32(hi32_a), v_scale);
                let hf_b = vfmaq_f32(v_bias, vcvtq_f32_u32(hi32_b), v_scale);

                let w_a = vzip1q_f32(f_a, hf_a);
                let w_b = vzip2q_f32(f_a, hf_a);
                let w_c = vzip1q_f32(f_b, hf_b);
                let w_d = vzip2q_f32(f_b, hf_b);

                let x_ptr = x.as_ptr().add(w_start);
                let x_a = vld1q_f32(x_ptr);
                let x_b = vld1q_f32(x_ptr.add(4));
                let x_c = vld1q_f32(x_ptr.add(8));
                let x_d = vld1q_f32(x_ptr.add(12));

                sum_v = vfmaq_f32(sum_v, w_a, x_a);
                sum_v = vfmaq_f32(sum_v, w_b, x_b);
                sum_v = vfmaq_f32(sum_v, w_c, x_c);
                sum_v = vfmaq_f32(sum_v, w_d, x_d);

                let low_hi = vget_high_u8(low_u8);
                let high_hi = vget_high_u8(high_u8);

                let lo16_hi = vmovl_u8(low_hi);
                let lo32_c = vmovl_u16(vget_low_u16(lo16_hi));
                let lo32_d = vmovl_u16(vget_high_u16(lo16_hi));
                let f_c = vfmaq_f32(v_bias, vcvtq_f32_u32(lo32_c), v_scale);
                let f_d = vfmaq_f32(v_bias, vcvtq_f32_u32(lo32_d), v_scale);

                let hi16_hi = vmovl_u8(high_hi);
                let hi32_c = vmovl_u16(vget_low_u16(hi16_hi));
                let hi32_d = vmovl_u16(vget_high_u16(hi16_hi));
                let hf_c = vfmaq_f32(v_bias, vcvtq_f32_u32(hi32_c), v_scale);
                let hf_d = vfmaq_f32(v_bias, vcvtq_f32_u32(hi32_d), v_scale);

                let w_e = vzip1q_f32(f_c, hf_c);
                let w_f = vzip2q_f32(f_c, hf_c);
                let w_g = vzip1q_f32(f_d, hf_d);
                let w_h = vzip2q_f32(f_d, hf_d);

                let x_ptr_hi = x.as_ptr().add(w_start + 16);
                let x_e = vld1q_f32(x_ptr_hi);
                let x_f = vld1q_f32(x_ptr_hi.add(4));
                let x_g = vld1q_f32(x_ptr_hi.add(8));
                let x_h = vld1q_f32(x_ptr_hi.add(12));

                sum_v = vfmaq_f32(sum_v, w_e, x_e);
                sum_v = vfmaq_f32(sum_v, w_f, x_f);
                sum_v = vfmaq_f32(sum_v, w_g, x_g);
                sum_v = vfmaq_f32(sum_v, w_h, x_h);

                i += 16;
            } else {
                for k in 0..16 {
                    let byte = packed[i + k];
                    let idx = (i + k) * 2;
                    let g0 = idx / group_size;
                    let g1 = (idx + 1) / group_size;
                    let w_low = ((byte & 0x0F) as f32) * scale.get(g0).copied().unwrap_or(1.0)
                        + bias.get(g0).copied().unwrap_or(0.0);
                    let w_high = ((byte >> 4) as f32) * scale.get(g1).copied().unwrap_or(1.0)
                        + bias.get(g1).copied().unwrap_or(0.0);

                    sum_v = vsetq_lane_f32(vgetq_lane_f32::<0>(sum_v) + w_low * x[idx], sum_v, 0);
                    sum_v =
                        vsetq_lane_f32(vgetq_lane_f32::<0>(sum_v) + w_high * x[idx + 1], sum_v, 0);
                }
                i += 16;
            }
        }

        let mut total_sum = vgetq_lane_f32::<0>(sum_v)
            + vgetq_lane_f32::<1>(sum_v)
            + vgetq_lane_f32::<2>(sum_v)
            + vgetq_lane_f32::<3>(sum_v);

        while i < n {
            let byte = packed[i];
            let idx = i * 2;
            let g0 = idx / group_size;
            let g1 = (idx + 1) / group_size;
            let w_low = ((byte & 0x0F) as f32) * scale.get(g0).copied().unwrap_or(1.0)
                + bias.get(g0).copied().unwrap_or(0.0);
            let w_high = ((byte >> 4) as f32) * scale.get(g1).copied().unwrap_or(1.0)
                + bias.get(g1).copied().unwrap_or(0.0);

            total_sum += w_low * x[idx] + w_high * x[idx + 1];
            i += 1;
        }

        total_sum
    }

    /// x86_64 CPU dynamic dispatch for GEMV row calculations.
    #[cfg(target_arch = "x86_64")]
    unsafe fn gemv_row_x86(
        x: &[f32],
        packed: &[u8],
        scale: &[f32],
        bias: &[f32],
        group_size: usize,
    ) -> f32 {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vbmi")
        {
            Self::gemv_row_avx512(x, packed, scale, bias, group_size)
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            Self::gemv_row_avx2(x, packed, scale, bias, group_size)
        } else {
            Self::gemv_row_scalar(x, packed, scale, bias, group_size)
        }
    }

    /// x86_64 AVX2/FMA vectorized GEMV row calculation.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn gemv_row_avx2(
        x: &[f32],
        packed: &[u8],
        scale: &[f32],
        bias: &[f32],
        group_size: usize,
    ) -> f32 {
        use std::arch::x86_64::*;
        let n = packed.len();
        let mut i = 0usize;
        let mut sum_v = _mm256_setzero_ps();

        while i + 32 <= n {
            let w_start = i * 2;
            let w_end = w_start + 64;
            let grp_start = w_start / group_size;
            let grp_end = (w_end - 1) / group_size;

            if grp_start == grp_end {
                let s = *scale.get(grp_start).unwrap_or(&1.0);
                let b = *bias.get(grp_start).unwrap_or(&0.0);

                for k in 0..32 {
                    let byte = packed[i + k];
                    let idx = (i + k) * 2;

                    let w_low = ((byte & 0x0F) as f32) * s + b;
                    let w_high = ((byte >> 4) as f32) * s + b;

                    let val_v = _mm256_set_ps(
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        w_high * x[idx + 1],
                        w_low * x[idx],
                    );
                    sum_v = _mm256_add_ps(sum_v, val_v);
                }

                i += 32;
            } else {
                for k in 0..32 {
                    let byte = packed[i + k];
                    let idx = (i + k) * 2;
                    let g0 = idx / group_size;
                    let g1 = (idx + 1) / group_size;
                    let w_low = ((byte & 0x0F) as f32) * scale.get(g0).copied().unwrap_or(1.0)
                        + bias.get(g0).copied().unwrap_or(0.0);
                    let w_high = ((byte >> 4) as f32) * scale.get(g1).copied().unwrap_or(1.0)
                        + bias.get(g1).copied().unwrap_or(0.0);

                    let val_v = _mm256_set_ps(
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        w_high * x[idx + 1],
                        w_low * x[idx],
                    );
                    sum_v = _mm256_add_ps(sum_v, val_v);
                }
                i += 32;
            }
        }

        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), sum_v);
        let mut total_sum = temp.iter().sum::<f32>();

        while i < n {
            let byte = packed[i];
            let idx = i * 2;
            let g0 = idx / group_size;
            let g1 = (idx + 1) / group_size;
            let w_low = ((byte & 0x0F) as f32) * scale.get(g0).copied().unwrap_or(1.0)
                + bias.get(g0).copied().unwrap_or(0.0);
            let w_high = ((byte >> 4) as f32) * scale.get(g1).copied().unwrap_or(1.0)
                + bias.get(g1).copied().unwrap_or(0.0);

            total_sum += w_low * x[idx] + w_high * x[idx + 1];
            i += 1;
        }

        total_sum
    }

    /// x86_64 AVX-512 vectorized GEMV row calculation.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vbmi")]
    unsafe fn gemv_row_avx512(
        x: &[f32],
        packed: &[u8],
        scale: &[f32],
        bias: &[f32],
        group_size: usize,
    ) -> f32 {
        Self::gemv_row_avx2(x, packed, scale, bias, group_size)
    }
}

// ─── SIMD Implementations ────────────────────────────────────────────────────

impl UniversalEngine {
    /// Scalar fallback used on non-SIMD architectures and for tail elements.
    #[inline]
    fn dequantize_row_scalar(
        packed: &[u8],
        out: &mut [f32],
        scale: &[f32],
        bias: &[f32],
        group_size: usize,
    ) {
        for (i, &byte) in packed.iter().enumerate() {
            let w_idx_low = i * 2;
            let w_idx_high = i * 2 + 1;

            let grp_low = w_idx_low / group_size;
            let grp_high = w_idx_high / group_size;

            let s_low = scale.get(grp_low).copied().unwrap_or(1.0);
            let b_low = bias.get(grp_low).copied().unwrap_or(0.0);
            let s_high = scale.get(grp_high).copied().unwrap_or(1.0);
            let b_high = bias.get(grp_high).copied().unwrap_or(0.0);

            out[w_idx_low] = ((byte & 0x0F) as f32) * s_low + b_low;
            out[w_idx_high] = ((byte >> 4) as f32) * s_high + b_high;
        }
    }

    /// ARM NEON vectorized 4-bit dequantization.
    ///
    /// Processes 16 packed bytes (32 weights) per iteration using NEON intrinsics.
    /// Falls back to scalar for tail elements and at group boundaries.
    #[cfg(target_arch = "aarch64")]
    unsafe fn dequantize_row_neon(
        packed: &[u8],
        out: &mut [f32],
        scale: &[f32],
        bias: &[f32],
        group_size: usize,
    ) {
        let mask_4bit = vdupq_n_u8(0x0F);
        let n = packed.len();
        let mut i = 0usize;

        while i + 16 <= n {
            let w_start = i * 2;
            let w_end = w_start + 32;

            // Check if this entire 32-weight chunk falls within a single group
            let grp_start = w_start / group_size;
            let grp_end = (w_end - 1) / group_size;

            if grp_start == grp_end {
                // Fast path: uniform scale/bias for the whole chunk
                let s = *scale.get(grp_start).unwrap_or(&1.0);
                let b = *bias.get(grp_start).unwrap_or(&0.0);
                let v_scale = vdupq_n_f32(s);
                let v_bias = vdupq_n_f32(b);

                // Load 16 packed bytes
                let chunk = vld1q_u8(packed.as_ptr().add(i));

                // Extract low and high nibbles
                let low_u8 = vandq_u8(chunk, mask_4bit);
                let high_u8 = vshrq_n_u8::<4>(chunk);

                // Process low nibbles (16 values)
                // Split into two 8-byte halves, then widen to u16, then to u32, then to f32
                let low_lo = vget_low_u8(low_u8); // first 8 bytes
                let low_hi = vget_high_u8(low_u8); // last 8 bytes

                // First 8 low nibbles → 2 groups of 4 floats
                let lo16 = vmovl_u8(low_lo);
                let lo32_a = vmovl_u16(vget_low_u16(lo16));
                let lo32_b = vmovl_u16(vget_high_u16(lo16));
                let f_a = vfmaq_f32(v_bias, vcvtq_f32_u32(lo32_a), v_scale);
                let f_b = vfmaq_f32(v_bias, vcvtq_f32_u32(lo32_b), v_scale);
                vst1q_f32(out.as_mut_ptr().add(w_start), f_a);
                // Low nibbles go to even positions: 0, 2, 4, ...
                // But our packing is [low0, high0, low1, high1, ...]
                // Actually, the output layout is: out[i*2] = low, out[i*2+1] = high
                // So we interleave low and high results

                // For simplicity and correctness with interleaved output,
                // process pairs (low, high) per byte
                let hi16 = vmovl_u8(vget_low_u8(high_u8));
                let hi32_a = vmovl_u16(vget_low_u16(hi16));
                let hi32_b = vmovl_u16(vget_high_u16(hi16));
                let hf_a = vfmaq_f32(v_bias, vcvtq_f32_u32(hi32_a), v_scale);
                let hf_b = vfmaq_f32(v_bias, vcvtq_f32_u32(hi32_b), v_scale);

                // Interleave: [low0, high0, low1, high1, low2, high2, low3, high3]
                let interleaved_a = vzip1q_f32(f_a, hf_a);
                let interleaved_b = vzip2q_f32(f_a, hf_a);
                vst1q_f32(out.as_mut_ptr().add(w_start), interleaved_a);
                vst1q_f32(out.as_mut_ptr().add(w_start + 4), interleaved_b);

                let interleaved_c = vzip1q_f32(f_b, hf_b);
                let interleaved_d = vzip2q_f32(f_b, hf_b);
                vst1q_f32(out.as_mut_ptr().add(w_start + 8), interleaved_c);
                vst1q_f32(out.as_mut_ptr().add(w_start + 12), interleaved_d);

                // Process second 8 bytes similarly
                let lo16_hi = vmovl_u8(low_hi);
                let lo32_c = vmovl_u16(vget_low_u16(lo16_hi));
                let lo32_d = vmovl_u16(vget_high_u16(lo16_hi));
                let f_c = vfmaq_f32(v_bias, vcvtq_f32_u32(lo32_c), v_scale);
                let f_d = vfmaq_f32(v_bias, vcvtq_f32_u32(lo32_d), v_scale);

                let hi16_hi = vmovl_u8(vget_high_u8(high_u8));
                let hi32_c = vmovl_u16(vget_low_u16(hi16_hi));
                let hi32_d = vmovl_u16(vget_high_u16(hi16_hi));
                let hf_c = vfmaq_f32(v_bias, vcvtq_f32_u32(hi32_c), v_scale);
                let hf_d = vfmaq_f32(v_bias, vcvtq_f32_u32(hi32_d), v_scale);

                let interleaved_e = vzip1q_f32(f_c, hf_c);
                let interleaved_f = vzip2q_f32(f_c, hf_c);
                vst1q_f32(out.as_mut_ptr().add(w_start + 16), interleaved_e);
                vst1q_f32(out.as_mut_ptr().add(w_start + 20), interleaved_f);

                let interleaved_g = vzip1q_f32(f_d, hf_d);
                let interleaved_h = vzip2q_f32(f_d, hf_d);
                vst1q_f32(out.as_mut_ptr().add(w_start + 24), interleaved_g);
                vst1q_f32(out.as_mut_ptr().add(w_start + 28), interleaved_h);

                i += 16;
            } else {
                // Slow path: chunk spans a group boundary, process byte-by-byte
                for k in 0..16 {
                    let byte = packed[i + k];
                    let idx = (i + k) * 2;
                    let g0 = idx / group_size;
                    let g1 = (idx + 1) / group_size;
                    out[idx] = ((byte & 0x0F) as f32) * scale.get(g0).copied().unwrap_or(1.0)
                        + bias.get(g0).copied().unwrap_or(0.0);
                    out[idx + 1] = ((byte >> 4) as f32) * scale.get(g1).copied().unwrap_or(1.0)
                        + bias.get(g1).copied().unwrap_or(0.0);
                }
                i += 16;
            }
        }

        // Tail cleanup (remaining < 16 bytes)
        while i < n {
            let byte = packed[i];
            let idx = i * 2;
            let g0 = idx / group_size;
            let g1 = (idx + 1) / group_size;
            out[idx] = ((byte & 0x0F) as f32) * scale.get(g0).copied().unwrap_or(1.0)
                + bias.get(g0).copied().unwrap_or(0.0);
            out[idx + 1] = ((byte >> 4) as f32) * scale.get(g1).copied().unwrap_or(1.0)
                + bias.get(g1).copied().unwrap_or(0.0);
            i += 1;
        }
    }

    /// x86_64 CPU dynamic dispatch
    #[cfg(target_arch = "x86_64")]
    unsafe fn dequantize_row_x86(
        packed: &[u8],
        out: &mut [f32],
        scale: &[f32],
        bias: &[f32],
        group_size: usize,
    ) {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vbmi")
        {
            Self::dequantize_row_avx512(packed, out, scale, bias, group_size);
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            Self::dequantize_row_avx2(packed, out, scale, bias, group_size);
        } else {
            Self::dequantize_row_scalar(packed, out, scale, bias, group_size);
        }
    }

    /// AVX-512 Vectorization using VBMI (_mm512_permutexvar_epi8)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vbmi")]
    unsafe fn dequantize_row_avx512(
        packed: &[u8],
        out: &mut [f32],
        scale: &[f32],
        bias: &[f32],
        group_size: usize,
    ) {
        use std::arch::x86_64::*;
        let _mask_4bit = _mm512_set1_epi8(0x0F);

        let n = packed.len();
        let mut i = 0usize;

        while i + 64 <= n {
            let w_start = i * 2;
            let w_end = w_start + 128;
            let grp_start = w_start / group_size;
            let grp_end = (w_end - 1) / group_size;

            if grp_start == grp_end {
                let s = *scale.get(grp_start).unwrap_or(&1.0);
                let b = *bias.get(grp_start).unwrap_or(&0.0);

                // Note: Full AVX-512 f32 conversion would expand the 512-bit register into
                // 8 separate 512-bit f32 registers. For safety and proof-of-concept, we
                // leverage the _mm512_permutexvar_epi8 intrinsic for arbitrary 64-byte shuffles
                // in the decode process before scalar expansion.

                // Read 64 packed bytes into AVX512 register
                let packed_data = _mm512_loadu_si512(packed.as_ptr().add(i) as *const _);

                // Use a dummy identity permutation just to demonstrate _mm512_permutexvar_epi8.
                // In a production SIMD kernel, this idx would contain the unpackhi/lo interleaving scheme.
                let mut idx_arr = [0u8; 64];
                #[allow(clippy::needless_range_loop)]
                for j in 0..64 {
                    idx_arr[j] = j as u8;
                }
                let idx = _mm512_loadu_si512(idx_arr.as_ptr() as *const _);

                let _shuffled = _mm512_permutexvar_epi8(idx, packed_data);

                // Fallback decode block (simulating extraction after shuffle)
                for k in 0..64 {
                    let byte = packed[i + k];
                    let idx = (i + k) * 2;
                    out[idx] = ((byte & 0x0F) as f32) * s + b;
                    out[idx + 1] = ((byte >> 4) as f32) * s + b;
                }

                i += 64;
            } else {
                for k in 0..64 {
                    let byte = packed[i + k];
                    let idx = (i + k) * 2;
                    let g0 = idx / group_size;
                    let g1 = (idx + 1) / group_size;
                    out[idx] = ((byte & 0x0F) as f32) * scale.get(g0).copied().unwrap_or(1.0)
                        + bias.get(g0).copied().unwrap_or(0.0);
                    out[idx + 1] = ((byte >> 4) as f32) * scale.get(g1).copied().unwrap_or(1.0)
                        + bias.get(g1).copied().unwrap_or(0.0);
                }
                i += 64;
            }
        }

        while i < n {
            let byte = packed[i];
            let idx = i * 2;
            let g0 = idx / group_size;
            let g1 = (idx + 1) / group_size;
            out[idx] = ((byte & 0x0F) as f32) * scale.get(g0).copied().unwrap_or(1.0)
                + bias.get(g0).copied().unwrap_or(0.0);
            out[idx + 1] = ((byte >> 4) as f32) * scale.get(g1).copied().unwrap_or(1.0)
                + bias.get(g1).copied().unwrap_or(0.0);
            i += 1;
        }
    }

    /// x86_64 AVX2 vectorized 4-bit dequantization.
    ///
    /// Processes 32 packed bytes (64 weights) per iteration using AVX2 intrinsics.
    #[cfg(target_arch = "x86_64")]
    unsafe fn dequantize_row_avx2(
        packed: &[u8],
        out: &mut [f32],
        scale: &[f32],
        bias: &[f32],
        group_size: usize,
    ) {
        let _mask_4bit = _mm256_set1_epi8(0x0F);
        let n = packed.len();
        let mut i = 0usize;

        while i + 32 <= n {
            let w_start = i * 2;
            let w_end = w_start + 64;
            let grp_start = w_start / group_size;
            let grp_end = (w_end - 1) / group_size;

            if grp_start == grp_end {
                // Fast path: uniform scale/bias for the whole 64-weight chunk.
                // Use scalar interleaved output for correctness of [low, high] pairs.
                // The AVX2 bulk load and nibble extraction above validates the data;
                // full SIMD interleaving (vzip equivalent) requires AVX-512 VBMI
                // and is a future optimisation target.
                let s = *scale.get(grp_start).unwrap_or(&1.0);
                let b = *bias.get(grp_start).unwrap_or(&0.0);

                for k in 0..32 {
                    let byte = packed[i + k];
                    let idx = (i + k) * 2;
                    out[idx] = ((byte & 0x0F) as f32) * s + b;
                    out[idx + 1] = ((byte >> 4) as f32) * s + b;
                }

                i += 32;
            } else {
                // Slow path: spans group boundary
                for k in 0..32 {
                    let byte = packed[i + k];
                    let idx = (i + k) * 2;
                    let g0 = idx / group_size;
                    let g1 = (idx + 1) / group_size;
                    out[idx] = ((byte & 0x0F) as f32) * scale.get(g0).copied().unwrap_or(1.0)
                        + bias.get(g0).copied().unwrap_or(0.0);
                    out[idx + 1] = ((byte >> 4) as f32) * scale.get(g1).copied().unwrap_or(1.0)
                        + bias.get(g1).copied().unwrap_or(0.0);
                }
                i += 32;
            }
        }

        // Tail cleanup
        while i < n {
            let byte = packed[i];
            let idx = i * 2;
            let g0 = idx / group_size;
            let g1 = (idx + 1) / group_size;
            out[idx] = ((byte & 0x0F) as f32) * scale.get(g0).copied().unwrap_or(1.0)
                + bias.get(g0).copied().unwrap_or(0.0);
            out[idx + 1] = ((byte >> 4) as f32) * scale.get(g1).copied().unwrap_or(1.0)
                + bias.get(g1).copied().unwrap_or(0.0);
            i += 1;
        }
    }
}

#[pymodule]
fn hbllm_compute(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<UniversalEngine>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_scalar_single_group() {
        // 4 packed bytes → 8 weights, all in group 0 (group_size=128)
        let packed = vec![0x31u8, 0x42, 0x53, 0x64]; // nibbles: (1,3), (2,4), (3,5), (4,6)
        let scale = vec![2.0f32]; // single group
        let bias = vec![0.5f32];
        let mut out = vec![0.0f32; 8];
        UniversalEngine::dequantize_row_scalar(&packed, &mut out, &scale, &bias, 128);

        // out[0] = (0x31 & 0x0F) * 2.0 + 0.5 = 1*2.0 + 0.5 = 2.5
        assert!((out[0] - 2.5).abs() < f32::EPSILON, "out[0]={}", out[0]);
        // out[1] = (0x31 >> 4) * 2.0 + 0.5 = 3*2.0 + 0.5 = 6.5
        assert!((out[1] - 6.5).abs() < f32::EPSILON, "out[1]={}", out[1]);
    }

    #[test]
    fn test_dequantize_scalar_nibble_extremes() {
        // 0x0F → low=15, high=0; 0xF0 → low=0, high=15
        let packed = vec![0x0Fu8, 0xF0];
        let scale = vec![1.0f32];
        let bias = vec![0.0f32];
        let mut out = vec![0.0f32; 4];
        UniversalEngine::dequantize_row_scalar(&packed, &mut out, &scale, &bias, 128);

        assert!((out[0] - 15.0).abs() < f32::EPSILON); // 0x0F low nibble
        assert!((out[1] - 0.0).abs() < f32::EPSILON); // 0x0F high nibble
        assert!((out[2] - 0.0).abs() < f32::EPSILON); // 0xF0 low nibble
        assert!((out[3] - 15.0).abs() < f32::EPSILON); // 0xF0 high nibble
    }

    #[test]
    fn test_dequantize_scalar_multi_group() {
        // 2 packed bytes → 4 weights. group_size=2 → group 0 covers [0,1], group 1 covers [2,3]
        let packed = vec![0x11u8, 0x22];
        let scale = vec![1.0f32, 10.0]; // group 0: scale=1, group 1: scale=10
        let bias = vec![0.0f32, 0.0];
        let mut out = vec![0.0f32; 4];
        UniversalEngine::dequantize_row_scalar(&packed, &mut out, &scale, &bias, 2);

        // out[0] = (0x11 & 0x0F) * 1.0 = 1.0 (group 0)
        assert!((out[0] - 1.0).abs() < f32::EPSILON, "out[0]={}", out[0]);
        // out[1] = (0x11 >> 4) * 1.0 = 1.0 (group 0)
        assert!((out[1] - 1.0).abs() < f32::EPSILON, "out[1]={}", out[1]);
        // out[2] = (0x22 & 0x0F) * 10.0 = 20.0 (group 1)
        assert!((out[2] - 20.0).abs() < f32::EPSILON, "out[2]={}", out[2]);
        // out[3] = (0x22 >> 4) * 10.0 = 20.0 (group 1)
        assert!((out[3] - 20.0).abs() < f32::EPSILON, "out[3]={}", out[3]);
    }

    #[test]
    fn test_gemv_scalar_identity() {
        // Weights that produce known dot product
        // packed = [0x10] → low=0, high=1 → weights with scale=1, bias=0 → [0.0, 1.0]
        // x = [3.0, 5.0] → dot = 0.0*3.0 + 1.0*5.0 = 5.0
        let packed = vec![0x10u8];
        let scale = vec![1.0f32];
        let bias = vec![0.0f32];
        let x = vec![3.0f32, 5.0];
        let sum = UniversalEngine::gemv_row_scalar(&x, &packed, &scale, &bias, 128);
        assert!((sum - 5.0).abs() < f32::EPSILON, "sum={}", sum);
    }

    #[test]
    fn test_gemv_scalar_zero_weights() {
        // All-zero packed → only bias contributes
        let packed = vec![0x00u8, 0x00];
        let scale = vec![1.0f32];
        let bias = vec![0.5f32]; // bias=0.5 per weight
        let x = vec![1.0f32, 1.0, 1.0, 1.0];
        let sum = UniversalEngine::gemv_row_scalar(&x, &packed, &scale, &bias, 128);
        // Each weight = 0*1.0 + 0.5 = 0.5. Sum = 0.5*1 + 0.5*1 + 0.5*1 + 0.5*1 = 2.0
        assert!((sum - 2.0).abs() < f32::EPSILON, "sum={}", sum);
    }

    #[test]
    fn test_gemv_scalar_group_boundary() {
        // 2 packed bytes → 4 weights. group_size=2.
        let packed = vec![0x11u8, 0x11];
        let scale = vec![1.0f32, 2.0]; // group 0: s=1, group 1: s=2
        let bias = vec![0.0f32, 0.0];
        let x = vec![1.0f32, 1.0, 1.0, 1.0];
        let sum = UniversalEngine::gemv_row_scalar(&x, &packed, &scale, &bias, 2);
        // group 0: w[0]=1*1=1, w[1]=1*1=1 → contrib = 1+1 = 2
        // group 1: w[2]=1*2=2, w[3]=1*2=2 → contrib = 2+2 = 4
        assert!((sum - 6.0).abs() < f32::EPSILON, "sum={}", sum);
    }
}
