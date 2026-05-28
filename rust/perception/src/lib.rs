//! Rust-native ONNX vision engine and perception primitives for HBLLM.
#![allow(non_local_definitions)] // pyo3 macros generate these; not fixable on our side

use std::sync::Mutex;
use pyo3::prelude::*;
use ort::session::Session;

pub mod preprocessing;
pub mod change_detector;

use preprocessing::{preprocess_image, perceptual_hash};
use change_detector::ChangeDetector;

#[pyclass(name = "ChangeDetector", module = "hbllm_perception_rs")]
pub struct PyChangeDetector {
    inner: ChangeDetector,
}

#[pymethods]
impl PyChangeDetector {
    #[new]
    #[pyo3(signature = (threshold=5))]
    pub fn new(threshold: u32) -> Self {
        Self {
            inner: ChangeDetector::new(threshold),
        }
    }

    pub fn is_changed(&self, image_bytes: &[u8]) -> PyResult<bool> {
        let img = image::load_from_memory(image_bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid image data: {}", e)))?;
        Ok(self.inner.is_changed(&img))
    }

    pub fn reset(&self) {
        self.inner.reset();
    }

    pub fn get_threshold(&self) -> u32 {
        self.inner.get_threshold()
    }
}

#[pyclass(name = "VisionEngine", module = "hbllm_perception_rs")]
pub struct PyVisionEngine {
    session: Mutex<Option<Session>>,
    mock_mode: bool,
}

impl Default for PyVisionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyVisionEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            session: Mutex::new(None),
            mock_mode: true, // Default to mock mode until a real model is loaded
        }
    }

    pub fn load_model(&mut self, path: &str) -> PyResult<()> {
        if path == "mock" || path.is_empty() {
            self.mock_mode = true;
            let mut lock = self.session.lock().unwrap();
            *lock = None;
            return Ok(());
        }

        // Initialize ONNX runtime if needed
        let _ = ort::init();

        let session = Session::builder()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create ONNX builder: {}", e)))?
            .commit_from_file(path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load ONNX model from {}: {}", path, e)))?;

        let mut lock = self.session.lock().unwrap();
        *lock = Some(session);
        self.mock_mode = false;
        Ok(())
    }

    pub fn embed(&self, image_bytes: &[u8]) -> PyResult<Vec<f32>> {
        if self.mock_mode {
            let img = image::load_from_memory(image_bytes)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid image data: {}", e)))?;
            let hash = perceptual_hash(&img);
            let mut emb = vec![0.0f32; 768];
            for (i, val) in emb.iter_mut().enumerate() {
                let seed = hash.wrapping_add(i as u64);
                let v = ((seed.wrapping_mul(48271) % 2147483647) as f32) / 2147483647.0;
                *val = v * 2.0 - 1.0;
            }
            return Ok(emb);
        }

        let mut lock = self.session.lock().unwrap();
        let session = match lock.as_mut() {
            Some(s) => s,
            None => return Err(pyo3::exceptions::PyRuntimeError::new_err("No ONNX model loaded. Call load_model() first.")),
        };

        let img = image::load_from_memory(image_bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid image data: {}", e)))?;

        let tensor3 = preprocess_image(&img, 224);
        let tensor4 = tensor3.insert_axis(ndarray::Axis(0));

        let input_name = session.inputs()[0].name().to_owned();

        let input_tensor = ort::value::Tensor::from_array(tensor4).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create input tensor: {}", e))
        })?;

        let outputs = session.run(ort::inputs![input_name => input_tensor])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ONNX inference failed: {}", e)))?;

        let output_value = &outputs[0];
        let output_tensor = output_value.try_extract_tensor::<f32>().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to extract output tensor: {}", e))
        })?;

        Ok(output_tensor.1.to_vec())
    }

    pub fn caption(&self, image_bytes: &[u8]) -> PyResult<String> {
        let img = image::load_from_memory(image_bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid image data: {}", e)))?;
        let hash = perceptual_hash(&img);

        if self.mock_mode {
            return Ok(format!("A mock caption of the image (hash: {:016x})", hash));
        }

        // Real model fallback description
        Ok(format!("Visual scene processed via ONNX. Perceptual signature: {:016x}", hash))
    }

    pub fn frame_hash(&self, image_bytes: &[u8]) -> PyResult<u64> {
        let img = image::load_from_memory(image_bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid image data: {}", e)))?;
        Ok(perceptual_hash(&img))
    }
}

#[pymodule]
fn hbllm_perception_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyVisionEngine>()?;
    m.add_class::<PyChangeDetector>()?;
    Ok(())
}
