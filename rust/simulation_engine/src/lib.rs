#![allow(non_local_definitions)]
//! Native Mental Simulation Engine & Resident Cognitive Runtime — PyO3 Bindings.

pub mod geometry;
pub mod rollout;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

use crate::geometry::{evaluate_support_stability as rust_eval_support, FastAABB};
use crate::rollout::{
    evaluate_parallel_rollouts as rust_eval_rollouts, ActionStep, BranchSpec,
    NativeCognitiveRuntime as RustRuntime,
};

#[pyclass]
pub struct NativeCognitiveRuntime {
    inner: RustRuntime,
}

impl Default for NativeCognitiveRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl NativeCognitiveRuntime {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RustRuntime::new(),
        }
    }

    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    pub fn canonical_hash(&self) -> String {
        self.inner.canonical_hash()
    }

    pub fn add_node(
        &self,
        id: String,
        node_type: String,
        lifecycle: String,
        properties: HashMap<String, String>,
        created_at: f64,
    ) {
        self.inner
            .add_node(id, node_type, lifecycle, properties, created_at);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_edge(
        &self,
        id: String,
        edge_type: String,
        sources: Vec<String>,
        targets: Vec<String>,
        weight: f64,
        properties: HashMap<String, String>,
        created_at: f64,
    ) {
        self.inner.add_edge(
            id, edge_type, sources, targets, weight, properties, created_at,
        );
    }

    pub fn clear_cache(&self) {
        self.inner.clear_cache();
    }

    pub fn cache_size(&self) -> usize {
        self.inner.cache_size()
    }

    /// Evaluates branch rollouts against resident state with DAG memoization, releasing the GIL.
    pub fn evaluate_rollouts<'py>(
        &self,
        py: Python<'py>,
        branches: &PyList,
        seed_hash: String,
    ) -> PyResult<(&'py PyList, &'py PyDict)> {
        // 1. Extract BranchSpecs from Python
        let mut specs = Vec::with_capacity(branches.len());
        for item in branches.iter() {
            let dict: &PyDict = item.extract()?;
            let branch_id: u32 = dict
                .get_item("branch_id")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0);
            let initial_risk: f64 = dict
                .get_item("initial_risk")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.0);
            let initial_cost: f64 = dict
                .get_item("initial_cost")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.0);
            let max_steps: u32 = dict
                .get_item("max_steps")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(20);

            let py_actions: &PyList = dict
                .get_item("actions")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_else(|| PyList::empty(py));
            let mut actions = Vec::with_capacity(py_actions.len());

            for act_item in py_actions.iter() {
                let act_dict: &PyDict = act_item.extract()?;
                let operator: String = act_dict
                    .get_item("operator")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_default();
                let subject: String = act_dict
                    .get_item("subject")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_default();
                let target: String = act_dict
                    .get_item("target")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_default();
                let parameters: HashMap<String, f64> = act_dict
                    .get_item("parameters")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_default();

                actions.push(ActionStep {
                    operator,
                    subject,
                    target,
                    parameters,
                });
            }

            specs.push(BranchSpec {
                branch_id,
                actions,
                initial_risk,
                initial_cost,
                max_steps,
            });
        }

        // 2. Release GIL and evaluate
        let runtime = self.inner.clone();
        let (results, stats) =
            py.allow_threads(move || runtime.evaluate_rollouts(&specs, &seed_hash));

        // 3. Convert results to Python
        let py_results = PyList::empty(py);
        for r in results {
            let dict = PyDict::new(py);
            dict.set_item("branch_id", r.branch_id)?;
            dict.set_item("success_probability", r.success_probability)?;
            dict.set_item("risk_score", r.risk_score)?;
            dict.set_item("trajectory_cost", r.trajectory_cost)?;
            dict.set_item("final_state_hash", r.final_state_hash)?;
            dict.set_item("steps_executed", r.steps_executed)?;
            dict.set_item("terminal_status", r.terminal_status)?;
            py_results.append(dict)?;
        }

        let py_stats = PyDict::new(py);
        py_stats.set_item("cache_hits", stats.cache_hits)?;
        py_stats.set_item("cache_misses", stats.cache_misses)?;
        py_stats.set_item("transitions_evaluated", stats.transitions_evaluated)?;
        py_stats.set_item("branches_reused", stats.branches_reused)?;

        Ok((py_results, py_stats))
    }
}

#[pyfunction]
pub fn evaluate_support_stability(
    upper_bounds: (f64, f64, f64, f64, f64, f64),
    base_bounds: (f64, f64, f64, f64, f64, f64),
    tolerance: Option<f64>,
) -> (bool, f64) {
    let upper = FastAABB::new(
        upper_bounds.0,
        upper_bounds.1,
        upper_bounds.2,
        upper_bounds.3,
        upper_bounds.4,
        upper_bounds.5,
    );
    let base = FastAABB::new(
        base_bounds.0,
        base_bounds.1,
        base_bounds.2,
        base_bounds.3,
        base_bounds.4,
        base_bounds.5,
    );
    rust_eval_support(&upper, &base, tolerance.unwrap_or(0.05))
}

#[pyfunction]
pub fn evaluate_parallel_rollouts<'py>(
    py: Python<'py>,
    branches: &PyList,
    seed_hash: String,
) -> PyResult<&'py PyList> {
    let mut specs = Vec::with_capacity(branches.len());
    for item in branches.iter() {
        let dict: &PyDict = item.extract()?;
        let branch_id: u32 = dict
            .get_item("branch_id")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0);
        let initial_risk: f64 = dict
            .get_item("initial_risk")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0.0);
        let initial_cost: f64 = dict
            .get_item("initial_cost")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(0.0);
        let max_steps: u32 = dict
            .get_item("max_steps")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(20);

        let py_actions: &PyList = dict
            .get_item("actions")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_else(|| PyList::empty(py));
        let mut actions = Vec::with_capacity(py_actions.len());

        for act_item in py_actions.iter() {
            let act_dict: &PyDict = act_item.extract()?;
            let operator: String = act_dict
                .get_item("operator")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_default();
            let subject: String = act_dict
                .get_item("subject")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_default();
            let target: String = act_dict
                .get_item("target")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_default();
            let parameters: HashMap<String, f64> = act_dict
                .get_item("parameters")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_default();

            actions.push(ActionStep {
                operator,
                subject,
                target,
                parameters,
            });
        }

        specs.push(BranchSpec {
            branch_id,
            actions,
            initial_risk,
            initial_cost,
            max_steps,
        });
    }

    let results = py.allow_threads(move || rust_eval_rollouts(specs, &seed_hash));

    let py_results = PyList::empty(py);
    for r in results {
        let dict = PyDict::new(py);
        dict.set_item("branch_id", r.branch_id)?;
        dict.set_item("success_probability", r.success_probability)?;
        dict.set_item("risk_score", r.risk_score)?;
        dict.set_item("trajectory_cost", r.trajectory_cost)?;
        dict.set_item("final_state_hash", r.final_state_hash)?;
        dict.set_item("steps_executed", r.steps_executed)?;
        dict.set_item("terminal_status", r.terminal_status)?;
        py_results.append(dict)?;
    }

    Ok(py_results)
}

#[pymodule]
fn hbllm_simulation_engine(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<NativeCognitiveRuntime>()?;
    m.add_function(wrap_pyfunction!(evaluate_support_stability, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_parallel_rollouts, m)?)?;
    Ok(())
}
