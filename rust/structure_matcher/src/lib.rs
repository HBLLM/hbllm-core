#![allow(non_local_definitions)]
//! Native Relational Structure Matcher — PyO3 Bindings.

pub mod matcher;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::matcher::{match_relational_schema as rust_match, Pattern, PatternEdge, TargetEdge, TargetGraph};

#[pyfunction]
pub fn match_relational_schema<'py>(
    py: Python<'py>,
    pattern_dict: &PyDict,
    target_dict: &PyDict,
    min_systematicity: Option<f64>,
) -> PyResult<&'py PyList> {
    // 1. Extract pattern
    let py_vars: &PyList = pattern_dict.get_item("variables")?.map(|v| v.extract()).transpose()?.unwrap_or_else(|| PyList::empty(py));
    let mut variables = Vec::with_capacity(py_vars.len());
    for v in py_vars.iter() {
        variables.push(v.extract::<String>()?);
    }

    let py_edges: &PyList = pattern_dict.get_item("edges")?.map(|v| v.extract()).transpose()?.unwrap_or_else(|| PyList::empty(py));
    let mut edges = Vec::with_capacity(py_edges.len());
    for e_item in py_edges.iter() {
        let e_dict: &PyDict = e_item.extract()?;
        let rel_type: String = e_dict.get_item("rel_type")?.map(|v| v.extract()).transpose()?.unwrap_or_default();
        let source_var: String = e_dict.get_item("source_var")?.map(|v| v.extract()).transpose()?.unwrap_or_default();
        let target_var: String = e_dict.get_item("target_var")?.map(|v| v.extract()).transpose()?.unwrap_or_default();

        edges.push(PatternEdge {
            rel_type,
            source_var,
            target_var,
        });
    }

    let pattern = Pattern {
        variables,
        edges,
        surface_distractors: Vec::new(),
    };

    // 2. Extract target graph
    let py_target_nodes: &PyList = target_dict.get_item("nodes")?.map(|v| v.extract()).transpose()?.unwrap_or_else(|| PyList::empty(py));
    let mut target_nodes = Vec::with_capacity(py_target_nodes.len());
    for n in py_target_nodes.iter() {
        target_nodes.push(n.extract::<String>()?);
    }

    let py_target_edges: &PyList = target_dict.get_item("edges")?.map(|v| v.extract()).transpose()?.unwrap_or_else(|| PyList::empty(py));
    let mut target_edges = Vec::with_capacity(py_target_edges.len());
    for te_item in py_target_edges.iter() {
        let te_dict: &PyDict = te_item.extract()?;
        let rel_type: String = te_dict.get_item("rel_type")?.map(|v| v.extract()).transpose()?.unwrap_or_default();
        let source: String = te_dict.get_item("source")?.map(|v| v.extract()).transpose()?.unwrap_or_default();
        let target: String = te_dict.get_item("target")?.map(|v| v.extract()).transpose()?.unwrap_or_default();

        target_edges.push(TargetEdge {
            rel_type,
            source,
            target,
        });
    }

    let target = TargetGraph {
        nodes: target_nodes,
        edges: target_edges,
    };

    // 3. Match
    let alignments = rust_match(&pattern, &target, min_systematicity.unwrap_or(0.3));

    // 4. Return ranked candidate alignments
    let py_results = PyList::empty(py);
    for a in alignments {
        let dict = PyDict::new(py);
        dict.set_item("mapping", a.mapping)?;
        dict.set_item("systematicity_score", a.systematicity_score)?;
        dict.set_item("matched_relations_count", a.matched_relations_count)?;
        dict.set_item("total_relations_count", a.total_relations_count)?;
        dict.set_item("surface_overlap_score", a.surface_overlap_score)?;
        dict.set_item("structural_consistency", a.structural_consistency)?;
        py_results.append(dict)?;
    }

    Ok(py_results)
}

#[pymodule]
fn hbllm_structure_matcher(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(match_relational_schema, m)?)?;
    Ok(())
}
