//! Native Relational Structure Matcher — PyO3 Bindings.

pub mod matcher;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::matcher::{
    match_relational_schema_with_profile as rust_match_with_profile, Pattern, PatternEdge,
    TargetEdge, TargetGraph,
};

#[pyfunction]
#[pyo3(signature = (pattern_dict, target_dict, min_systematicity=None, profile_id=None))]
pub fn match_relational_schema<'py>(
    py: Python<'py>,
    pattern_dict: &Bound<'py, PyDict>,
    target_dict: &Bound<'py, PyDict>,
    min_systematicity: Option<f64>,
    profile_id: Option<&str>,
) -> PyResult<Bound<'py, PyList>> {
    // 1. Extract pattern
    let py_vars: Bound<'_, PyList> = pattern_dict
        .get_item("variables")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_else(|| PyList::empty(py));
    let mut variables = Vec::with_capacity(py_vars.len());
    for v in py_vars.iter() {
        variables.push(v.extract::<String>()?);
    }

    let py_edges: Bound<'_, PyList> = pattern_dict
        .get_item("edges")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_else(|| PyList::empty(py));
    let mut edges = Vec::with_capacity(py_edges.len());
    for e_item in py_edges.iter() {
        let e_dict = e_item.downcast::<PyDict>()?;
        let rel_type: String = e_dict
            .get_item("rel_type")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_default();
        let source_var: String = e_dict
            .get_item("source_var")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_default();
        let target_var: String = e_dict
            .get_item("target_var")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_default();
        let min_degree: Option<f64> = e_dict
            .get_item("min_degree")?
            .map(|v| v.extract())
            .transpose()?;

        edges.push(PatternEdge {
            rel_type,
            source_var,
            target_var,
            min_degree,
        });
    }

    let pattern = Pattern {
        variables,
        edges,
        surface_distractors: Vec::new(),
    };

    // 2. Extract target graph
    let py_target_nodes: Bound<'_, PyList> = target_dict
        .get_item("nodes")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_else(|| PyList::empty(py));
    let mut target_nodes = Vec::with_capacity(py_target_nodes.len());
    for n in py_target_nodes.iter() {
        target_nodes.push(n.extract::<String>()?);
    }

    let py_target_edges: Bound<'_, PyList> = target_dict
        .get_item("edges")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_else(|| PyList::empty(py));
    let mut target_edges = Vec::with_capacity(py_target_edges.len());
    for te_item in py_target_edges.iter() {
        let te_dict = te_item.downcast::<PyDict>()?;
        let rel_type: String = te_dict
            .get_item("rel_type")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_default();
        let source: String = te_dict
            .get_item("source")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_default();
        let target: String = te_dict
            .get_item("target")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or_default();
        let degree: f64 = te_dict
            .get_item("degree")?
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or(1.0);

        target_edges.push(TargetEdge {
            rel_type,
            source,
            target,
            degree,
        });
    }

    let target = TargetGraph {
        nodes: target_nodes,
        edges: target_edges,
    };

    // 3. Match
    let alignments = rust_match_with_profile(
        &pattern,
        &target,
        min_systematicity.unwrap_or(0.3),
        profile_id.unwrap_or("product"),
    );

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
        dict.set_item("structure_valid", a.structure_valid)?;
        dict.set_item("binding_valid", a.binding_valid)?;
        dict.set_item("semantic_degree", a.semantic_degree)?;
        dict.set_item("profile_id", a.profile_id)?;
        py_results.append(dict)?;
    }

    Ok(py_results)
}

#[pymodule]
fn hbllm_structure_matcher(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(match_relational_schema, m)?)?;
    Ok(())
}
