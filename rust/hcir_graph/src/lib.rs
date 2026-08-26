#![allow(non_local_definitions)]
//! Native HCIR Graph Substrate — PyO3 Bindings.

pub mod chunk;
pub mod graph;
pub mod hash;

use std::collections::HashMap;
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::graph::{FastEdge, FastNode, GraphState};

#[pyclass(name = "NativeGraph")]
pub struct PyNativeGraph {
    pub state: RwLock<GraphState>,
}

#[pymethods]
impl PyNativeGraph {
    #[new]
    pub fn new() -> Self {
        Self {
            state: RwLock::new(GraphState::new()),
        }
    }

    /// O(1) root snapshot creation via chunk-granular structural sharing.
    pub fn snapshot(&self) -> Self {
        let current = self.state.read();
        Self {
            state: RwLock::new(current.snapshot()),
        }
    }

    pub fn node_count(&self) -> usize {
        self.state.read().node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.state.read().edge_count()
    }

    pub fn has_node(&self, id: &str) -> bool {
        self.state.read().contains_node(id)
    }

    pub fn has_edge(&self, id: &str) -> bool {
        self.state.read().contains_edge(id)
    }

    pub fn add_node(
        &self,
        id: String,
        node_type: String,
        lifecycle: Option<String>,
        properties: Option<HashMap<String, String>>,
        created_at: Option<f64>,
    ) {
        let node = FastNode {
            id,
            node_type,
            lifecycle: lifecycle.unwrap_or_else(|| "ACTIVE".to_string()),
            properties: properties.unwrap_or_default(),
            created_at: created_at.unwrap_or(0.0),
        };
        let mut state = self.state.write();
        *state = state.with_node_added(node);
    }

    pub fn add_edge(
        &self,
        id: String,
        edge_type: String,
        sources: Vec<String>,
        targets: Vec<String>,
        weight: Option<f64>,
        properties: Option<HashMap<String, String>>,
        created_at: Option<f64>,
    ) {
        let edge = FastEdge {
            id,
            edge_type,
            sources,
            targets,
            weight: weight.unwrap_or(1.0),
            properties: properties.unwrap_or_default(),
            created_at: created_at.unwrap_or(0.0),
        };
        let mut state = self.state.write();
        *state = state.with_edge_added(edge);
    }

    pub fn get_node<'py>(&self, py: Python<'py>, id: &str) -> PyResult<Option<&'py PyDict>> {
        let state = self.state.read();
        if let Some(node) = state.get_node(id) {
            let dict = PyDict::new(py);
            dict.set_item("id", &node.id)?;
            dict.set_item("node_type", &node.node_type)?;
            dict.set_item("lifecycle", &node.lifecycle)?;
            dict.set_item("properties", &node.properties)?;
            dict.set_item("created_at", node.created_at)?;
            Ok(Some(dict))
        } else {
            Ok(None)
        }
    }

    pub fn get_edge<'py>(&self, py: Python<'py>, id: &str) -> PyResult<Option<&'py PyDict>> {
        let state = self.state.read();
        if let Some(edge) = state.get_edge(id) {
            let dict = PyDict::new(py);
            dict.set_item("id", &edge.id)?;
            dict.set_item("edge_type", &edge.edge_type)?;
            dict.set_item("sources", &edge.sources)?;
            dict.set_item("targets", &edge.targets)?;
            dict.set_item("weight", edge.weight)?;
            dict.set_item("properties", &edge.properties)?;
            dict.set_item("created_at", edge.created_at)?;
            Ok(Some(dict))
        } else {
            Ok(None)
        }
    }

    pub fn edges_from<'py>(&self, py: Python<'py>, node_id: &str) -> PyResult<&'py PyList> {
        let state = self.state.read();
        let edges = state.edges_from(node_id);
        let list = PyList::empty(py);
        for edge in edges {
            let dict = PyDict::new(py);
            dict.set_item("id", &edge.id)?;
            dict.set_item("edge_type", &edge.edge_type)?;
            dict.set_item("sources", &edge.sources)?;
            dict.set_item("targets", &edge.targets)?;
            dict.set_item("weight", edge.weight)?;
            dict.set_item("properties", &edge.properties)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    pub fn edges_to<'py>(&self, py: Python<'py>, node_id: &str) -> PyResult<&'py PyList> {
        let state = self.state.read();
        let edges = state.edges_to(node_id);
        let list = PyList::empty(py);
        for edge in edges {
            let dict = PyDict::new(py);
            dict.set_item("id", &edge.id)?;
            dict.set_item("edge_type", &edge.edge_type)?;
            dict.set_item("sources", &edge.sources)?;
            dict.set_item("targets", &edge.targets)?;
            dict.set_item("weight", edge.weight)?;
            dict.set_item("properties", &edge.properties)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    pub fn nodes_of_type<'py>(&self, py: Python<'py>, node_type: &str) -> PyResult<&'py PyList> {
        let state = self.state.read();
        let nodes = state.nodes_of_type(node_type);
        let list = PyList::empty(py);
        for node in nodes {
            let dict = PyDict::new(py);
            dict.set_item("id", &node.id)?;
            dict.set_item("node_type", &node.node_type)?;
            dict.set_item("lifecycle", &node.lifecycle)?;
            dict.set_item("properties", &node.properties)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    pub fn bfs_path(&self, start: &str, end: &str, max_depth: Option<usize>) -> Vec<String> {
        self.state.read().bfs_path(start, end, max_depth.unwrap_or(10))
    }

    pub fn subgraph_bfs<'py>(
        &self,
        py: Python<'py>,
        center_id: &str,
        max_depth: Option<usize>,
    ) -> PyResult<&'py PyTuple> {
        let state = self.state.read();
        let (nodes, edges) = state.subgraph_bfs(center_id, max_depth.unwrap_or(3));

        let py_nodes = PyList::empty(py);
        for n in nodes {
            let dict = PyDict::new(py);
            dict.set_item("id", &n.id)?;
            dict.set_item("node_type", &n.node_type)?;
            dict.set_item("lifecycle", &n.lifecycle)?;
            dict.set_item("properties", &n.properties)?;
            py_nodes.append(dict)?;
        }

        let py_edges = PyList::empty(py);
        for e in edges {
            let dict = PyDict::new(py);
            dict.set_item("id", &e.id)?;
            dict.set_item("edge_type", &e.edge_type)?;
            dict.set_item("sources", &e.sources)?;
            dict.set_item("targets", &e.targets)?;
            dict.set_item("weight", e.weight)?;
            dict.set_item("properties", &e.properties)?;
            py_edges.append(dict)?;
        }

        Ok(PyTuple::new(py, &[py_nodes, py_edges]))
    }

    pub fn canonical_hash(&self) -> String {
        self.state.read().canonical_hash()
    }
}

#[pymodule]
fn hbllm_hcir_graph(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNativeGraph>()?;
    Ok(())
}
