#![allow(non_local_definitions)]
//! Native HCIR Graph Substrate — PyO3 Bindings.

pub mod chunk;
pub mod graph;
pub mod hash;

use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;

use crate::graph::{FastEdge, FastNode, GraphState};

#[pyclass(name = "NativeGraph")]
pub struct PyNativeGraph {
    pub state: RwLock<GraphState>,
}

impl Default for PyNativeGraph {
    fn default() -> Self {
        Self::new()
    }
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

    #[allow(clippy::too_many_arguments)]
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
        for e in edges {
            let dict = PyDict::new(py);
            dict.set_item("id", &e.id)?;
            dict.set_item("edge_type", &e.edge_type)?;
            dict.set_item("sources", &e.sources)?;
            dict.set_item("targets", &e.targets)?;
            dict.set_item("weight", e.weight)?;
            dict.set_item("properties", &e.properties)?;
            dict.set_item("created_at", e.created_at)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    pub fn edges_to<'py>(&self, py: Python<'py>, node_id: &str) -> PyResult<&'py PyList> {
        let state = self.state.read();
        let edges = state.edges_to(node_id);
        let list = PyList::empty(py);
        for e in edges {
            let dict = PyDict::new(py);
            dict.set_item("id", &e.id)?;
            dict.set_item("edge_type", &e.edge_type)?;
            dict.set_item("sources", &e.sources)?;
            dict.set_item("targets", &e.targets)?;
            dict.set_item("weight", e.weight)?;
            dict.set_item("properties", &e.properties)?;
            dict.set_item("created_at", e.created_at)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    pub fn nodes_of_type<'py>(&self, py: Python<'py>, node_type: &str) -> PyResult<&'py PyList> {
        let state = self.state.read();
        let nodes = state.nodes_of_type(node_type);
        let list = PyList::empty(py);
        for n in nodes {
            let dict = PyDict::new(py);
            dict.set_item("id", &n.id)?;
            dict.set_item("node_type", &n.node_type)?;
            dict.set_item("lifecycle", &n.lifecycle)?;
            dict.set_item("properties", &n.properties)?;
            dict.set_item("created_at", n.created_at)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    pub fn bfs_path(&self, start: &str, end: &str, max_depth: Option<usize>) -> Vec<String> {
        self.state
            .read()
            .bfs_path(start, end, max_depth.unwrap_or(10))
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
            dict.set_item("created_at", n.created_at)?;
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
            dict.set_item("created_at", e.created_at)?;
            py_edges.append(dict)?;
        }

        Ok(PyTuple::new(py, [py_nodes, py_edges]))
    }

    pub fn canonical_hash(&self) -> String {
        self.state.read().canonical_hash()
    }
}

/// Canonical BLAKE3 hashing of arbitrary byte buffers (grids, AST chunks, state diffs).
#[pyfunction]
pub fn blake3_hash(data: &[u8]) -> String {
    let hash = blake3::hash(data);
    hash.to_hex().to_string()
}

/// Fast BFS connected components labeling for 2D spatial lattices.
#[pyfunction]
pub fn topological_connected_components(
    height: usize,
    width: usize,
    barriers: Vec<(usize, usize)>,
) -> Vec<Vec<(usize, usize)>> {
    if height == 0 || width == 0 {
        return Vec::new();
    }

    let mut is_barrier = vec![false; height * width];
    for (r, c) in barriers {
        if r < height && c < width {
            is_barrier[r * width + c] = true;
        }
    }

    let mut visited = vec![false; height * width];
    let mut components = Vec::new();
    let mut queue = std::collections::VecDeque::new();

    for r in 0..height {
        for c in 0..width {
            let idx = r * width + c;
            if is_barrier[idx] || visited[idx] {
                continue;
            }

            let mut comp = Vec::new();
            visited[idx] = true;
            queue.push_back((r, c));

            while let Some((curr_r, curr_c)) = queue.pop_front() {
                comp.push((curr_r, curr_c));

                if curr_r > 0 {
                    let nidx = (curr_r - 1) * width + curr_c;
                    if !is_barrier[nidx] && !visited[nidx] {
                        visited[nidx] = true;
                        queue.push_back((curr_r - 1, curr_c));
                    }
                }
                if curr_r + 1 < height {
                    let nidx = (curr_r + 1) * width + curr_c;
                    if !is_barrier[nidx] && !visited[nidx] {
                        visited[nidx] = true;
                        queue.push_back((curr_r + 1, curr_c));
                    }
                }
                if curr_c > 0 {
                    let nidx = curr_r * width + (curr_c - 1);
                    if !is_barrier[nidx] && !visited[nidx] {
                        visited[nidx] = true;
                        queue.push_back((curr_r, curr_c - 1));
                    }
                }
                if curr_c + 1 < width {
                    let nidx = curr_r * width + (curr_c + 1);
                    if !is_barrier[nidx] && !visited[nidx] {
                        visited[nidx] = true;
                        queue.push_back((curr_r, curr_c + 1));
                    }
                }
            }

            components.push(comp);
        }
    }

    components
}

/// Sweeps forward space-time collision cones against a target position over a time horizon.
/// Returns the earliest time_step (1-indexed) where an intercept occurs within clearance distance,
/// or None if no collision occurs within max_horizon.
///
/// trajectories: Vec<(init_r, init_c, vel_r, vel_c, max_steps)>
#[pyfunction]
pub fn fast_collision_cone_sweep(
    trajectories: Vec<(i32, i32, i32, i32, usize)>,
    target_pos: (i32, i32),
    max_horizon: usize,
    clearance: i32,
) -> Option<usize> {
    let (tr, tc) = target_pos;
    for step in 1..=max_horizon {
        let step_i32 = step as i32;
        for &(r0, c0, vr, vc, max_s) in &trajectories {
            if max_s > 0 && step > max_s {
                continue;
            }
            let pos_r = r0 + vr * step_i32;
            let pos_c = c0 + vc * step_i32;
            let dr = (pos_r - tr).abs();
            let dc = (pos_c - tc).abs();
            if dr <= clearance && dc <= clearance {
                return Some(step);
            }
        }
    }
    None
}

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone, PartialEq)]
struct AStarNode {
    cost: f64,
    f_score: f64,
    r: usize,
    c: usize,
}

impl Eq for AStarNode {}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f_score
            .partial_cmp(&self.f_score)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Computes the optimal geodesic collision-free path on a 2D grid via accelerated A*.
#[pyfunction]
pub fn fast_geodesic_astar(
    height: usize,
    width: usize,
    start: (usize, usize),
    goal: (usize, usize),
    barriers: Vec<(usize, usize)>,
    step_size: Option<usize>,
    max_iterations: Option<usize>,
) -> Vec<(usize, usize)> {
    if height == 0 || width == 0 {
        return Vec::new();
    }
    let (sr, sc) = start;
    let (gr, gc) = goal;
    if sr >= height || sc >= width || gr >= height || gc >= width {
        return Vec::new();
    }
    if start == goal {
        return vec![start];
    }

    let mut is_barrier = vec![false; height * width];
    for (r, c) in barriers {
        if r < height && c < width && (r, c) != goal {
            is_barrier[r * width + c] = true;
        }
    }

    let step = step_size.unwrap_or(1).max(1);
    let max_iter = max_iterations.unwrap_or(3500);

    let heuristic = |r: usize, c: usize| -> f64 {
        let dr = (gr as f64) - (r as f64);
        let dc = (gc as f64) - (c as f64);
        (dr * dr + dc * dc).sqrt()
    };

    let mut g_score = vec![f64::INFINITY; height * width];
    let mut came_from = vec![usize::MAX; height * width];
    let mut heap = BinaryHeap::new();

    let start_idx = sr * width + sc;
    g_score[start_idx] = 0.0;
    heap.push(AStarNode {
        cost: 0.0,
        f_score: heuristic(sr, sc),
        r: sr,
        c: sc,
    });

    let mut best_r = sr;
    let mut best_c = sc;
    let mut min_h = heuristic(sr, sc);
    let mut iters = 0;

    let deltas: [(i32, i32); 4] = [
        (-(step as i32), 0),
        (step as i32, 0),
        (0, -(step as i32)),
        (0, step as i32),
    ];

    while let Some(current) = heap.pop() {
        iters += 1;
        if iters > max_iter {
            break;
        }

        let curr_idx = current.r * width + current.c;
        if (current.r, current.c) == goal {
            let mut path = Vec::new();
            let mut curr = curr_idx;
            path.push((curr / width, curr % width));
            while came_from[curr] != usize::MAX {
                curr = came_from[curr];
                path.push((curr / width, curr % width));
            }
            path.reverse();
            return path;
        }

        let h_curr = heuristic(current.r, current.c);
        if h_curr < min_h {
            min_h = h_curr;
            best_r = current.r;
            best_c = current.c;
        }

        if current.cost > g_score[curr_idx] {
            continue;
        }

        for &(dr, dc) in &deltas {
            let nr_i = (current.r as i32) + dr;
            let nc_i = (current.c as i32) + dc;
            if nr_i < 0 || nr_i >= (height as i32) || nc_i < 0 || nc_i >= (width as i32) {
                continue;
            }
            let nr = nr_i as usize;
            let nc = nc_i as usize;
            let nidx = nr * width + nc;

            if is_barrier[nidx] {
                continue;
            }

            let tentative_g = current.cost + (step as f64);
            if tentative_g < g_score[nidx] {
                g_score[nidx] = tentative_g;
                came_from[nidx] = curr_idx;
                let f = tentative_g + heuristic(nr, nc);
                heap.push(AStarNode {
                    cost: tentative_g,
                    f_score: f,
                    r: nr,
                    c: nc,
                });
            }
        }
    }

    let mut path = Vec::new();
    let mut curr = best_r * width + best_c;
    path.push((best_r, best_c));
    while came_from[curr] != usize::MAX {
        curr = came_from[curr];
        path.push((curr / width, curr % width));
    }
    path.reverse();
    path
}

#[pymodule]
fn hbllm_hcir_graph(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNativeGraph>()?;
    m.add_function(wrap_pyfunction!(blake3_hash, m)?)?;
    m.add_function(wrap_pyfunction!(topological_connected_components, m)?)?;
    m.add_function(wrap_pyfunction!(fast_collision_cone_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(fast_geodesic_astar, m)?)?;
    Ok(())
}
