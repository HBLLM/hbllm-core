//! High-performance persistent GraphState with chunked structural sharing.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::chunk::ChunkedStore;
use crate::hash::compute_canonical_hash;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FastNode {
    pub id: String,
    pub node_type: String,
    pub lifecycle: String,
    pub properties: HashMap<String, String>,
    pub created_at: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FastEdge {
    pub id: String,
    pub edge_type: String,
    pub sources: Vec<String>,
    pub targets: Vec<String>,
    pub weight: f64,
    pub properties: HashMap<String, String>,
    pub created_at: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StateDelta {
    pub added_nodes: Vec<FastNode>,
    pub modified_nodes: Vec<FastNode>,
    pub removed_node_ids: Vec<String>,
    pub added_edges: Vec<FastEdge>,
    pub removed_edge_ids: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct GraphState {
    pub nodes: ChunkedStore<FastNode>,
    pub edges: ChunkedStore<FastEdge>,
    pub outgoing_index: ChunkedStore<Vec<String>>,
    pub incoming_index: ChunkedStore<Vec<String>>,
    pub type_index: ChunkedStore<Vec<String>>,
}

impl GraphState {
    pub fn new() -> Self {
        Self {
            nodes: ChunkedStore::new(),
            edges: ChunkedStore::new(),
            outgoing_index: ChunkedStore::new(),
            incoming_index: ChunkedStore::new(),
            type_index: ChunkedStore::new(),
        }
    }

    /// O(1) root snapshot creation via chunk-granular structural sharing.
    pub fn snapshot(&self) -> Self {
        self.clone()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn get_node(&self, id: &str) -> Option<&FastNode> {
        self.nodes.get(id)
    }

    pub fn get_edge(&self, id: &str) -> Option<&FastEdge> {
        self.edges.get(id)
    }

    pub fn contains_node(&self, id: &str) -> bool {
        self.nodes.contains_key(id)
    }

    pub fn contains_edge(&self, id: &str) -> bool {
        self.edges.contains_key(id)
    }

    /// Returns a new GraphState with the node inserted/updated, copying only the affected chunk.
    pub fn with_node_added(&self, node: FastNode) -> Self {
        let node_id = node.id.clone();
        let node_type = node.node_type.clone();

        let new_nodes = self.nodes.with_inserted(node_id.clone(), node);

        // Update type index
        let mut current_type_ids = self.type_index.get(&node_type).cloned().unwrap_or_default();
        if !current_type_ids.contains(&node_id) {
            current_type_ids.push(node_id);
        }
        let new_type_index = self.type_index.with_inserted(node_type, current_type_ids);

        Self {
            nodes: new_nodes,
            edges: self.edges.clone(),
            outgoing_index: self.outgoing_index.clone(),
            incoming_index: self.incoming_index.clone(),
            type_index: new_type_index,
        }
    }

    /// Returns a new GraphState with the edge inserted/updated, copying only affected chunks.
    pub fn with_edge_added(&self, edge: FastEdge) -> Self {
        let edge_id = edge.id.clone();
        let sources = edge.sources.clone();
        let targets = edge.targets.clone();

        let new_edges = self.edges.with_inserted(edge_id.clone(), edge);

        // Update outgoing indices for each source
        let mut new_outgoing = self.outgoing_index.clone();
        for src in &sources {
            let mut list = new_outgoing.get(src).cloned().unwrap_or_default();
            if !list.contains(&edge_id) {
                list.push(edge_id.clone());
            }
            new_outgoing = new_outgoing.with_inserted(src.clone(), list);
        }

        // Update incoming indices for each target
        let mut new_incoming = self.incoming_index.clone();
        for tgt in &targets {
            let mut list = new_incoming.get(tgt).cloned().unwrap_or_default();
            if !list.contains(&edge_id) {
                list.push(edge_id.clone());
            }
            new_incoming = new_incoming.with_inserted(tgt.clone(), list);
        }

        Self {
            nodes: self.nodes.clone(),
            edges: new_edges,
            outgoing_index: new_outgoing,
            incoming_index: new_incoming,
            type_index: self.type_index.clone(),
        }
    }

    /// Applies a StateDelta to create the next immutable GraphState version.
    pub fn with_delta_applied(&self, delta: &StateDelta) -> Self {
        let mut next = self.clone();
        for node in &delta.added_nodes {
            next = next.with_node_added(node.clone());
        }
        for node in &delta.modified_nodes {
            next = next.with_node_added(node.clone());
        }
        for edge in &delta.added_edges {
            next = next.with_edge_added(edge.clone());
        }
        next
    }

    pub fn edges_from(&self, node_id: &str) -> Vec<&FastEdge> {
        if let Some(edge_ids) = self.outgoing_index.get(node_id) {
            edge_ids.iter().filter_map(|eid| self.edges.get(eid)).collect()
        } else {
            Vec::new()
        }
    }

    pub fn edges_to(&self, node_id: &str) -> Vec<&FastEdge> {
        if let Some(edge_ids) = self.incoming_index.get(node_id) {
            edge_ids.iter().filter_map(|eid| self.edges.get(eid)).collect()
        } else {
            Vec::new()
        }
    }

    pub fn nodes_of_type(&self, node_type: &str) -> Vec<&FastNode> {
        if let Some(node_ids) = self.type_index.get(node_type) {
            node_ids.iter().filter_map(|nid| self.nodes.get(nid)).collect()
        } else {
            Vec::new()
        }
    }

    pub fn bfs_path(&self, start: &str, end: &str, max_depth: usize) -> Vec<String> {
        if start == end {
            return vec![start.to_string()];
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, Vec<String>)> = VecDeque::new();

        visited.insert(start.to_string());
        queue.push_back((start.to_string(), vec![start.to_string()]));

        while let Some((curr, path)) = queue.pop_front() {
            if path.len() > max_depth {
                break;
            }

            for edge in self.edges_from(&curr) {
                for target in &edge.targets {
                    if target == end {
                        let mut res = path.clone();
                        res.push(target.clone());
                        return res;
                    }
                    if !visited.contains(target) {
                        visited.insert(target.clone());
                        let mut new_path = path.clone();
                        new_path.push(target.clone());
                        queue.push_back((target.clone(), new_path));
                    }
                }
            }
        }
        Vec::new()
    }

    pub fn subgraph_bfs(&self, center_id: &str, max_depth: usize) -> (Vec<FastNode>, Vec<FastEdge>) {
        let mut visited_nodes: HashSet<String> = HashSet::new();
        let mut collected_edges: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        if self.contains_node(center_id) {
            visited_nodes.insert(center_id.to_string());
            queue.push_back((center_id.to_string(), 0));
        }

        while let Some((curr, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            for edge in self.edges_from(&curr) {
                collected_edges.insert(edge.id.clone());
                for tgt in &edge.targets {
                    if !visited_nodes.contains(tgt) {
                        visited_nodes.insert(tgt.clone());
                        queue.push_back((tgt.clone(), depth + 1));
                    }
                }
            }

            for edge in self.edges_to(&curr) {
                collected_edges.insert(edge.id.clone());
                for src in &edge.sources {
                    if !visited_nodes.contains(src) {
                        visited_nodes.insert(src.clone());
                        queue.push_back((src.clone(), depth + 1));
                    }
                }
            }
        }

        let nodes: Vec<FastNode> = visited_nodes
            .iter()
            .filter_map(|nid| self.get_node(nid).cloned())
            .collect();
        let edges: Vec<FastEdge> = collected_edges
            .iter()
            .filter_map(|eid| self.get_edge(eid).cloned())
            .collect();

        (nodes, edges)
    }

    pub fn canonical_hash(&self) -> String {
        compute_canonical_hash(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: &str, node_type: &str) -> FastNode {
        FastNode {
            id: id.to_string(),
            node_type: node_type.to_string(),
            lifecycle: "ACTIVE".to_string(),
            properties: HashMap::new(),
            created_at: 1000.0,
        }
    }

    fn make_edge(id: &str, edge_type: &str, src: &str, tgt: &str) -> FastEdge {
        FastEdge {
            id: id.to_string(),
            edge_type: edge_type.to_string(),
            sources: vec![src.to_string()],
            targets: vec![tgt.to_string()],
            weight: 1.0,
            properties: HashMap::new(),
            created_at: 1000.0,
        }
    }

    #[test]
    fn test_snapshot_creation_and_mutation() {
        let root = GraphState::new();
        let root = root.with_node_added(make_node("n1", "Entity"));
        assert_eq!(root.node_count(), 1);

        let branch = root.snapshot();
        let branch = branch.with_node_added(make_node("n2", "Entity"));

        assert_eq!(root.node_count(), 1);
        assert_eq!(branch.node_count(), 2);
        assert!(branch.contains_node("n1"));
        assert!(branch.contains_node("n2"));
        assert!(!root.contains_node("n2"));
    }

    #[test]
    fn test_snapshot_isolation_tree() {
        let root = GraphState::new()
            .with_node_added(make_node("root_node", "Root"))
            .with_node_added(make_node("common", "Shared"));

        let snap_a = root.snapshot();
        let snap_b = root.snapshot();

        // Mutate A
        let mutated_a = snap_a
            .with_node_added(make_node("a_only", "BranchA"))
            .with_edge_added(make_edge("e_a", "CONNECTED", "root_node", "a_only"));

        // Assert root unchanged
        assert_eq!(root.node_count(), 2);
        assert_eq!(root.edge_count(), 0);
        assert!(!root.contains_node("a_only"));

        // Assert B unchanged
        assert_eq!(snap_b.node_count(), 2);
        assert_eq!(snap_b.edge_count(), 0);
        assert!(!snap_b.contains_node("a_only"));

        // Assert A changed
        assert_eq!(mutated_a.node_count(), 3);
        assert_eq!(mutated_a.edge_count(), 1);
        assert!(mutated_a.contains_node("a_only"));
    }

    #[test]
    fn test_nested_snapshot_isolation() {
        let root = GraphState::new().with_node_added(make_node("root", "Root"));
        let a = root.snapshot();
        let b = a.snapshot();
        let c = a.snapshot();

        let mutated_b = b.with_node_added(make_node("b_node", "B"));
        let mutated_c = c.with_node_added(make_node("c_node", "C"));

        assert_ne!(mutated_b.canonical_hash(), mutated_c.canonical_hash());
        assert_eq!(a.canonical_hash(), root.canonical_hash());
        assert_eq!(root.node_count(), 1);
    }

    #[test]
    fn test_canonical_hash_order_invariance() {
        let mut props1 = HashMap::new();
        props1.insert("color".to_string(), "red".to_string());
        props1.insert("size".to_string(), "large".to_string());

        let mut props2 = HashMap::new();
        props2.insert("size".to_string(), "large".to_string());
        props2.insert("color".to_string(), "red".to_string());

        let mut n1 = make_node("node_1", "Entity");
        n1.properties = props1;

        let mut n2 = make_node("node_2", "Entity");
        n2.properties = props2;

        // Insert n1 then n2
        let g1 = GraphState::new().with_node_added(n1.clone()).with_node_added(n2.clone());

        // Insert n2 then n1 (reversed order)
        let g2 = GraphState::new().with_node_added(n2).with_node_added(n1);

        assert_eq!(g1.canonical_hash(), g2.canonical_hash());
        assert!(g1.canonical_hash().len() > 10);
    }

    #[test]
    fn test_bfs_path_and_subgraph() {
        let g = GraphState::new()
            .with_node_added(make_node("A", "Node"))
            .with_node_added(make_node("B", "Node"))
            .with_node_added(make_node("C", "Node"))
            .with_edge_added(make_edge("e1", "NEXT", "A", "B"))
            .with_edge_added(make_edge("e2", "NEXT", "B", "C"));

        let path = g.bfs_path("A", "C", 5);
        assert_eq!(path, vec!["A", "B", "C"]);

        let (sub_nodes, sub_edges) = g.subgraph_bfs("B", 1);
        assert_eq!(sub_nodes.len(), 3);
        assert_eq!(sub_edges.len(), 2);
    }
}

