//! Versioned Canonical HCIR State Hashing with BLAKE3.
//!
//! Enforces deterministic serialization:
//! 1. Header prefix `HCIR_STATE_V1`
//! 2. Nodes sorted deterministically by canonical Node ID
//! 3. Node attributes sorted by key
//! 4. Edges sorted deterministically by canonical Edge ID / (sources, type, targets) tuple
//! 5. Direct streaming into BLAKE3 hasher without JSON allocation overhead.

use crate::graph::{FastEdge, FastNode, GraphState};

pub const CANONICAL_HASH_VERSION: &str = "HCIR_STATE_V1";

pub fn compute_canonical_hash(state: &GraphState) -> String {
    let mut hasher = blake3::Hasher::new();

    // 1. Version header
    hasher.update(CANONICAL_HASH_VERSION.as_bytes());
    hasher.update(b"\n");

    // 2. Sorted nodes
    let mut node_keys: Vec<&String> = state.nodes.keys().collect();
    node_keys.sort_unstable();

    hasher.update((node_keys.len() as u64).to_le_bytes().as_slice());
    hasher.update(b"\n");

    for key in node_keys {
        if let Some(node) = state.nodes.get(key) {
            hash_node(&mut hasher, node);
        }
    }

    // 3. Sorted edges
    let mut edge_keys: Vec<&String> = state.edges.keys().collect();
    edge_keys.sort_unstable();

    hasher.update((edge_keys.len() as u64).to_le_bytes().as_slice());
    hasher.update(b"\n");

    for key in edge_keys {
        if let Some(edge) = state.edges.get(key) {
            hash_edge(&mut hasher, edge);
        }
    }

    hasher.finalize().to_hex().to_string()
}

fn hash_node(hasher: &mut blake3::Hasher, node: &FastNode) {
    hasher.update(b"NODE:");
    hasher.update(node.id.as_bytes());
    hasher.update(b"|");
    hasher.update(node.node_type.as_bytes());
    hasher.update(b"|");
    hasher.update(node.lifecycle.as_bytes());
    hasher.update(b"|");

    // Sort attributes by key
    let mut prop_keys: Vec<&String> = node.properties.keys().collect();
    prop_keys.sort_unstable();

    for pk in prop_keys {
        hasher.update(pk.as_bytes());
        hasher.update(b"=");
        if let Some(val) = node.properties.get(pk) {
            hasher.update(val.as_bytes());
        }
        hasher.update(b";");
    }
    hasher.update(b"\n");
}

fn hash_edge(hasher: &mut blake3::Hasher, edge: &FastEdge) {
    hasher.update(b"EDGE:");
    hasher.update(edge.id.as_bytes());
    hasher.update(b"|");
    hasher.update(edge.edge_type.as_bytes());
    hasher.update(b"|SRC:");

    let mut sorted_sources = edge.sources.clone();
    sorted_sources.sort_unstable();
    for s in sorted_sources {
        hasher.update(s.as_bytes());
        hasher.update(b",");
    }

    hasher.update(b"|TGT:");
    let mut sorted_targets = edge.targets.clone();
    sorted_targets.sort_unstable();
    for t in sorted_targets {
        hasher.update(t.as_bytes());
        hasher.update(b",");
    }

    hasher.update(b"|W:");
    hasher.update(edge.weight.to_bits().to_le_bytes().as_slice());
    hasher.update(b"\n");
}
