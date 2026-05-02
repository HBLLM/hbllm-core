#![allow(non_local_definitions)]
//! HBLLM Knowledge Graph — fast graph algorithms for entity-relation traversal.
//!
//! Provides:
//! - BFS shortest path
//! - Depth-limited subgraph extraction
//! - String similarity for entity disambiguation (Jaccard)
//! - Neighbor lookups via adjacency list

use pyo3::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

// ── Graph Algorithms ────────────────────────────────────────────────────────

/// BFS shortest path between two nodes in an adjacency list graph.
/// Returns the path as a list of node labels, or empty if no path exists.
fn bfs_shortest_path(
    adj: &HashMap<String, Vec<(String, String)>>, // node → [(neighbor, relation)]
    start: &str,
    end: &str,
    max_depth: usize,
) -> Vec<String> {
    if start == end {
        return vec![start.to_string()];
    }

    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, Vec<String>)> = VecDeque::new();

    visited.insert(start.to_string());
    queue.push_back((start.to_string(), vec![start.to_string()]));

    while let Some((current, path)) = queue.pop_front() {
        if path.len() > max_depth {
            break;
        }

        if let Some(neighbors) = adj.get(&current) {
            for (neighbor, _relation) in neighbors {
                if neighbor == end {
                    let mut result = path.clone();
                    result.push(neighbor.clone());
                    return result;
                }

                if !visited.contains(neighbor) {
                    visited.insert(neighbor.clone());
                    let mut new_path = path.clone();
                    new_path.push(neighbor.clone());
                    queue.push_back((neighbor.clone(), new_path));
                }
            }
        }
    }

    Vec::new() // No path found
}

/// Extract a subgraph around a center node up to a given depth.
/// Returns (nodes, edges) where edges are (source, relation, target) tuples.
fn extract_subgraph(
    adj: &HashMap<String, Vec<(String, String)>>,
    center: &str,
    depth: usize,
) -> (Vec<String>, Vec<(String, String, String)>) {
    let mut visited: HashSet<String> = HashSet::new();
    let mut edges: Vec<(String, String, String)> = Vec::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();

    visited.insert(center.to_string());
    queue.push_back((center.to_string(), 0));

    while let Some((current, current_depth)) = queue.pop_front() {
        if current_depth >= depth {
            continue;
        }

        if let Some(neighbors) = adj.get(&current) {
            for (neighbor, relation) in neighbors {
                edges.push((current.clone(), relation.clone(), neighbor.clone()));

                if !visited.contains(neighbor) {
                    visited.insert(neighbor.clone());
                    queue.push_back((neighbor.clone(), current_depth + 1));
                }
            }
        }
    }

    let nodes: Vec<String> = visited.into_iter().collect();
    (nodes, edges)
}

/// Jaccard similarity between two strings (character bigram sets).
fn jaccard_similarity(a: &str, b: &str) -> f64 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();

    let a_bigrams: HashSet<(char, char)> = a_lower.chars().zip(a_lower.chars().skip(1)).collect();
    let b_bigrams: HashSet<(char, char)> = b_lower.chars().zip(b_lower.chars().skip(1)).collect();

    if a_bigrams.is_empty() && b_bigrams.is_empty() {
        return 1.0;
    }

    let intersection = a_bigrams.intersection(&b_bigrams).count() as f64;
    let union = a_bigrams.union(&b_bigrams).count() as f64;

    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Find pairs of entities that are similar above a threshold.
/// Returns Vec<(entity_a, entity_b, similarity)>.
fn find_similar_entities(
    entities: &[String],
    threshold: f64,
) -> Vec<(String, String, f64)> {
    let mut results = Vec::new();

    for i in 0..entities.len() {
        for j in (i + 1)..entities.len() {
            let sim = jaccard_similarity(&entities[i], &entities[j]);
            if sim >= threshold {
                results.push((entities[i].clone(), entities[j].clone(), sim));
            }
        }
    }

    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ── Python Bindings ─────────────────────────────────────────────────────────

/// Fast BFS shortest path on an adjacency list.
/// adj_list: dict[str, list[tuple[str, str]]]  (node → [(neighbor, relation)])
#[pyfunction]
fn shortest_path(
    adj_list: HashMap<String, Vec<(String, String)>>,
    start: &str,
    end: &str,
    max_depth: usize,
) -> Vec<String> {
    bfs_shortest_path(&adj_list, start, end, max_depth)
}

/// Extract subgraph around a center node.
/// Returns (node_list, edge_list) where edges are (source, relation, target).
#[pyfunction]
fn subgraph(
    adj_list: HashMap<String, Vec<(String, String)>>,
    center: &str,
    depth: usize,
) -> (Vec<String>, Vec<(String, String, String)>) {
    extract_subgraph(&adj_list, center, depth)
}

/// Find similar entity pairs above a similarity threshold.
#[pyfunction]
fn disambiguate_entities(
    entities: Vec<String>,
    threshold: f64,
) -> Vec<(String, String, f64)> {
    find_similar_entities(&entities, threshold)
}

/// Compute Jaccard bigram similarity between two strings.
#[pyfunction]
fn entity_similarity(a: &str, b: &str) -> f64 {
    jaccard_similarity(a, b)
}

#[pymodule]
fn hbllm_knowledge_graph(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shortest_path, m)?)?;
    m.add_function(wrap_pyfunction!(subgraph, m)?)?;
    m.add_function(wrap_pyfunction!(disambiguate_entities, m)?)?;
    m.add_function(wrap_pyfunction!(entity_similarity, m)?)?;
    Ok(())
}
