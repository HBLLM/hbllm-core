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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> HashMap<String, Vec<(String, String)>> {
        let mut adj = HashMap::new();
        adj.insert("A".into(), vec![
            ("B".into(), "knows".into()),
            ("C".into(), "likes".into()),
        ]);
        adj.insert("B".into(), vec![
            ("D".into(), "works_with".into()),
        ]);
        adj.insert("C".into(), vec![
            ("D".into(), "related_to".into()),
        ]);
        adj
    }

    #[test]
    fn test_bfs_direct_neighbor() {
        let adj = make_graph();
        let path = bfs_shortest_path(&adj, "A", "B", 5);
        assert_eq!(path, vec!["A", "B"]);
    }

    #[test]
    fn test_bfs_two_hops() {
        let adj = make_graph();
        let path = bfs_shortest_path(&adj, "A", "D", 5);
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], "A");
        assert_eq!(path[2], "D");
    }

    #[test]
    fn test_bfs_no_path() {
        let adj = make_graph();
        let path = bfs_shortest_path(&adj, "D", "A", 10); // No reverse edges
        assert!(path.is_empty());
    }

    #[test]
    fn test_bfs_same_node() {
        let adj = make_graph();
        let path = bfs_shortest_path(&adj, "A", "A", 5);
        assert_eq!(path, vec!["A"]);
    }

    #[test]
    fn test_bfs_max_depth() {
        let adj = make_graph();
        // A→B→D is 3 nodes (2 hops). With max_depth=1, path length limited to 1 hop.
        let path = bfs_shortest_path(&adj, "A", "D", 1);
        assert!(path.is_empty(), "Should not find D at depth 1");
    }

    #[test]
    fn test_subgraph_depth_0() {
        let adj = make_graph();
        let (nodes, edges) = extract_subgraph(&adj, "A", 0);
        assert_eq!(nodes.len(), 1);
        assert!(nodes.contains(&"A".to_string()));
        assert!(edges.is_empty());
    }

    #[test]
    fn test_subgraph_depth_1() {
        let adj = make_graph();
        let (nodes, edges) = extract_subgraph(&adj, "A", 1);
        assert!(nodes.contains(&"A".to_string()));
        assert!(nodes.contains(&"B".to_string()));
        assert!(nodes.contains(&"C".to_string()));
        assert_eq!(edges.len(), 2); // A→B and A→C
    }

    #[test]
    fn test_jaccard_identical() {
        let sim = jaccard_similarity("hello", "hello");
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let sim = jaccard_similarity("abc", "xyz");
        assert!((sim - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jaccard_empty_strings() {
        let sim = jaccard_similarity("", "");
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_find_similar_threshold() {
        let entities = vec![
            "machine learning".to_string(),
            "machine learning model".to_string(),
            "quantum physics".to_string(),
        ];
        let results = find_similar_entities(&entities, 0.5);
        // "machine learning" and "machine learning model" should be similar
        assert!(!results.is_empty());
        assert!(results[0].2 >= 0.5);
        // "quantum physics" should not match
        let has_quantum_ml_pair = results.iter().any(|(a, b, _)| {
            (a.contains("quantum") && b.contains("machine"))
                || (a.contains("machine") && b.contains("quantum"))
        });
        assert!(!has_quantum_ml_pair);
    }
}
