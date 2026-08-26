//! Native Relational Structure Matcher.
//!
//! Performs constraint-satisfaction graph isomorphism to find ranked candidate
//! variable-to-entity alignments and systematicity scores for analogical transfer.

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatternEdge {
    pub rel_type: String,
    pub source_var: String,
    pub target_var: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pattern {
    pub variables: Vec<String>,
    pub edges: Vec<PatternEdge>,
    pub surface_distractors: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TargetEdge {
    pub rel_type: String,
    pub source: String,
    pub target: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TargetGraph {
    pub nodes: Vec<String>,
    pub edges: Vec<TargetEdge>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateAlignment {
    pub mapping: HashMap<String, String>,
    pub systematicity_score: f64,
    pub matched_relations_count: usize,
    pub total_relations_count: usize,
    pub surface_overlap_score: f64,
    pub structural_consistency: bool,
}

/// Finds all valid 1-to-1 variable-to-entity mappings and ranks them by systematicity.
pub fn match_relational_schema(
    pattern: &Pattern,
    target: &TargetGraph,
    min_systematicity: f64,
) -> Vec<CandidateAlignment> {
    if pattern.variables.is_empty() || pattern.edges.is_empty() || target.nodes.is_empty() {
        return Vec::new();
    }

    // Build target edge lookup: (rel_type, source, target) -> bool
    let mut target_edges_set: HashSet<(String, String, String)> = HashSet::new();
    for e in &target.edges {
        target_edges_set.insert((e.rel_type.clone(), e.source.clone(), e.target.clone()));
    }

    let mut candidate_alignments = Vec::new();
    let mut current_mapping = HashMap::new();
    let mut used_target_nodes = HashSet::new();

    backtrack_match(
        0,
        pattern,
        target,
        &target_edges_set,
        &mut current_mapping,
        &mut used_target_nodes,
        min_systematicity,
        &mut candidate_alignments,
    );

    // Sort ranked candidate alignments descending by systematicity score
    candidate_alignments.sort_by(|a, b| {
        b.systematicity_score
            .partial_cmp(&a.systematicity_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidate_alignments
}

fn backtrack_match(
    var_idx: usize,
    pattern: &Pattern,
    target: &TargetGraph,
    target_edges_set: &HashSet<(String, String, String)>,
    current_mapping: &mut HashMap<String, String>,
    used_target_nodes: &mut HashSet<String>,
    min_systematicity: f64,
    results: &mut Vec<CandidateAlignment>,
) {
    if var_idx == pattern.variables.len() {
        // Complete assignment — evaluate systematicity
        let mut matched_count = 0usize;
        let total_count = pattern.edges.len();

        for pe in &pattern.edges {
            if let (Some(src_node), Some(tgt_node)) = (
                current_mapping.get(&pe.source_var),
                current_mapping.get(&pe.target_var),
            ) {
                if target_edges_set.contains(&(pe.rel_type.clone(), src_node.clone(), tgt_node.clone())) {
                    matched_count += 1;
                }
            }
        }

        let systematicity = if total_count > 0 {
            matched_count as f64 / total_count as f64
        } else {
            0.0
        };

        if systematicity >= min_systematicity && matched_count > 0 {
            results.push(CandidateAlignment {
                mapping: current_mapping.clone(),
                systematicity_score: systematicity,
                matched_relations_count: matched_count,
                total_relations_count: total_count,
                surface_overlap_score: 0.0,
                structural_consistency: matched_count == total_count,
            });
        }
        return;
    }

    let var_name = &pattern.variables[var_idx];

    for node in &target.nodes {
        if !used_target_nodes.contains(node) {
            // Assign var -> node
            current_mapping.insert(var_name.clone(), node.clone());
            used_target_nodes.insert(node.clone());

            backtrack_match(
                var_idx + 1,
                pattern,
                target,
                target_edges_set,
                current_mapping,
                used_target_nodes,
                min_systematicity,
                results,
            );

            // Backtrack
            current_mapping.remove(var_name);
            used_target_nodes.remove(node);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analogical_isomorphism_ranking() {
        // Solar system schema: Sun AT_CENTER_OF Earth, Sun CAUSES_ORBIT Earth
        let pattern = Pattern {
            variables: vec!["X".to_string(), "Y".to_string()],
            edges: vec![
                PatternEdge {
                    rel_type: "CENTRAL_TO".to_string(),
                    source_var: "X".to_string(),
                    target_var: "Y".to_string(),
                },
                PatternEdge {
                    rel_type: "ATTRACTS".to_string(),
                    source_var: "X".to_string(),
                    target_var: "Y".to_string(),
                },
            ],
            surface_distractors: vec!["hot".to_string(), "yellow".to_string()],
        };

        // Atom target domain: Nucleus, Electron, Wall
        let target = TargetGraph {
            nodes: vec!["nucleus".to_string(), "electron".to_string(), "wall".to_string()],
            edges: vec![
                TargetEdge {
                    rel_type: "CENTRAL_TO".to_string(),
                    source: "nucleus".to_string(),
                    target: "electron".to_string(),
                },
                TargetEdge {
                    rel_type: "ATTRACTS".to_string(),
                    source: "nucleus".to_string(),
                    target: "electron".to_string(),
                },
                TargetEdge {
                    rel_type: "CENTRAL_TO".to_string(),
                    source: "wall".to_string(),
                    target: "electron".to_string(),
                },
            ],
        };

        let alignments = match_relational_schema(&pattern, &target, 0.5);
        assert!(!alignments.is_empty());

        let top = &alignments[0];
        assert_eq!(top.mapping.get("X").unwrap(), "nucleus");
        assert_eq!(top.mapping.get("Y").unwrap(), "electron");
        assert_eq!(top.systematicity_score, 1.0);
        assert!(top.structural_consistency);
    }
}
