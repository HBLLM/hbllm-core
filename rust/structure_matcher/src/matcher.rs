//! Native Relational Structure Matcher.
//!
//! Performs constraint-satisfaction graph isomorphism to find ranked candidate
//! variable-to-entity alignments and systematicity scores for analogical transfer.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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

    let ctx = MatchContext {
        pattern,
        target,
        target_edges_set: &target_edges_set,
        min_systematicity,
    };

    let mut candidate_alignments = Vec::new();
    let mut current_mapping = HashMap::new();
    let mut used_target_nodes = HashSet::new();

    backtrack_match(
        0,
        &ctx,
        &mut current_mapping,
        &mut used_target_nodes,
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

struct MatchContext<'a> {
    pattern: &'a Pattern,
    target: &'a TargetGraph,
    target_edges_set: &'a HashSet<(String, String, String)>,
    min_systematicity: f64,
}

fn backtrack_match(
    var_idx: usize,
    ctx: &MatchContext<'_>,
    current_mapping: &mut HashMap<String, String>,
    used_target_nodes: &mut HashSet<String>,
    results: &mut Vec<CandidateAlignment>,
) {
    if var_idx == ctx.pattern.variables.len() {
        // Complete assignment — evaluate systematicity
        let mut matched_count = 0usize;
        let total_count = ctx.pattern.edges.len();

        for pe in &ctx.pattern.edges {
            if let (Some(src_node), Some(tgt_node)) = (
                current_mapping.get(&pe.source_var),
                current_mapping.get(&pe.target_var),
            ) {
                if ctx.target_edges_set.contains(&(
                    pe.rel_type.clone(),
                    src_node.clone(),
                    tgt_node.clone(),
                )) {
                    matched_count += 1;
                }
            }
        }

        let systematicity = if total_count > 0 {
            matched_count as f64 / total_count as f64
        } else {
            0.0
        };

        if systematicity >= ctx.min_systematicity && matched_count > 0 {
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

    let var_name = &ctx.pattern.variables[var_idx];

    for node in &ctx.target.nodes {
        if !used_target_nodes.contains(node) {
            // Assign var -> node
            current_mapping.insert(var_name.clone(), node.clone());
            used_target_nodes.insert(node.clone());

            backtrack_match(
                var_idx + 1,
                ctx,
                current_mapping,
                used_target_nodes,
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
            nodes: vec![
                "nucleus".to_string(),
                "electron".to_string(),
                "wall".to_string(),
            ],
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

    #[test]
    fn test_empty_inputs_return_empty() {
        let empty_pat = Pattern {
            variables: vec![],
            edges: vec![],
            surface_distractors: vec![],
        };
        let target = TargetGraph {
            nodes: vec!["a".to_string()],
            edges: vec![],
        };
        assert!(match_relational_schema(&empty_pat, &target, 0.3).is_empty());

        let non_empty_pat = Pattern {
            variables: vec!["X".to_string(), "Y".to_string()],
            edges: vec![PatternEdge {
                rel_type: "LINKS".to_string(),
                source_var: "X".to_string(),
                target_var: "Y".to_string(),
            }],
            surface_distractors: vec![],
        };
        let empty_target = TargetGraph {
            nodes: vec![],
            edges: vec![],
        };
        assert!(match_relational_schema(&non_empty_pat, &empty_target, 0.3).is_empty());
    }

    #[test]
    fn test_min_systematicity_filters_partial_matches() {
        let pattern = Pattern {
            variables: vec!["X".to_string(), "Y".to_string()],
            edges: vec![
                PatternEdge {
                    rel_type: "R1".to_string(),
                    source_var: "X".to_string(),
                    target_var: "Y".to_string(),
                },
                PatternEdge {
                    rel_type: "R2".to_string(),
                    source_var: "X".to_string(),
                    target_var: "Y".to_string(),
                },
            ],
            surface_distractors: vec![],
        };
        let target = TargetGraph {
            nodes: vec!["n1".to_string(), "n2".to_string()],
            edges: vec![TargetEdge {
                rel_type: "R1".to_string(),
                source: "n1".to_string(),
                target: "n2".to_string(),
            }],
        };
        assert_eq!(match_relational_schema(&pattern, &target, 0.4).len(), 1);
        assert!(match_relational_schema(&pattern, &target, 0.9).is_empty());
    }

    #[test]
    fn test_three_variable_chain_alignment() {
        let pattern = Pattern {
            variables: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            edges: vec![
                PatternEdge {
                    rel_type: "NEXT".to_string(),
                    source_var: "A".to_string(),
                    target_var: "B".to_string(),
                },
                PatternEdge {
                    rel_type: "NEXT".to_string(),
                    source_var: "B".to_string(),
                    target_var: "C".to_string(),
                },
            ],
            surface_distractors: vec!["ignored".to_string()],
        };
        let target = TargetGraph {
            nodes: vec!["n1".to_string(), "n2".to_string(), "n3".to_string()],
            edges: vec![
                TargetEdge {
                    rel_type: "NEXT".to_string(),
                    source: "n1".to_string(),
                    target: "n2".to_string(),
                },
                TargetEdge {
                    rel_type: "NEXT".to_string(),
                    source: "n2".to_string(),
                    target: "n3".to_string(),
                },
            ],
        };
        let alignments = match_relational_schema(&pattern, &target, 1.0);
        assert_eq!(alignments.len(), 1);
        assert_eq!(alignments[0].mapping.get("A").unwrap(), "n1");
        assert_eq!(alignments[0].mapping.get("B").unwrap(), "n2");
        assert_eq!(alignments[0].mapping.get("C").unwrap(), "n3");
    }

    #[test]
    fn test_unmatchable_relation_returns_empty() {
        let pattern = Pattern {
            variables: vec!["X".to_string(), "Y".to_string()],
            edges: vec![PatternEdge {
                rel_type: "REQUIRES".to_string(),
                source_var: "X".to_string(),
                target_var: "Y".to_string(),
            }],
            surface_distractors: vec![],
        };
        let target = TargetGraph {
            nodes: vec!["x".to_string(), "y".to_string()],
            edges: vec![TargetEdge {
                rel_type: "BLOCKS".to_string(),
                source: "x".to_string(),
                target: "y".to_string(),
            }],
        };
        assert!(match_relational_schema(&pattern, &target, 0.1).is_empty());
    }
}
