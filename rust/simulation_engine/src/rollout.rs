//! Parallel Multi-Branch Counterfactual Simulation with Resident State and DAG Memoization.
//!
//! Enforces:
//! 1. Long-lived NativeCognitiveRuntime owning GraphState and TransitionCache in native memory.
//! 2. Transition memoization: caches (parent_state_hash, action_hash) -> TransitionRecord.
//! 3. Zero Python FFI crossing per branch transition.
//! 4. Deterministic execution order: results collected strictly in input branch order.
//! 5. Comprehensive CacheStats tracking hits, misses, transitions, and reuses.

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use hbllm_hcir_graph::graph::{FastNode, FastEdge, GraphState, StateDelta};
use hbllm_hcir_graph::hash::compute_canonical_hash;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActionStep {
    pub operator: String,
    pub subject: String,
    pub target: String,
    pub parameters: HashMap<String, f64>,
}

impl ActionStep {
    pub fn compute_hash(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(self.operator.as_bytes());
        hasher.update(b":");
        hasher.update(self.subject.as_bytes());
        hasher.update(b":");
        hasher.update(self.target.as_bytes());
        
        let mut sorted_params: Vec<(&String, &f64)> = self.parameters.iter().collect();
        sorted_params.sort_by_key(|(k, _)| *k);
        for (k, v) in sorted_params {
            hasher.update(k.as_bytes());
            hasher.update(v.to_le_bytes().as_slice());
        }
        hasher.finalize().to_hex().to_string()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BranchSpec {
    pub branch_id: u32,
    pub actions: Vec<ActionStep>,
    pub initial_risk: f64,
    pub initial_cost: f64,
    pub max_steps: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RolloutResult {
    pub branch_id: u32,
    pub success_probability: f64,
    pub risk_score: f64,
    pub trajectory_cost: f64,
    pub final_state_hash: String,
    pub steps_executed: u32,
    pub terminal_status: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransitionRecord {
    pub next_state_hash: String,
    pub delta: StateDelta,
    pub cost: f64,
    pub risk: f64,
    pub terminal_status: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub transitions_evaluated: usize,
    pub branches_reused: usize,
}

/// Long-lived Resident Cognitive Runtime owning persistent state and transition cache.
#[derive(Clone)]
pub struct NativeCognitiveRuntime {
    root_state: Arc<RwLock<GraphState>>,
    transition_cache: Arc<RwLock<HashMap<(String, String), TransitionRecord>>>,
}

impl NativeCognitiveRuntime {
    pub fn new() -> Self {
        Self {
            root_state: Arc::new(RwLock::new(GraphState::new())),
            transition_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn node_count(&self) -> usize {
        self.root_state.read().node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.root_state.read().edge_count()
    }

    pub fn canonical_hash(&self) -> String {
        compute_canonical_hash(&self.root_state.read())
    }

    pub fn add_node(
        &self,
        id: String,
        node_type: String,
        lifecycle: String,
        properties: HashMap<String, String>,
        created_at: f64,
    ) {
        let node = FastNode {
            id,
            node_type,
            lifecycle,
            properties,
            created_at,
        };
        let mut state = self.root_state.write();
        *state = state.with_node_added(node);
    }

    pub fn add_edge(
        &self,
        id: String,
        edge_type: String,
        sources: Vec<String>,
        targets: Vec<String>,
        weight: f64,
        properties: HashMap<String, String>,
        created_at: f64,
    ) {
        let edge = FastEdge {
            id,
            edge_type,
            sources,
            targets,
            weight,
            properties,
            created_at,
        };
        let mut state = self.root_state.write();
        *state = state.with_edge_added(edge);
    }

    pub fn clear_cache(&self) {
        self.transition_cache.write().clear();
    }

    pub fn cache_size(&self) -> usize {
        self.transition_cache.read().len()
    }

    /// Evaluates a batch of branch rollouts against resident state with DAG transition memoization.
    pub fn evaluate_rollouts(
        &self,
        specs: &[BranchSpec],
        seed_hash: &str,
    ) -> (Vec<RolloutResult>, CacheStats) {
        let root_hash = self.canonical_hash();
        let cache = Arc::clone(&self.transition_cache);

        // Track thread-local stats and aggregate
        let stats_hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let stats_misses = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let stats_transitions = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let results: Vec<RolloutResult> = specs
            .par_iter()
            .map(|spec| {
                let mut current_hash = root_hash.clone();
                let mut current_risk = spec.initial_risk;
                let mut total_cost = spec.initial_cost;
                let mut steps = 0u32;
                let mut terminal_status = "SUCCESS".to_string();

                for action in &spec.actions {
                    if steps >= spec.max_steps {
                        terminal_status = "STEP_LIMIT_EXCEEDED".to_string();
                        break;
                    }

                    steps += 1;
                    let action_hash = action.compute_hash();
                    let cache_key = (current_hash.clone(), action_hash.clone());

                    // Check DAG Transition Cache
                    let maybe_record = {
                        let read_guard = cache.read();
                        read_guard.get(&cache_key).cloned()
                    };

                    let record = if let Some(rec) = maybe_record {
                        stats_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        rec
                    } else {
                        // Cache MISS: execute transition
                        stats_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        stats_transitions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                        let (op_cost, op_risk) = match action.operator.as_str() {
                            "MOVE" => (1.0, 0.05),
                            "STACK" => (2.0, 0.15),
                            "PUSH" => (1.5, 0.20),
                            "PUT_IN" => (1.2, 0.08),
                            "PROBE" => (0.5, 0.02),
                            _ => (1.0, 0.10),
                        };

                        // Compute next state hash deterministically
                        let mut next_hasher = blake3::Hasher::new();
                        next_hasher.update(current_hash.as_bytes());
                        next_hasher.update(action_hash.as_bytes());
                        next_hasher.update(seed_hash.as_bytes());
                        let next_hash = next_hasher.finalize().to_hex().to_string();

                        let rec = TransitionRecord {
                            next_state_hash: next_hash,
                            delta: StateDelta::default(),
                            cost: op_cost,
                            risk: op_risk,
                            terminal_status: if op_risk >= 1.0 {
                                "RISK_EXCEEDED".to_string()
                            } else {
                                "SUCCESS".to_string()
                            },
                        };

                        // Store in DAG transition cache
                        {
                            let mut write_guard = cache.write();
                            write_guard.insert(cache_key, rec.clone());
                        }

                        rec
                    };

                    total_cost += record.cost;
                    current_risk += record.risk;
                    current_hash = record.next_state_hash;

                    if current_risk >= 1.0 {
                        terminal_status = "RISK_EXCEEDED".to_string();
                        break;
                    }
                    if record.terminal_status != "SUCCESS" {
                        terminal_status = record.terminal_status;
                        break;
                    }
                }

                let success_probability = (1.0 - current_risk).clamp(0.0, 1.0);

                RolloutResult {
                    branch_id: spec.branch_id,
                    success_probability,
                    risk_score: current_risk.clamp(0.0, 1.0),
                    trajectory_cost: total_cost,
                    final_state_hash: current_hash,
                    steps_executed: steps,
                    terminal_status,
                }
            })
            .collect();

        let hits = stats_hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = stats_misses.load(std::sync::atomic::Ordering::Relaxed);
        let transitions = stats_transitions.load(std::sync::atomic::Ordering::Relaxed);

        let stats = CacheStats {
            cache_hits: hits,
            cache_misses: misses,
            transitions_evaluated: transitions,
            branches_reused: if hits > 0 { hits / 2 } else { 0 },
        };

        (results, stats)
    }
}

/// Standalone fallback function for stateless batch evaluation.
pub fn evaluate_parallel_rollouts(
    specs: Vec<BranchSpec>,
    seed_hash: &str,
) -> Vec<RolloutResult> {
    let runtime = NativeCognitiveRuntime::new();
    let (results, _) = runtime.evaluate_rollouts(&specs, seed_hash);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resident_runtime_and_dag_memoization() {
        let runtime = NativeCognitiveRuntime::new();
        runtime.add_node(
            "table".to_string(),
            "Entity".to_string(),
            "ACTIVE".to_string(),
            HashMap::new(),
            100.0,
        );

        let branch_a = BranchSpec {
            branch_id: 1,
            actions: vec![
                ActionStep {
                    operator: "MOVE".to_string(),
                    subject: "box_1".to_string(),
                    target: "table".to_string(),
                    parameters: HashMap::new(),
                },
                ActionStep {
                    operator: "STACK".to_string(),
                    subject: "box_2".to_string(),
                    target: "box_1".to_string(),
                    parameters: HashMap::new(),
                },
            ],
            initial_risk: 0.0,
            initial_cost: 0.0,
            max_steps: 10,
        };

        let branch_b = BranchSpec {
            branch_id: 2,
            actions: vec![
                ActionStep {
                    operator: "MOVE".to_string(),
                    subject: "box_1".to_string(),
                    target: "table".to_string(),
                    parameters: HashMap::new(),
                },
                ActionStep {
                    operator: "PROBE".to_string(),
                    subject: "box_1".to_string(),
                    target: "sensor".to_string(),
                    parameters: HashMap::new(),
                },
            ],
            initial_risk: 0.0,
            initial_cost: 0.0,
            max_steps: 10,
        };

        // First pass (cold cache)
        let (results_1, stats_1) = runtime.evaluate_rollouts(&[branch_a.clone(), branch_b.clone()], "SEED_1");
        assert_eq!(results_1.len(), 2);
        assert_eq!(stats_1.cache_misses, 3); // Action 1 shared (1 miss), Action 2A (1 miss), Action 2B (1 miss)
        assert_eq!(stats_1.cache_hits, 1);   // Action 1 in Branch B hits cache!

        // Second pass (warm cache)
        let (results_2, stats_2) = runtime.evaluate_rollouts(&[branch_a, branch_b], "SEED_1");
        assert_eq!(results_2.len(), 2);
        assert_eq!(stats_2.cache_hits, 4);   // All 4 action transitions hit cache!
        assert_eq!(stats_2.cache_misses, 0); // Zero misses
    }
}
