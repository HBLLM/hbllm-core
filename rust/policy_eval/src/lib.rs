#![allow(non_local_definitions)]
//! HBLLM Policy Evaluator — fast regex-based policy evaluation.
//!
//! Pre-compiles regex patterns and evaluates text against a batch of policies
//! in a single pass, avoiding Python regex overhead per-policy.

use pyo3::prelude::*;
use regex::Regex;
use std::collections::HashMap;

// ── Core Types ──────────────────────────────────────────────────────────────

struct CompiledPolicy {
    name: String,
    policy_type: String, // "deny", "require", "transform", "scope"
    action: String,      // "block", "warn", "append", "prepend", "replace", "restrict"
    pattern: Option<Regex>,
    content: String,
    severity: String,
    description: String,
    tenant_ids: Vec<String>,
    priority: i32,
    enabled: bool,
    // Conditions: Vec<(key, operator, value_as_string)>
    conditions: Vec<(String, String, String)>,
}

struct EvalResult {
    passed: bool,
    modified_text: String,
    violations: Vec<String>,
    warnings: Vec<String>,
    applied_policies: Vec<String>,
}

// ── Condition Evaluation ────────────────────────────────────────────────────

fn evaluate_condition(
    key: &str,
    op: &str,
    expected: &str,
    context: &HashMap<String, String>,
) -> bool {
    let actual = match context.get(key) {
        Some(v) => v,
        None => return false,
    };

    // Try numeric comparison first
    if let (Ok(a), Ok(e)) = (actual.parse::<f64>(), expected.parse::<f64>()) {
        return match op {
            "eq" => (a - e).abs() < f64::EPSILON,
            "neq" => (a - e).abs() >= f64::EPSILON,
            "gt" => a > e,
            "lt" => a < e,
            "gte" => a >= e,
            "lte" => a <= e,
            _ => false,
        };
    }

    // String comparison
    match op {
        "eq" => actual == expected,
        "neq" => actual != expected,
        "in" => expected.contains(actual.as_str()),
        "not_in" => !expected.contains(actual.as_str()),
        _ => false,
    }
}

// ── Evaluation Engine ───────────────────────────────────────────────────────

fn evaluate_policies(
    text: &str,
    policies: &[CompiledPolicy],
    tenant_id: &str,
    domain: &str,
    context: &HashMap<String, String>,
) -> EvalResult {
    let mut result = EvalResult {
        passed: true,
        modified_text: text.to_string(),
        violations: Vec::new(),
        warnings: Vec::new(),
        applied_policies: Vec::new(),
    };

    for policy in policies {
        if !policy.enabled {
            continue;
        }
        // Tenant check
        if !policy.tenant_ids.contains(&"*".to_string())
            && !policy.tenant_ids.contains(&tenant_id.to_string())
        {
            continue;
        }
        // Conditions check
        let conditions_met = policy
            .conditions
            .iter()
            .all(|(k, op, v)| evaluate_condition(k, op, v, context));
        if !conditions_met {
            continue;
        }

        match policy.policy_type.as_str() {
            "deny" => {
                if let Some(ref re) = policy.pattern {
                    if re.is_match(&result.modified_text) {
                        if policy.action == "block" {
                            result.passed = false;
                            result.violations.push(format!(
                                "[{}] {}: {}",
                                policy.severity.to_uppercase(),
                                policy.name,
                                policy.description
                            ));
                            result.modified_text = format!(
                                "I cannot provide this response due to policy: {}. {}",
                                policy.name, policy.description
                            );
                        } else if policy.action == "warn" {
                            result
                                .warnings
                                .push(format!("{}: {}", policy.name, policy.description));
                        } else if policy.action == "replace" {
                            result.modified_text = re
                                .replace_all(
                                    &result.modified_text,
                                    if policy.content.is_empty() {
                                        "[REDACTED]"
                                    } else {
                                        &policy.content
                                    },
                                )
                                .to_string();
                        }
                        result.applied_policies.push(policy.name.clone());
                    }
                }
            }
            "require" => {
                if let Some(ref re) = policy.pattern {
                    if !re.is_match(&result.modified_text) {
                        if policy.action == "block" {
                            result.passed = false;
                            result.violations.push(format!(
                                "[{}] {}: Missing required element",
                                policy.severity.to_uppercase(),
                                policy.name
                            ));
                        } else if policy.action == "append" {
                            result.modified_text =
                                format!("{}\n\n{}", result.modified_text, policy.content);
                        } else if policy.action == "warn" {
                            result
                                .warnings
                                .push(format!("{}: required element missing", policy.name));
                        }
                        result.applied_policies.push(policy.name.clone());
                    }
                }
            }
            "transform" => {
                if policy.action == "append" {
                    result.modified_text =
                        format!("{}\n\n{}", result.modified_text, policy.content);
                    result.applied_policies.push(policy.name.clone());
                } else if policy.action == "prepend" {
                    result.modified_text =
                        format!("{}\n\n{}", policy.content, result.modified_text);
                    result.applied_policies.push(policy.name.clone());
                } else if policy.action == "replace" {
                    if let Some(ref re) = policy.pattern {
                        let new_text = re.replace_all(&result.modified_text, &*policy.content);
                        if new_text != result.modified_text {
                            result.modified_text = new_text.to_string();
                            result.applied_policies.push(policy.name.clone());
                        }
                    }
                }
            }
            "scope" => {
                if policy.action == "restrict" && !domain.is_empty() {
                    let allowed: Vec<&str> = policy.content.split(',').map(|s| s.trim()).collect();
                    if !allowed.contains(&domain) {
                        result.passed = false;
                        result.violations.push(format!(
                            "[{}] {}: Domain '{}' not allowed",
                            policy.severity.to_uppercase(),
                            policy.name,
                            domain
                        ));
                        result.applied_policies.push(policy.name.clone());
                    }
                }
            }
            _ => {}
        }

        // Stop on first blocking violation
        if !result.passed && policy.action == "block" {
            break;
        }
    }

    result
}

// ── Python Bindings ─────────────────────────────────────────────────────────

/// Compiled policy set for fast repeated evaluation.
#[pyclass]
struct PolicySet {
    policies: Vec<CompiledPolicy>,
}

#[pymethods]
impl PolicySet {
    #[new]
    fn new() -> Self {
        PolicySet {
            policies: Vec::new(),
        }
    }

    /// Add a policy. Pattern is compiled once and reused.
    #[pyo3(signature = (name, policy_type, action, pattern, content, severity, description, tenant_ids, priority, enabled, conditions))]
    fn add_policy(
        &mut self,
        name: String,
        policy_type: String,
        action: String,
        pattern: String,
        content: String,
        severity: String,
        description: String,
        tenant_ids: Vec<String>,
        priority: i32,
        enabled: bool,
        conditions: Vec<(String, String, String)>,
    ) -> PyResult<()> {
        let compiled_pattern = if pattern.is_empty() {
            None
        } else {
            match Regex::new(&format!("(?i){}", pattern)) {
                Ok(re) => Some(re),
                Err(e) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid regex pattern '{}': {}",
                        pattern, e
                    )));
                }
            }
        };

        self.policies.push(CompiledPolicy {
            name,
            policy_type,
            action,
            pattern: compiled_pattern,
            content,
            severity,
            description,
            tenant_ids,
            priority,
            enabled,
            conditions,
        });

        // Re-sort by priority (descending)
        self.policies.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(())
    }

    /// Evaluate text against all policies.
    /// Returns (passed, modified_text, violations, warnings, applied_policies)
    fn evaluate(
        &self,
        text: &str,
        tenant_id: &str,
        domain: &str,
        context: HashMap<String, String>,
    ) -> (bool, String, Vec<String>, Vec<String>, Vec<String>) {
        let result = evaluate_policies(text, &self.policies, tenant_id, domain, &context);
        (
            result.passed,
            result.modified_text,
            result.violations,
            result.warnings,
            result.applied_policies,
        )
    }

    /// Number of loaded policies.
    fn len(&self) -> usize {
        self.policies.len()
    }

    fn clear(&mut self) {
        self.policies.clear();
    }
}

#[pymodule]
fn hbllm_policy_eval(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PolicySet>()?;
    Ok(())
}
