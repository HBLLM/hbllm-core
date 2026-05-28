#![allow(non_local_definitions)]
//! HBLLM Policy Evaluator — fast regex-based policy evaluation.
//!
//! Pre-compiles regex patterns and evaluates text against a batch of policies
//! in a single pass, avoiding Python regex overhead per-policy.

use pyo3::prelude::*;
use regex::Regex;
use std::cmp::Reverse;
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
            "scope" if policy.action == "restrict" && !domain.is_empty() => {
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
    #[allow(clippy::too_many_arguments)]
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
        self.policies.sort_by_key(|p| Reverse(p.priority));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_policy(
        name: &str,
        policy_type: &str,
        action: &str,
        pattern: &str,
        content: &str,
        severity: &str,
    ) -> CompiledPolicy {
        CompiledPolicy {
            name: name.to_string(),
            policy_type: policy_type.to_string(),
            action: action.to_string(),
            pattern: if pattern.is_empty() {
                None
            } else {
                Some(Regex::new(&format!("(?i){}", pattern)).unwrap())
            },
            content: content.to_string(),
            severity: severity.to_string(),
            description: format!("Test policy: {}", name),
            tenant_ids: vec!["*".to_string()],
            priority: 0,
            enabled: true,
            conditions: Vec::new(),
        }
    }

    #[test]
    fn test_deny_block_matching() {
        let policies = vec![make_policy(
            "no_secrets",
            "deny",
            "block",
            r"password\s*=",
            "",
            "critical",
        )];
        let ctx = HashMap::new();
        let result = evaluate_policies("my password = hunter2", &policies, "t1", "", &ctx);
        assert!(!result.passed);
        assert!(!result.violations.is_empty());
    }

    #[test]
    fn test_deny_warn_matching() {
        let policies = vec![make_policy("warn_todo", "deny", "warn", r"TODO", "", "low")];
        let ctx = HashMap::new();
        let result = evaluate_policies("There is a TODO here", &policies, "t1", "", &ctx);
        assert!(result.passed); // warn doesn't block
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_deny_replace() {
        let policies = vec![make_policy(
            "redact_email",
            "deny",
            "replace",
            r"\S+@\S+\.\S+",
            "[EMAIL]",
            "medium",
        )];
        let ctx = HashMap::new();
        let result = evaluate_policies(
            "Contact me at user@example.com please",
            &policies,
            "t1",
            "",
            &ctx,
        );
        assert!(result.passed);
        assert!(result.modified_text.contains("[EMAIL]"));
        assert!(!result.modified_text.contains("user@example.com"));
    }

    #[test]
    fn test_require_missing_blocks() {
        let policies = vec![make_policy(
            "require_disclaimer",
            "require",
            "block",
            r"DISCLAIMER",
            "",
            "high",
        )];
        let ctx = HashMap::new();
        let result = evaluate_policies(
            "Here is my response to your question",
            &policies,
            "t1",
            "",
            &ctx,
        );
        assert!(!result.passed);
    }

    #[test]
    fn test_require_present_passes() {
        let policies = vec![make_policy(
            "require_disclaimer",
            "require",
            "block",
            r"DISCLAIMER",
            "",
            "high",
        )];
        let ctx = HashMap::new();
        let result = evaluate_policies(
            "DISCLAIMER: This is informational only.",
            &policies,
            "t1",
            "",
            &ctx,
        );
        assert!(result.passed);
    }

    #[test]
    fn test_require_append() {
        let policies = vec![make_policy(
            "add_footer",
            "require",
            "append",
            r"FOOTER",
            "\n---\nGenerated by HBLLM",
            "low",
        )];
        let ctx = HashMap::new();
        let result = evaluate_policies("Response text here", &policies, "t1", "", &ctx);
        assert!(result.passed);
        assert!(result.modified_text.contains("Generated by HBLLM"));
    }

    #[test]
    fn test_transform_prepend() {
        let policies = vec![make_policy(
            "add_header",
            "transform",
            "prepend",
            "",
            "## Header",
            "low",
        )];
        let ctx = HashMap::new();
        let result = evaluate_policies("Body text", &policies, "t1", "", &ctx);
        assert!(result.modified_text.starts_with("## Header"));
    }

    #[test]
    fn test_scope_restrict_allowed() {
        let mut policy = make_policy(
            "scope_math",
            "scope",
            "restrict",
            "",
            "math,science",
            "high",
        );
        policy.tenant_ids = vec!["*".to_string()];
        let ctx = HashMap::new();
        let result = evaluate_policies("Calculate 2+2", &[policy], "t1", "math", &ctx);
        assert!(result.passed);
    }

    #[test]
    fn test_scope_restrict_denied() {
        let policy = make_policy(
            "scope_math",
            "scope",
            "restrict",
            "",
            "math,science",
            "high",
        );
        let ctx = HashMap::new();
        let result = evaluate_policies("Write a poem", &[policy], "t1", "creative", &ctx);
        assert!(!result.passed);
    }

    #[test]
    fn test_condition_numeric_gte() {
        let mut policy = make_policy("high_risk", "deny", "block", r"execute", "", "critical");
        policy.conditions = vec![(
            "risk_score".to_string(),
            "gte".to_string(),
            "0.8".to_string(),
        )];
        let mut ctx = HashMap::new();
        ctx.insert("risk_score".to_string(), "0.9".to_string());
        let result = evaluate_policies("execute command", &[policy], "t1", "", &ctx);
        assert!(!result.passed);
    }

    #[test]
    fn test_condition_not_met_skips() {
        let mut policy = make_policy("high_risk", "deny", "block", r"execute", "", "critical");
        policy.conditions = vec![(
            "risk_score".to_string(),
            "gte".to_string(),
            "0.8".to_string(),
        )];
        let mut ctx = HashMap::new();
        ctx.insert("risk_score".to_string(), "0.3".to_string());
        let result = evaluate_policies("execute command", &[policy], "t1", "", &ctx);
        assert!(result.passed); // Condition not met → policy skipped
    }

    #[test]
    fn test_tenant_filtering() {
        let mut policy = make_policy("tenant_only", "deny", "block", r"secret", "", "high");
        policy.tenant_ids = vec!["tenant_a".to_string()];
        let ctx = HashMap::new();
        // Policy should NOT apply to tenant_b
        let result = evaluate_policies("this is a secret", &[policy], "tenant_b", "", &ctx);
        assert!(result.passed);
    }

    #[test]
    fn test_disabled_policy_skipped() {
        let mut policy = make_policy("disabled", "deny", "block", r".*", "", "critical");
        policy.enabled = false;
        let ctx = HashMap::new();
        let result = evaluate_policies("anything", &[policy], "t1", "", &ctx);
        assert!(result.passed);
    }
}
