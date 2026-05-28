#![allow(non_local_definitions)]
//! HBLLM Concept Extractor — fast keyword extraction and topic clustering.
//!
//! Provides:
//! - Batch keyword extraction with stopword filtering
//! - Keyword frequency counting across documents
//! - Co-occurrence clustering
//! - Rule extraction from query patterns

use pyo3::prelude::*;
use regex::Regex;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

static WORD_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b[a-z]{3,}\b").unwrap());

static STOPWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "to", "in", "of", "for",
        "on", "with", "it", "this", "that", "be", "how", "what", "why", "when", "can", "do",
        "does", "did", "will", "would", "should", "could", "have", "has", "had", "not", "my", "i",
    ]
    .into_iter()
    .collect()
});

// ── Keyword Extraction ──────────────────────────────────────────────────────

fn extract_keywords(text: &str) -> HashSet<String> {
    WORD_RE
        .find_iter(&text.to_lowercase())
        .map(|m| m.as_str().to_string())
        .filter(|w| !STOPWORDS.contains(w.as_str()))
        .collect()
}

fn count_keywords_batch(texts: &[String]) -> HashMap<String, usize> {
    let mut freq: HashMap<String, usize> = HashMap::new();
    for text in texts {
        let unique = extract_keywords(text);
        for kw in unique {
            *freq.entry(kw).or_insert(0) += 1;
        }
    }
    freq
}

// ── Clustering ──────────────────────────────────────────────────────────────

/// Cluster queries by co-occurring significant keywords.
/// Returns Vec<(keywords, matching_query_indices)>.
fn cluster_queries(
    queries: &[String],
    min_frequency: usize,
    min_keyword_count: usize,
) -> Vec<(Vec<String>, Vec<usize>)> {
    let keyword_freq = count_keywords_batch(queries);

    // Significant keywords
    let significant: HashSet<&String> = keyword_freq
        .iter()
        .filter(|(_, &v)| v >= min_frequency)
        .map(|(k, _)| k)
        .collect();

    // Group queries by their significant keyword sets
    let mut clusters: HashMap<Vec<String>, Vec<usize>> = HashMap::new();

    for (i, query) in queries.iter().enumerate() {
        let words = extract_keywords(query);
        let mut matched: Vec<String> = words
            .iter()
            .filter(|w| significant.contains(w))
            .cloned()
            .collect();

        if matched.len() >= min_keyword_count {
            matched.sort();
            matched.truncate(5);
            clusters.entry(matched).or_default().push(i);
        }
    }

    let mut result: Vec<(Vec<String>, Vec<usize>)> = clusters.into_iter().collect();
    result.sort_by_key(|item| Reverse(item.1.len()));
    result
}

// ── Rule Extraction ─────────────────────────────────────────────────────────

fn extract_rules(queries: &[String]) -> Vec<String> {
    let mut rules = Vec::new();
    let len = queries.len();
    if len == 0 {
        return rules;
    }

    let question_count = queries.iter().filter(|q| q.contains('?')).count();
    if question_count as f64 > len as f64 * 0.7 {
        rules.push("Users frequently ask questions about this topic".to_string());
    }

    let howto_count = queries
        .iter()
        .filter(|q| q.to_lowercase().contains("how"))
        .count();
    if howto_count as f64 > len as f64 * 0.3 {
        rules.push("Topic requires procedural/how-to knowledge".to_string());
    }

    let error_words = ["error", "fix", "bug", "issue", "problem"];
    let error_count = queries
        .iter()
        .filter(|q| {
            let lower = q.to_lowercase();
            error_words.iter().any(|w| lower.contains(w))
        })
        .count();
    if error_count as f64 > len as f64 * 0.3 {
        rules.push("Topic is frequently related to troubleshooting".to_string());
    }

    rules
}

// ── Python Bindings ─────────────────────────────────────────────────────────

/// Count keyword frequencies across a batch of texts.
/// Returns dict[str, int] of keyword → frequency.
#[pyfunction]
fn batch_keyword_count(texts: Vec<String>) -> HashMap<String, usize> {
    count_keywords_batch(&texts)
}

/// Cluster queries by co-occurring keywords.
/// Returns list of (keywords, query_indices) tuples.
#[pyfunction]
#[pyo3(signature = (queries, min_frequency=3, min_keyword_count=2))]
fn cluster_by_keywords(
    queries: Vec<String>,
    min_frequency: usize,
    min_keyword_count: usize,
) -> Vec<(Vec<String>, Vec<usize>)> {
    cluster_queries(&queries, min_frequency, min_keyword_count)
}

/// Extract generalized rules from a set of queries.
#[pyfunction]
fn extract_query_rules(queries: Vec<String>) -> Vec<String> {
    extract_rules(&queries)
}

/// Extract keywords from a single text (with stopword removal).
#[pyfunction]
fn extract_text_keywords(text: &str) -> Vec<String> {
    extract_keywords(text).into_iter().collect()
}

#[pymodule]
fn hbllm_concept_extract(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_keyword_count, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_by_keywords, m)?)?;
    m.add_function(wrap_pyfunction!(extract_query_rules, m)?)?;
    m.add_function(wrap_pyfunction!(extract_text_keywords, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_keywords_filters_stopwords() {
        let kw = extract_keywords("the cat and the dog");
        assert!(kw.contains("cat"));
        assert!(kw.contains("dog"));
        assert!(!kw.contains("the"));
        assert!(!kw.contains("and"));
    }

    #[test]
    fn test_extract_keywords_lowercase() {
        let kw = extract_keywords("Rust Programming Language");
        assert!(kw.contains("rust"));
        assert!(kw.contains("programming"));
        assert!(kw.contains("language"));
    }

    #[test]
    fn test_extract_keywords_short_words_filtered() {
        let kw = extract_keywords("go is ok");
        // "go", "is", "ok" are all <=2 chars or stopwords → all filtered by regex [a-z]{3,}
        assert!(kw.is_empty());
    }

    #[test]
    fn test_count_keywords_batch() {
        let texts = vec![
            "rust programming".to_string(),
            "rust performance".to_string(),
            "python programming".to_string(),
        ];
        let freq = count_keywords_batch(&texts);
        assert_eq!(freq.get("rust"), Some(&2));
        assert_eq!(freq.get("programming"), Some(&2));
        assert_eq!(freq.get("performance"), Some(&1));
        assert_eq!(freq.get("python"), Some(&1));
    }

    #[test]
    fn test_cluster_queries_groups() {
        let queries: Vec<String> = (0..5)
            .map(|_| "rust async programming".to_string())
            .chain((0..5).map(|_| "python web framework".to_string()))
            .collect();
        let clusters = cluster_queries(&queries, 3, 2);
        assert!(!clusters.is_empty());
    }

    #[test]
    fn test_cluster_queries_min_frequency() {
        let queries = vec![
            "unique rare keyword".to_string(),
            "another unique query".to_string(),
        ];
        // min_frequency=5 means no keyword appears enough
        let clusters = cluster_queries(&queries, 5, 1);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_extract_rules_questions() {
        let queries: Vec<String> = (0..10)
            .map(|i| format!("how do I fix issue {}?", i))
            .collect();
        let rules = extract_rules(&queries);
        assert!(rules.iter().any(|r| r.contains("questions")));
    }

    #[test]
    fn test_extract_rules_howto() {
        let queries: Vec<String> = (0..5)
            .map(|i| format!("how to configure setting {}", i))
            .chain(std::iter::once("what is rust?".to_string()))
            .collect();
        let rules = extract_rules(&queries);
        assert!(rules.iter().any(|r| r.contains("procedural")));
    }

    #[test]
    fn test_extract_rules_errors() {
        let queries: Vec<String> = (0..5)
            .map(|i| format!("fix error {} in production", i))
            .collect();
        let rules = extract_rules(&queries);
        assert!(rules.iter().any(|r| r.contains("troubleshooting")));
    }

    #[test]
    fn test_extract_rules_empty() {
        let rules = extract_rules(&[]);
        assert!(rules.is_empty());
    }
}
