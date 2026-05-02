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
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

static WORD_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b[a-z]{3,}\b").unwrap());

static STOPWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "to", "in", "of",
        "for", "on", "with", "it", "this", "that", "be", "how", "what", "why", "when", "can",
        "do", "does", "did", "will", "would", "should", "could", "have", "has", "had", "not",
        "my", "i",
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
            clusters
                .entry(matched)
                .or_insert_with(Vec::new)
                .push(i);
        }
    }

    let mut result: Vec<(Vec<String>, Vec<usize>)> = clusters.into_iter().collect();
    result.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
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
