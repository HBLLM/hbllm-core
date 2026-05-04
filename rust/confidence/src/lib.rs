#![allow(non_local_definitions)]
//! HBLLM Confidence Estimator — fast text analysis for response quality scoring.
//!
//! Ports the per-response scoring pipeline from Python to Rust:
//! - Lexical relevance (query-response overlap)
//! - Coherence (sentence structure consistency)
//! - Factuality risk (hallucination indicators)
//! - Uncertainty detection (hedge language)
//! - Detail level scoring

use pyo3::prelude::*;
use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

// ── Compiled Regex Patterns ─────────────────────────────────────────────────

static HEDGE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(i think|maybe|perhaps|possibly|might be|not sure|could be|i believe|it seems|approximately|roughly|around)\b"
    ).unwrap()
});

static DEFINITIVE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(definitely|certainly|always|never|absolutely|exactly|proven|guaranteed|100%)\b"
    ).unwrap()
});

static FACTUAL_CLAIM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(\d{4}[-/]\d{2}[-/]\d{2}|\d+\.\d+%|\$\d+|founded in \d{4}|born in \d{4}|died in \d{4})\b"
    ).unwrap()
});

static LONG_NUMERIC_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b\d{6,}\b").unwrap()
});

static WORD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b\w+\b").unwrap()
});

// ── Stopwords ───────────────────────────────────────────────────────────────

static STOPWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "to",
        "in", "of", "for", "on", "with", "it", "this", "that",
    ]
    .into_iter()
    .collect()
});

// ── Scoring Functions ───────────────────────────────────────────────────────

fn extract_words(text: &str) -> HashSet<String> {
    WORD_RE
        .find_iter(&text.to_lowercase())
        .map(|m| m.as_str().to_string())
        .filter(|w| !STOPWORDS.contains(w.as_str()))
        .collect()
}

fn score_relevance(query: &str, response: &str) -> f64 {
    let q_words = extract_words(query);
    let r_words = extract_words(response);

    if q_words.is_empty() {
        return 0.5;
    }

    let overlap = q_words.intersection(&r_words).count() as f64;
    (overlap / q_words.len() as f64 * 1.5).min(1.0)
}

fn score_coherence(response: &str) -> f64 {
    let sentences: Vec<&str> = response
        .split(['.', '!', '?'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if sentences.len() <= 1 {
        return 0.6;
    }

    // Repetition check
    let unique: HashSet<String> = sentences.iter().map(|s| s.to_lowercase()).collect();
    let repetition_ratio = unique.len() as f64 / sentences.len() as f64;

    // Sentence length consistency (coefficient of variation)
    let lengths: Vec<f64> = sentences
        .iter()
        .map(|s| s.split_whitespace().count() as f64)
        .collect();

    let mean = lengths.iter().sum::<f64>() / lengths.len() as f64;
    let variance = lengths.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / lengths.len() as f64;
    let cv = variance.sqrt() / mean.max(1.0);
    let length_consistency = (1.0 - cv * 0.5).max(0.0);

    repetition_ratio * 0.6 + length_consistency * 0.4
}

fn score_factuality_risk(response: &str) -> f64 {
    let mut risk = 0.0;

    let claims = FACTUAL_CLAIM_RE.find_iter(response).count();
    if claims > 0 {
        risk += (claims as f64 * 0.15).min(0.5);
    }

    let definitive = DEFINITIVE_RE.find_iter(response).count();
    if definitive > 0 {
        risk += (definitive as f64 * 0.1).min(0.3);
    }

    if LONG_NUMERIC_RE.is_match(response) {
        risk += 0.2;
    }

    risk.min(1.0)
}

fn score_uncertainty(response: &str) -> f64 {
    let hedges = HEDGE_RE.find_iter(response).count();
    let word_count = response.split_whitespace().count();
    if word_count == 0 {
        return 0.5;
    }
    (hedges as f64 / (word_count as f64 / 20.0).max(1.0)).min(1.0)
}

fn score_detail(response: &str) -> f64 {
    let word_count = response.split_whitespace().count();
    if word_count < 5 {
        0.1
    } else if word_count < 20 {
        0.4
    } else if word_count < 100 {
        0.7
    } else {
        0.9
    }
}

// ── Python Bindings ─────────────────────────────────────────────────────────

/// Full confidence estimation returning (overall, relevance, coherence, factuality_risk, uncertainty, detail, flags)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn estimate_confidence(
    query: &str,
    response: &str,
    hallucination_threshold: f64,
    w_relevance: f64,
    w_coherence: f64,
    w_factuality: f64,
    w_uncertainty: f64,
    w_detail: f64,
) -> (f64, f64, f64, f64, f64, f64, Vec<String>) {
    let relevance = score_relevance(query, response);
    let coherence = score_coherence(response);
    let factuality_risk = score_factuality_risk(response);
    let uncertainty = score_uncertainty(response);
    let detail = score_detail(response);

    let mut overall = w_relevance * relevance
        + w_coherence * coherence
        + w_factuality * (1.0 - factuality_risk)
        + w_uncertainty * (1.0 - uncertainty)
        + w_detail * detail;

    let mut flags = Vec::new();

    if factuality_risk > hallucination_threshold {
        flags.push("high_hallucination_risk".to_string());
        overall *= 0.8;
    }
    if uncertainty > 0.6 {
        flags.push("high_uncertainty".to_string());
    }
    if relevance < 0.3 {
        flags.push("low_relevance".to_string());
        overall *= 0.8;
    }
    if response.split_whitespace().count() < 5 {
        flags.push("too_brief".to_string());
        overall *= 0.8;
    }

    overall = overall.clamp(0.0, 1.0);

    (
        (overall * 1000.0).round() / 1000.0,
        (relevance * 1000.0).round() / 1000.0,
        (coherence * 1000.0).round() / 1000.0,
        (factuality_risk * 1000.0).round() / 1000.0,
        (uncertainty * 1000.0).round() / 1000.0,
        (detail * 1000.0).round() / 1000.0,
        flags,
    )
}

/// Quick confidence score (just the overall number).
#[pyfunction]
fn quick_score(query: &str, response: &str) -> f64 {
    let (overall, ..) = estimate_confidence(query, response, 0.6, 0.25, 0.20, 0.25, 0.15, 0.15);
    overall
}

#[pymodule]
fn hbllm_confidence(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate_confidence, m)?)?;
    m.add_function(wrap_pyfunction!(quick_score, m)?)?;
    Ok(())
}
