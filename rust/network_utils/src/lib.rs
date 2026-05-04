#![allow(non_local_definitions)]
//! HBLLM Network Utils — fast circuit breaker and token bucket rate limiter.
//!
//! These primitives are called on every message through the bus, so even
//! small per-call savings compound at high throughput.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

// ── Circuit Breaker ─────────────────────────────────────────────────────────

#[pyclass]
#[derive(Clone)]
struct CircuitBreaker {
    #[pyo3(get)]
    node_id: String,
    failure_threshold: u32,
    recovery_timeout_secs: f64,
    half_open_max: u32,

    state: u8, // 0=closed, 1=open, 2=half_open
    failure_count: u32,
    success_count: u32,
    last_failure: Option<Instant>,
    half_open_calls: u32,
}

#[pymethods]
impl CircuitBreaker {
    #[new]
    #[pyo3(signature = (node_id, failure_threshold=3, recovery_timeout=30.0, half_open_max=1))]
    fn new(
        node_id: String,
        failure_threshold: u32,
        recovery_timeout: f64,
        half_open_max: u32,
    ) -> Self {
        CircuitBreaker {
            node_id,
            failure_threshold,
            recovery_timeout_secs: recovery_timeout,
            half_open_max,
            state: 0,
            failure_count: 0,
            success_count: 0,
            last_failure: None,
            half_open_calls: 0,
        }
    }

    /// Current state as string: "closed", "open", "half_open".
    #[getter]
    fn state_name(&self) -> &str {
        match self.state {
            0 => "closed",
            1 => "open",
            2 => "half_open",
            _ => "unknown",
        }
    }

    /// Check if request can pass through; triggers auto-transition.
    fn can_execute(&mut self) -> bool {
        // Auto-transition OPEN → HALF_OPEN
        if self.state == 1 {
            if let Some(last) = self.last_failure {
                if last.elapsed().as_secs_f64() >= self.recovery_timeout_secs {
                    self.state = 2;
                    self.half_open_calls = 0;
                }
            }
        }

        match self.state {
            0 => true,
            2 => self.half_open_calls < self.half_open_max,
            _ => false,
        }
    }

    fn record_success(&mut self) {
        if self.state == 2 {
            self.success_count += 1;
            if self.success_count >= self.half_open_max {
                self.state = 0; // CLOSED
                self.failure_count = 0;
                self.success_count = 0;
            }
        } else {
            self.failure_count = 0;
        }
    }

    fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure = Some(Instant::now());

        if self.state == 2 || self.failure_count >= self.failure_threshold {
            self.state = 1; // OPEN
        }
    }

    fn reset(&mut self) {
        self.state = 0;
        self.failure_count = 0;
        self.success_count = 0;
        self.half_open_calls = 0;
    }

    /// Seconds until retry is allowed (0 if not OPEN).
    fn time_until_retry(&self) -> f64 {
        if self.state != 1 {
            return 0.0;
        }
        match self.last_failure {
            Some(last) => {
                (self.recovery_timeout_secs - last.elapsed().as_secs_f64()).max(0.0)
            }
            None => 0.0,
        }
    }
}

// ── Rate Limiter (Token Bucket) ─────────────────────────────────────────────

#[pyclass]
struct RateLimiter {
    target_rpm: f64,
    burst_limit: f64,
    buckets: HashMap<String, (f64, Instant)>, // tenant → (tokens, last_refill)
}

#[pymethods]
impl RateLimiter {
    #[new]
    #[pyo3(signature = (target_rpm=60.0, burst_multiplier=1.5))]
    fn new(target_rpm: f64, burst_multiplier: f64) -> Self {
        RateLimiter {
            target_rpm,
            burst_limit: target_rpm * burst_multiplier,
            buckets: HashMap::new(),
        }
    }

    /// Check if a request from tenant_id should be allowed.
    /// Returns true if allowed, false if rate-limited.
    fn allow(&mut self, tenant_id: &str) -> bool {
        if tenant_id.is_empty() || tenant_id == "system" {
            return true;
        }

        let now = Instant::now();

        let (tokens, last_refill) = self
            .buckets
            .entry(tenant_id.to_string())
            .or_insert((self.burst_limit, now));

        // Refill tokens
        let elapsed = last_refill.elapsed().as_secs_f64();
        let refill = elapsed * (self.target_rpm / 60.0);
        *tokens = (*tokens + refill).min(self.burst_limit);
        *last_refill = now;

        if *tokens >= 1.0 {
            *tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Reset all buckets.
    fn reset(&mut self) {
        self.buckets.clear();
    }
}

// ── Python Module ───────────────────────────────────────────────────────────

#[pymodule]
fn hbllm_network_utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CircuitBreaker>()?;
    m.add_class::<RateLimiter>()?;
    Ok(())
}
