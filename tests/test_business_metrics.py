import pytest

from hbllm.network.business_metrics import BusinessMetrics


def test_record_inference_cost():
    metrics = BusinessMetrics()
    metrics.record_inference_cost("tenant_1", 1000, "gpt-4")

    dashboard = metrics.get_tenant_dashboard("tenant_1")
    assert dashboard["total_tokens"] == 1000
    assert dashboard["total_cost_usd"] == 0.03  # 1k tokens @ 0.03
    assert dashboard["api_calls"] == 1

    metrics.record_inference_cost("tenant_1", 500, "gpt-3.5")
    dashboard = metrics.get_tenant_dashboard("tenant_1")
    assert dashboard["total_tokens"] == 1500
    assert dashboard["total_cost_usd"] == 0.03 + 0.001
    assert dashboard["api_calls"] == 2


def test_record_satisfaction():
    metrics = BusinessMetrics()
    metrics.record_satisfaction("tenant_1", 1.0)
    metrics.record_satisfaction("tenant_1", 0.0)

    dashboard = metrics.get_tenant_dashboard("tenant_1")
    assert dashboard["average_satisfaction"] == 0.5


def test_record_response_quality():
    metrics = BusinessMetrics()
    metrics.record_response_quality("tenant_1", "good")  # 1.0
    metrics.record_response_quality("tenant_1", "neutral")  # 0.5
    metrics.record_response_quality("tenant_1", "bad")  # 0.0

    dashboard = metrics.get_tenant_dashboard("tenant_1")
    assert dashboard["average_satisfaction"] == 0.5


def test_unknown_tenant():
    metrics = BusinessMetrics()
    assert metrics.get_tenant_dashboard("unknown") == {}
