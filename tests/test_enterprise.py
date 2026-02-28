"""Tests for MetricsCollector and PluginManager."""

import pytest
import tempfile
import os
from pathlib import Path

from hbllm.network.metrics import MetricsCollector
from hbllm.network.plugin_manager import PluginManager, PluginInfo


# ─── MetricsCollector Tests ──────────────────────────────────────────────────

class TestMetricsCollector:
    def setup_method(self):
        MetricsCollector.reset()

    def test_singleton(self):
        m1 = MetricsCollector.get_instance()
        m2 = MetricsCollector.get_instance()
        assert m1 is m2

    def test_backend(self):
        m = MetricsCollector.get_instance()
        assert m.backend in ("prometheus", "inmemory")

    def test_record_request(self):
        m = MetricsCollector.get_instance()
        m.record_request("router.query", tenant_id="t1")
        m.record_request("router.query", tenant_id="t1", status="error")
        # Should not raise

    def test_record_message(self):
        m = MetricsCollector.get_instance()
        m.record_message("workspace.update", "event")
        # Should not raise

    def test_record_error(self):
        m = MetricsCollector.get_instance()
        m.record_error("router_01", "timeout")
        # Should not raise

    def test_observe_duration(self):
        m = MetricsCollector.get_instance()
        m.observe_duration("pipeline", 1.5)
        m.observe_duration("pipeline", 0.3)

    def test_observe_node_latency(self):
        m = MetricsCollector.get_instance()
        m.observe_node_latency("router_01", 0.05)

    def test_gauges(self):
        m = MetricsCollector.get_instance()
        m.set_active_nodes(10)
        m.set_healthy_nodes(9)
        m.inc_active_requests()
        m.inc_active_requests()
        m.dec_active_requests()

    def test_measure_latency_context(self):
        m = MetricsCollector.get_instance()
        with m.measure_latency("test_stage"):
            x = sum(range(100))
        # Should record a duration

    def test_get_metrics_text(self):
        m = MetricsCollector.get_instance()
        m.record_request("test")
        text = m.get_metrics_text()
        assert isinstance(text, str)

    def test_snapshot(self):
        m = MetricsCollector.get_instance()
        m.record_request("test")
        snap = m.snapshot()
        assert "backend" in snap


# ─── PluginInfo Tests ────────────────────────────────────────────────────────

class TestPluginInfo:
    def test_to_dict(self):
        info = PluginInfo(
            name="test_plugin",
            path="/tmp/test.py",
            version="1.0.0",
            description="A test plugin",
        )
        d = info.to_dict()
        assert d["name"] == "test_plugin"
        assert d["loaded"] is False
        assert d["error"] is None


# ─── PluginManager Tests ─────────────────────────────────────────────────────

class TestPluginManager:
    def test_discover_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            pm = PluginManager(plugin_dirs=[d])
            discovered = pm.discover()
            assert len(discovered) == 0

    def test_discover_nonexistent_dir(self):
        pm = PluginManager(plugin_dirs=["/nonexistent/path"])
        discovered = pm.discover()
        assert len(discovered) == 0

    def test_discover_plugin_file(self):
        with tempfile.TemporaryDirectory() as d:
            plugin_file = Path(d) / "my_plugin.py"
            plugin_file.write_text(
                'async def register(bus):\n    return []\n'
            )

            pm = PluginManager(plugin_dirs=[d])
            discovered = pm.discover()
            assert len(discovered) == 1
            assert discovered[0].name == "my_plugin"

    def test_skip_file_without_register(self):
        with tempfile.TemporaryDirectory() as d:
            plugin_file = Path(d) / "not_a_plugin.py"
            plugin_file.write_text('print("hello")\n')

            pm = PluginManager(plugin_dirs=[d])
            discovered = pm.discover()
            assert len(discovered) == 0

    def test_skip_underscore_files(self):
        with tempfile.TemporaryDirectory() as d:
            plugin_file = Path(d) / "__init__.py"
            plugin_file.write_text('async def register(bus): pass\n')

            pm = PluginManager(plugin_dirs=[d])
            discovered = pm.discover()
            assert len(discovered) == 0

    def test_discover_with_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            plugin_file = Path(d) / "fancy_plugin.py"
            plugin_file.write_text(
                '__plugin__ = {"name": "fancy", "version": "2.0", "description": "A fancy plugin"}\n'
                'async def register(bus):\n    return []\n'
            )

            pm = PluginManager(plugin_dirs=[d])
            discovered = pm.discover()
            assert len(discovered) == 1
            assert discovered[0].name == "fancy"
            assert discovered[0].version == "2.0"

    def test_loaded_count(self):
        pm = PluginManager()
        assert pm.loaded_count == 0

    def test_list_plugins(self):
        pm = PluginManager()
        assert pm.list_plugins() == []
