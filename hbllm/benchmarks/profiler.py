"""
Performance Profiler — measures memory, throughput, and pipeline latency.

Complements the benchmark runner by profiling real resource usage rather
than comparing architectures. Answers: "how does HBLLM perform under load?"

Usage::

    python -m hbllm.benchmarks.profiler --suite all
    python -m hbllm.benchmarks.profiler --suite memory
    python -m hbllm.benchmarks.profiler --suite throughput
    python -m hbllm.benchmarks.profiler --output profile.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hbllm.benchmarks.runner import BenchmarkResult, BenchmarkReport

logger = logging.getLogger(__name__)


# ── Memory Profiler ──────────────────────────────────────────────────────────

async def profile_memory() -> BenchmarkReport:
    """Measure memory footprint of each memory system under load."""
    report = BenchmarkReport(suite="profile_memory")

    # Episodic Memory
    import tempfile
    from hbllm.memory.episodic import EpisodicMemory

    sizes = [100, 1_000, 10_000]
    for n in sizes:
        with tempfile.TemporaryDirectory() as tmp:
            mem = EpisodicMemory(db_path=Path(tmp) / "ep.db")
            t0 = time.perf_counter()
            for i in range(n):
                mem.store_turn(f"s{i % 10}", "user", f"Message number {i} " * 5)
            elapsed = time.perf_counter() - t0

            db_size = (Path(tmp) / "ep.db").stat().st_size

            report.add(BenchmarkResult(
                name=f"EpisodicMemory — {n} turns",
                metric=f"ep_write_{n}",
                value=round(elapsed * 1000, 2),
                unit="ms",
                metadata={"db_size_kb": round(db_size / 1024, 1)},
            ))
            report.add(BenchmarkResult(
                name=f"EpisodicMemory DB — {n} turns",
                metric=f"ep_size_{n}",
                value=round(db_size / 1024, 1),
                unit="KB",
            ))

    # Semantic Memory (TF-IDF mode)
    from hbllm.memory.semantic import SemanticMemory

    for n in [100, 1_000]:
        mem = SemanticMemory()
        mem._use_tfidf = True

        t0 = time.perf_counter()
        for i in range(n):
            mem.store(f"Document about topic {i} with various keywords and content")
        write_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        mem.search("topic keywords content")
        search_time = time.perf_counter() - t0

        vec_size = sys.getsizeof(mem.vectors.tobytes()) if mem.vectors is not None else 0

        report.add(BenchmarkResult(
            name=f"SemanticMemory TF-IDF — {n} docs write",
            metric=f"sem_write_{n}",
            value=round(write_time * 1000, 2),
            unit="ms",
        ))
        report.add(BenchmarkResult(
            name=f"SemanticMemory TF-IDF — {n} docs search",
            metric=f"sem_search_{n}",
            value=round(search_time * 1000, 3),
            unit="ms",
        ))
        report.add(BenchmarkResult(
            name=f"SemanticMemory vectors — {n} docs",
            metric=f"sem_vec_size_{n}",
            value=round(vec_size / 1024, 1),
            unit="KB",
        ))

    # Procedural Memory
    from hbllm.memory.procedural import ProceduralMemory

    with tempfile.TemporaryDirectory() as tmp:
        mem = ProceduralMemory(db_path=Path(tmp) / "proc.db")
        t0 = time.perf_counter()
        for i in range(500):
            mem.store_skill(
                "t1", f"skill_{i}", f"trigger {i}",
                [{"step": j} for j in range(3)],
            )
        elapsed = time.perf_counter() - t0

        report.add(BenchmarkResult(
            name="ProceduralMemory — 500 skills write",
            metric="proc_write_500",
            value=round(elapsed * 1000, 2),
            unit="ms",
        ))

        t0 = time.perf_counter()
        mem.find_skill("t1", "skill_250")
        search_time = time.perf_counter() - t0
        report.add(BenchmarkResult(
            name="ProceduralMemory — skill search",
            metric="proc_search",
            value=round(search_time * 1000, 3),
            unit="ms",
        ))

    return report


# ── Bus Throughput ───────────────────────────────────────────────────────────

async def profile_throughput() -> BenchmarkReport:
    """Measure sustained bus throughput with varying payload sizes."""
    from hbllm.network.bus import InProcessBus
    from hbllm.network.messages import Message, MessageType

    report = BenchmarkReport(suite="profile_throughput")

    payload_sizes = {
        "tiny (10B)": {"x": "y"},
        "small (1KB)": {"text": "x" * 1024},
        "medium (10KB)": {"text": "x" * 10240},
        "large (100KB)": {"text": "x" * 102400},
    }

    for label, payload in payload_sizes.items():
        bus = InProcessBus()
        await bus.start()

        count = 0

        async def counter(msg):
            nonlocal count
            count += 1

        await bus.subscribe("perf.throughput", counter)

        n_messages = 5000
        t0 = time.perf_counter()

        for _ in range(n_messages):
            msg = Message(
                type=MessageType.EVENT,
                source_node_id="profiler",
                topic="perf.throughput",
                payload=payload,
            )
            await bus.publish("perf.throughput", msg)

        await asyncio.sleep(0.5)
        elapsed = time.perf_counter() - t0

        throughput = count / elapsed if elapsed > 0 else 0

        report.add(BenchmarkResult(
            name=f"Bus throughput — {label}",
            metric=f"throughput_{label.split()[0]}",
            value=round(throughput),
            unit="msg/s",
            metadata={"messages_delivered": count, "elapsed_s": round(elapsed, 3)},
        ))

        await bus.stop()

    return report


# ── Node Startup ─────────────────────────────────────────────────────────────

async def profile_startup() -> BenchmarkReport:
    """Time to start/stop each node type."""
    from hbllm.network.bus import InProcessBus
    from hbllm.brain.router_node import RouterNode
    from hbllm.brain.decision_node import DecisionNode
    from hbllm.brain.planner_node import PlannerNode
    from hbllm.brain.critic_node import CriticNode

    report = BenchmarkReport(suite="profile_startup")

    node_classes = [
        ("RouterNode", RouterNode),
        ("DecisionNode", DecisionNode),
        ("PlannerNode", PlannerNode),
        ("CriticNode", CriticNode),
    ]

    bus = InProcessBus()
    await bus.start()

    for name, cls in node_classes:
        # Measure start
        node = cls(node_id=f"perf_{name.lower()}")
        t0 = time.perf_counter()
        await node.start(bus)
        start_time = time.perf_counter() - t0

        # Measure stop
        t0 = time.perf_counter()
        await node.stop()
        stop_time = time.perf_counter() - t0

        report.add(BenchmarkResult(
            name=f"{name} start",
            metric=f"start_{name.lower()}",
            value=round(start_time * 1000, 3),
            unit="ms",
        ))
        report.add(BenchmarkResult(
            name=f"{name} stop",
            metric=f"stop_{name.lower()}",
            value=round(stop_time * 1000, 3),
            unit="ms",
        ))

    await bus.stop()
    return report


# ── Pipeline Latency ─────────────────────────────────────────────────────────

async def profile_pipeline() -> BenchmarkReport:
    """End-to-end message flow latency: publish → subscribe → callback."""
    from hbllm.network.bus import InProcessBus
    from hbllm.network.messages import Message, MessageType
    import statistics

    report = BenchmarkReport(suite="profile_pipeline")

    bus = InProcessBus()
    await bus.start()

    latencies: list[float] = []
    received = asyncio.Event()

    async def measure(msg):
        sent_at = msg.payload.get("sent_at", 0)
        latencies.append(time.perf_counter() - sent_at)
        if len(latencies) >= 1000:
            received.set()

    await bus.subscribe("pipeline.perf", measure)

    for _ in range(1000):
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="profiler",
            topic="pipeline.perf",
            payload={"sent_at": time.perf_counter()},
        )
        await bus.publish("pipeline.perf", msg)

    try:
        await asyncio.wait_for(received.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        pass

    if latencies:
        latencies_ms = [l * 1000 for l in latencies]
        latencies_ms.sort()

        report.add(BenchmarkResult(
            name="Pipeline p50", metric="pipeline_p50",
            value=round(latencies_ms[len(latencies_ms) // 2], 4), unit="ms",
        ))
        report.add(BenchmarkResult(
            name="Pipeline p99", metric="pipeline_p99",
            value=round(latencies_ms[int(len(latencies_ms) * 0.99)], 4), unit="ms",
        ))
        report.add(BenchmarkResult(
            name="Pipeline mean", metric="pipeline_mean",
            value=round(statistics.mean(latencies_ms), 4), unit="ms",
        ))
        report.add(BenchmarkResult(
            name="Pipeline stdev", metric="pipeline_stdev",
            value=round(statistics.stdev(latencies_ms), 4) if len(latencies_ms) > 1 else 0, unit="ms",
        ))

    await bus.stop()
    return report


# ── Suite Registry ───────────────────────────────────────────────────────────

PROFILE_SUITES = {
    "memory": profile_memory,
    "throughput": profile_throughput,
    "startup": profile_startup,
    "pipeline": profile_pipeline,
}


async def run_profile(suite: str) -> BenchmarkReport:
    if suite not in PROFILE_SUITES:
        raise ValueError(f"Unknown suite: {suite}. Available: {list(PROFILE_SUITES.keys())}")
    return await PROFILE_SUITES[suite]()


# ── CLI ──────────────────────────────────────────────────────────────────────

def _print_report(report: BenchmarkReport) -> None:
    """Human-readable profile report."""
    print(f"\n{'=' * 70}")
    print(f"  HBLLM Performance Profile — {report.suite}")
    print(f"{'=' * 70}\n")

    for r in report.results:
        val = f"{r.value:,.2f}" if isinstance(r.value, float) else str(r.value)
        print(f"  {r.name:<45} {val:>12} {r.unit}")
        if r.metadata:
            for k, v in r.metadata.items():
                print(f"    └ {k}: {v}")

    print(f"\n{'=' * 70}\n")


async def main():
    parser = argparse.ArgumentParser(description="HBLLM Performance Profiler")
    parser.add_argument("--suite", default="all", choices=list(PROFILE_SUITES.keys()) + ["all"])
    parser.add_argument("--output", type=str, help="Save JSON to file")
    args = parser.parse_args()

    suites_to_run = list(PROFILE_SUITES.keys()) if args.suite == "all" else [args.suite]
    all_reports = []

    for suite in suites_to_run:
        report = await run_profile(suite)
        _print_report(report)
        all_reports.append(report)

    if args.output:
        path = Path(args.output)
        combined = {"profiles": [r.to_dict() for r in all_reports]}
        path.write_text(json.dumps(combined, indent=2))
        print(f"Saved results to {path}")


if __name__ == "__main__":
    asyncio.run(main())
