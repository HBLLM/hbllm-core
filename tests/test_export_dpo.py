"""Tests for the DPO Dataset Export CLI."""

import json
import os
import pytest
from pathlib import Path

from hbllm.cli.export_dpo import (
    read_dpo_queue,
    read_reflection_logs,
    collect_all_pairs,
    export_jsonl,
    build_stats,
    main,
)


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace with sample DPO data."""
    reflection_dir = tmp_path / "workspace" / "reflection"
    reflection_dir.mkdir(parents=True)

    # DPO queue (LearnerNode format)
    queue_data = [
        ["How do I reset my password?", "Go to Settings > Security > Reset.", "I don't know."],
        ["What is Python?", "Python is a programming language.", "Python is a snake."],
        {"prompt": "Explain DPO", "chosen": "DPO optimizes preferences directly.", "rejected": "DPO is a protocol."},
    ]
    with open(reflection_dir / "dpo_queue.json", "w") as f:
        json.dump(queue_data, f)

    # Reflection logs (MetaReasoningNode format)
    reflection_data = [
        {"instruction": "How do I fix auth?", "response": "Unknown error.", "rejected": True, "domain": "auth_domain"},
        {"instruction": "Configure TLS?", "response": "Not sure.", "rejected": True, "domain": "security"},
    ]
    with open(reflection_dir / "reflection_auth_domain_abc123.jsonl", "w") as f:
        for item in reflection_data:
            f.write(json.dumps(item) + "\n")

    return tmp_path


# ── Unit Tests ──────────────────────────────────────────────────────────────


class TestReadDPOQueue:
    def test_reads_list_format(self, tmp_workspace):
        queue_path = tmp_workspace / "workspace" / "reflection" / "dpo_queue.json"
        pairs = read_dpo_queue(queue_path)
        assert len(pairs) == 3
        assert pairs[0]["prompt"] == "How do I reset my password?"
        assert pairs[0]["chosen"] == "Go to Settings > Security > Reset."
        assert pairs[0]["rejected"] == "I don't know."

    def test_reads_dict_format(self, tmp_workspace):
        queue_path = tmp_workspace / "workspace" / "reflection" / "dpo_queue.json"
        pairs = read_dpo_queue(queue_path)
        assert pairs[2]["prompt"] == "Explain DPO"
        assert pairs[2]["chosen"] == "DPO optimizes preferences directly."

    def test_missing_file_returns_empty(self, tmp_path):
        pairs = read_dpo_queue(tmp_path / "nonexistent.json")
        assert pairs == []

    def test_corrupt_json_returns_empty(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json{{{")
        pairs = read_dpo_queue(bad_file)
        assert pairs == []


class TestReadReflectionLogs:
    def test_reads_reflection_files(self, tmp_workspace):
        reflection_dir = tmp_workspace / "workspace" / "reflection"
        pairs = read_reflection_logs(reflection_dir)
        assert len(pairs) == 2
        assert pairs[0]["prompt"] == "How do I fix auth?"
        assert pairs[0]["rejected"] == "Unknown error."
        assert pairs[0]["domain"] == "auth_domain"
        assert pairs[0]["source"] == "reflection"

    def test_chosen_is_empty_placeholder(self, tmp_workspace):
        """Reflection logs only have rejected — chosen is empty for trainer to fill."""
        reflection_dir = tmp_workspace / "workspace" / "reflection"
        pairs = read_reflection_logs(reflection_dir)
        assert all(p["chosen"] == "" for p in pairs)

    def test_missing_dir_returns_empty(self, tmp_path):
        pairs = read_reflection_logs(tmp_path / "nonexistent")
        assert pairs == []

    def test_skips_corrupt_lines(self, tmp_workspace):
        reflection_dir = tmp_workspace / "workspace" / "reflection"
        with open(reflection_dir / "reflection_bad_12345.jsonl", "w") as f:
            f.write("valid line is missing\n")
            f.write("{invalid json\n")
        # Should not crash — just skip bad files
        pairs = read_reflection_logs(reflection_dir)
        assert len(pairs) >= 2  # At least the good ones from fixture


class TestCollectAllPairs:
    def test_collects_from_all_sources(self, tmp_workspace):
        pairs = collect_all_pairs(
            workspace_dir=tmp_workspace / "workspace",
            sources=["queue", "reflection"],
        )
        # 3 from queue + 2 from reflection = 5
        assert len(pairs) == 5

    def test_deduplicates_pairs(self, tmp_workspace):
        """Duplicate prompt+rejected combos should be removed."""
        # Add a duplicate to the queue
        queue_path = tmp_workspace / "workspace" / "reflection" / "dpo_queue.json"
        with open(queue_path) as f:
            data = json.load(f)
        data.append(data[0])  # Duplicate first entry
        with open(queue_path, "w") as f:
            json.dump(data, f)

        pairs = collect_all_pairs(
            workspace_dir=tmp_workspace / "workspace",
            sources=["queue"],
        )
        assert len(pairs) == 3  # Deduped

    def test_single_source_filter(self, tmp_workspace):
        pairs = collect_all_pairs(
            workspace_dir=tmp_workspace / "workspace",
            sources=["queue"],
        )
        assert len(pairs) == 3

    def test_empty_workspace(self, tmp_path):
        pairs = collect_all_pairs(workspace_dir=tmp_path / "empty")
        assert pairs == []


class TestExportJsonl:
    def test_exports_valid_jsonl(self, tmp_workspace, tmp_path):
        pairs = collect_all_pairs(
            workspace_dir=tmp_workspace / "workspace",
            sources=["queue"],
        )
        output = tmp_path / "output.jsonl"
        count = export_jsonl(pairs, output)

        assert count == 3
        assert output.exists()

        # Verify each line is valid JSON with required fields
        with open(output) as f:
            for line in f:
                record = json.loads(line)
                assert "prompt" in record
                assert "chosen" in record
                assert "rejected" in record

    def test_skips_empty_prompts(self, tmp_path):
        pairs = [
            {"prompt": "", "chosen": "yes", "rejected": "no"},
            {"prompt": "valid", "chosen": "good", "rejected": "bad"},
        ]
        output = tmp_path / "filtered.jsonl"
        count = export_jsonl(pairs, output)
        assert count == 1

    def test_creates_parent_dirs(self, tmp_path):
        output = tmp_path / "deep" / "nested" / "dir" / "output.jsonl"
        export_jsonl([{"prompt": "q", "chosen": "a", "rejected": "b"}], output)
        assert output.exists()


class TestBuildStats:
    def test_computes_stats(self):
        pairs = [
            {"prompt": "q1", "chosen": "c1", "rejected": "r1", "source": "queue"},
            {"prompt": "q2", "chosen": "c2", "rejected": "r2", "source": "queue"},
            {"prompt": "q3", "chosen": "", "rejected": "r3", "source": "reflection", "domain": "auth"},
        ]
        stats = build_stats(pairs)
        assert stats["total_pairs"] == 3
        assert stats["complete_pairs"] == 2
        assert stats["incomplete_pairs"] == 1
        assert stats["sources"]["queue"] == 2
        assert stats["sources"]["reflection"] == 1
        assert stats["domains"]["auth"] == 1


class TestCLI:
    def test_main_exports_successfully(self, tmp_workspace, tmp_path):
        output = str(tmp_path / "cli_output.jsonl")
        exit_code = main([
            "--output", output,
            "--workspace", str(tmp_workspace / "workspace"),
            "--source", "queue", "reflection",
        ])
        assert exit_code == 0
        assert Path(output).exists()
        with open(output) as f:
            lines = f.readlines()
        assert len(lines) >= 3

    def test_main_empty_workspace(self, tmp_path):
        output = str(tmp_path / "empty.jsonl")
        exit_code = main([
            "--output", output,
            "--workspace", str(tmp_path / "nonexistent"),
        ])
        assert exit_code == 1  # No pairs found

    def test_main_with_stats(self, tmp_workspace, tmp_path, capsys):
        output = str(tmp_path / "stats_output.jsonl")
        exit_code = main([
            "--output", output,
            "--workspace", str(tmp_workspace / "workspace"),
            "--source", "queue",
            "--stats",
        ])
        assert exit_code == 0
        captured = capsys.readouterr()
        # Stats JSON is printed with indent=2, followed by the success message
        # Extract the JSON portion (everything between first { and matching })
        out = captured.out
        json_start = out.index("{")
        json_end = out.rindex("}") + 1
        stats = json.loads(out[json_start:json_end])
        assert stats["total_pairs"] == 3
