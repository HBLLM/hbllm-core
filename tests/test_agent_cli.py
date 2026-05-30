"""
Tests for the agent and code subcommands in the CLI interface.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hbllm.cli.agent import register_subcommands, run_code


def test_register_subcommands() -> None:
    """Verify that both subcommands 'agent' and 'code' register successfully."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    register_subcommands(subparsers)

    # Trigger parser to check subcommand setup
    args_agent = parser.parse_args(["agent", "--model", "openai/gpt-4o"])
    assert args_agent.command == "agent"
    assert args_agent.model == "openai/gpt-4o"

    args_code = parser.parse_args(["code", "./tests", "--model", "ollama/llama3"])
    assert args_code.command == "code"
    assert args_code.path == "./tests"
    assert args_code.model == "ollama/llama3"


def test_run_code_directory_error() -> None:
    """Verify that run_code exits/fails when target path is not a directory."""
    args = argparse.Namespace(
        path="/nonexistent/directory/path/here",
        model="openai/gpt-4o-mini",
        no_approval=False,
        index=False,
        no_index=False,
        task=None,
    )

    with patch("sys.exit", side_effect=SystemExit) as mock_exit:
        with pytest.raises(SystemExit):
            run_code(args)
        mock_exit.assert_called_once_with(1)


def test_run_code_success() -> None:
    """Verify that run_code changes directory, computes tenant_id/data_dir, and launches the agent."""
    with tempfile.TemporaryDirectory() as temp_dir:
        real_temp_dir = os.path.realpath(temp_dir)
        args = argparse.Namespace(
            path=real_temp_dir,
            model="openai/gpt-4o-mini",
            no_approval=True,
            index=False,
            no_index=True,
            task="list files",
        )

        original_cwd = os.getcwd()
        try:
            # We mock _run_agent to prevent it from booting up a real Brain/LLM provider
            with patch("hbllm.cli.agent._run_agent", new_callable=AsyncMock) as mock_run:
                run_code(args)

                # 1. Assert working directory was changed to target path
                assert os.path.realpath(os.getcwd()) == real_temp_dir

                # 2. Check call arguments for _run_agent
                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args[1]

                assert call_kwargs["model"] == "openai/gpt-4o-mini"
                assert call_kwargs["require_approval"] is False
                assert call_kwargs["task"] == "list files"
                assert call_kwargs["index"] is False
                assert call_kwargs["no_index"] is True

                # Check data directory is isolated under target_path/.hbllm
                data_dir = call_kwargs["data_dir"]
                assert data_dir == os.path.join(real_temp_dir, ".hbllm")
                assert call_kwargs["tenant_id"].startswith("dev_")

        finally:
            # Restore CWD
            os.chdir(original_cwd)
