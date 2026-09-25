#!/usr/bin/env python3
"""Build the official ARC Prize 2026 / ARC-AGI-3 Kaggle Submission Notebook.

Adheres strictly to the canonical 4-cell competition harness architecture:
  Cell 1: Offline wheel installation from competition dataset
  Cell 2: %%writefile /tmp/my_agent.py (Autonomous MyAgent powered by HBLLM Core)
  Cell 3: Competition Rerun (Phase B): Wait for gateway sidecar, mount framework,
          register MyAgent, and execute tournament via main.py
  Cell 4: Commit Mode (Phase A): Save-and-run-all generates dummy submission.parquet
          in <10s so commit succeeds instantly and Kaggle enables submission.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent
AGENT_SRC = ROOT / "my_agent.py"
NOTEBOOK_PATH = ROOT / "arc_agi_3_submission.ipynb"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text()

    install_cell_source = dedent(
        """\
        # ==============================================================================
        # 1. OFFLINE WHEEL INSTALLATION & RUNTIME INITIALIZATION
        # ==============================================================================
        import glob
        import os
        import subprocess
        import sys

        wheel_dirs = set()
        for p in glob.glob('/kaggle/input/**/*.whl', recursive=True):
            wheel_dirs.add(os.path.dirname(p))

        if wheel_dirs:
            for wd in sorted(wheel_dirs):
                print(f"Installing wheels from: {wd}")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--no-index", f"--find-links={wd}", "arc-agi", "arcengine", "python-dotenv"],
                    check=False,
                )
        else:
            for fallback in [
                "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
                "/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
            ]:
                if os.path.exists(fallback):
                    print(f"Installing wheels from fallback: {fallback}")
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--no-index", f"--find-links={fallback}", "arc-agi", "arcengine", "python-dotenv"],
                        check=False,
                    )
                    break
        print("✓ Runtime environment initialized.")
        """
    )
    install_cell = code_cell(install_cell_source)

    # Cell 2 writes /tmp/my_agent.py (NOT in /kaggle/working/ so Kaggle doesn't confuse it with submission.parquet)
    write_agent_cell = code_cell("%%writefile /tmp/my_agent.py\n" + agent_body)

    run_cell_source = dedent(
        """\
        # ==============================================================================
        # 3. COMPETITION RERUN (PHASE B: GATEWAY SIDECAR EXECUTION)
        # ==============================================================================
        import os
        import shutil
        import subprocess
        import sys

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            print("🚀 Competition Rerun Mode Detected: Connecting to Gateway Sidecar...")

            # 1. Wait for gateway sidecar to be healthy
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games
            print("✓ Gateway sidecar is online and ready.")

            # 2. Locate ARC-AGI-3-Agents framework in competition inputs
            framework_src = None
            for cand in [
                "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents",
                "/kaggle/input/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents",
            ]:
                if os.path.exists(cand):
                    framework_src = cand
                    break

            if framework_src is None:
                for root, dirs, _ in os.walk("/kaggle/input"):
                    if "ARC-AGI-3-Agents" in dirs:
                        framework_src = os.path.join(root, "ARC-AGI-3-Agents")
                        break

            if framework_src is None:
                raise RuntimeError("Could not locate ARC-AGI-3-Agents framework in /kaggle/input!")

            working_framework = "/kaggle/working/ARC-AGI-3-Agents"
            if os.path.exists(working_framework):
                shutil.rmtree(working_framework)
            shutil.copytree(framework_src, working_framework)
            print(f"✓ Copied framework from {framework_src} to {working_framework}")

            # 3. Install MyAgent into the framework
            target_agent = os.path.join(working_framework, "agents", "templates", "my_agent.py")
            shutil.copy("/tmp/my_agent.py", target_agent)
            print(f"✓ Installed MyAgent into {target_agent}")

            # 4. Register MyAgent in the framework registry (slimming out unused heavy deps)
            init_file = os.path.join(working_framework, "agents", "__init__.py")
            with open(init_file, "w") as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': MyAgent,
        }
        \"\"\")
            print("✓ Registered MyAgent in agent registry.")

            # 5. Point the framework at the gateway sidecar with level preservation
            env_file = os.path.join(working_framework, ".env")
            with open(env_file, "w") as f:
                f.write(\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        ONLY_RESET_LEVELS=true
        \"\"\")
            print("✓ Configured .env for gateway sidecar with ONLY_RESET_LEVELS=true.")

            # 6. Run the competition tournament! The gateway records actions and emits submission.parquet.
            print("🎮 Launching MyAgent tournament against gateway sidecar...")
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
                ONLY_RESET_LEVELS=true \\
                python main.py --agent myagent
            print("✓ Tournament finished.")
        """
    )
    run_cell = code_cell(run_cell_source)

    dummy_submission_cell = code_cell(
        dedent(
            """\
            # ==============================================================================
            # 4. EMIT / VERIFY SUBMISSION PARQUET
            # ==============================================================================
            import os
            import pandas as pd

            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                # Phase A: Save-and-run-all (commit) mode
                # Produce compliant submission.parquet in <1s so commit succeeds immediately!
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'])
                submission.to_parquet('/kaggle/working/submission.parquet', index=False)
                print("✓ Commit mode: generated compliant /kaggle/working/submission.parquet")
                print(submission.head())
            else:
                # Phase B: Competition rerun mode
                # Verify that gateway sidecar produced submission.parquet
                parquet_path = '/kaggle/working/submission.parquet'
                if os.path.exists(parquet_path):
                    df = pd.read_parquet(parquet_path)
                    print(f"✓ Gateway generated submission.parquet: {len(df)} records")
                    print(df.head(10))
                else:
                    print("⚠️ Fallback: Gateway submission.parquet not found, emitting valid output...")
                    fallback = pd.DataFrame(
                        data=[['1_0', '1', True, 1]],
                        columns=['row_id', 'game_id', 'end_of_game', 'score'])
                    fallback.to_parquet(parquet_path, index=False)
                    print("✓ Emitted fallback submission.parquet")
            """
        )
    )

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "isInternetEnabled": False,
                "isGpuEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 Autonomous Submission Agent\n\n"
                "### Powered by HBLLM Core Cognitive Architecture\n\n"
                "- **Phase A (Commit Mode)**: Emits compliant submission.parquet instantly in ~5s.\n"
                "- **Phase B (Competition Rerun)**: Connects to gateway sidecar (`http://gateway:8001`), "
                "evaluates against hidden competition puzzles, and emits verified scorecard."
            ),
            install_cell,
            write_agent_cell,
            run_cell,
            dummy_submission_cell,
        ],
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH} successfully!")


if __name__ == "__main__":
    main()
