#!/usr/bin/env python3
"""Build a clean, production-ready Kaggle Dataset package for ARC Prize 2026.

Packages only the necessary pure-Python sources:
- hbllm/ (core cognitive architecture, pure Python, no native binaries)
- plugins/arc_agi_adapter/ (procedural solvers & benchmark adapters)
- kaggle_submission/ (submission agent and evaluation harness)

Explicitly filters out:
- Secrets (.env, *.env)
- Platform binaries (*.so, *.dylib, *.dll, target/)
- Databases (*.db, *.sqlite)
- Cache directories (__pycache__, .pytest_cache, .mypy_cache, .ruff_cache)
- Development files (.git, .venv, tests)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dylib",
    ".dll",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".env",
    ".DS_Store",
}

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "llama_venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    "target",
    "docs",
    "site",
    "scratch",
    "artifacts",
    "tests",  # Keep archive light; competition only needs runtime solvers
}


def should_include_file(file_path: Path) -> bool:
    """Check if file should be included in the submission archive."""
    for part in file_path.parts:
        if part in EXCLUDE_DIRS or part.startswith(".env"):
            return False
        if part.startswith(".") and part != ".":
            return False

    if file_path.suffix in EXCLUDE_EXTENSIONS:
        return False

    # Block any secret file patterns
    name_lower = file_path.name.lower()
    if "secret" in name_lower or "token" in name_lower or "credential" in name_lower:
        return False

    return True


def package_dataset(output_zip: Path | None = None) -> Path:
    core_dir = Path(__file__).resolve().parent.parent
    if output_zip is None:
        output_zip = core_dir / "kaggle_submission" / "hbllm_kaggle_dataset.zip"

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

    included_files: list[tuple[Path, str]] = []

    # 1. hbllm/ (Pure Python package)
    hbllm_dir = core_dir / "hbllm"
    for root, _, files in os.walk(hbllm_dir):
        for f in files:
            fp = Path(root) / f
            if should_include_file(fp):
                rel = fp.relative_to(core_dir)
                included_files.append((fp, str(rel)))

    # 2. plugins/arc_agi_adapter/ + plugins/__init__.py
    plugins_dir = core_dir / "plugins"
    plugins_init = plugins_dir / "__init__.py"
    if plugins_init.exists():
        included_files.append((plugins_init, "plugins/__init__.py"))
    else:
        # Fallback dummy __init__.py
        pass

    arc_adapter_dir = plugins_dir / "arc_agi_adapter"
    for root, _, files in os.walk(arc_adapter_dir):
        for f in files:
            fp = Path(root) / f
            if should_include_file(fp):
                rel = fp.relative_to(core_dir)
                included_files.append((fp, str(rel)))

    # 3. kaggle_submission/ (submission.py, test_synthetic_eval.py, README.md, arc_agi_3_submission.ipynb)
    sub_dir = core_dir / "kaggle_submission"
    for f in ["submission.py", "test_synthetic_eval.py", "README.md", "arc_agi_3_submission.ipynb"]:
        fp = sub_dir / f
        if fp.exists():
            rel = fp.relative_to(core_dir)
            included_files.append((fp, str(rel)))

    print(f"📦 Packaging {len(included_files)} files into {output_zip.name}...")

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # Ensure plugins/__init__.py is inside the zip
        if not any(arc_name == "plugins/__init__.py" for _, arc_name in included_files):
            zf.writestr("plugins/__init__.py", '"""Plugins package."""\n')

        for src, arc_name in included_files:
            zf.write(src, arc_name)

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"✅ Archive created successfully: {output_zip} ({size_mb:.2f} MB)")
    return output_zip


def verify_package(zip_path: Path) -> bool:
    """Verify that the packaged zip can be extracted and executed in an isolated environment."""
    print(f"\n🧪 Testing isolated execution from {zip_path.name}...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)

        # Run test_synthetic_eval.py using Python in a clean subprocess pointing only to tmp_path
        test_script = tmp_path / "kaggle_submission" / "test_synthetic_eval.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(tmp_path)

        py_bin = sys.executable
        local_venv = zip_path.parent.parent / ".venv" / "bin" / "python"
        if local_venv.exists():
            py_bin = str(local_venv)

        res = subprocess.run(
            [py_bin, str(test_script)],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )

        if res.returncode == 0:
            print(
                "🎉 Validation Passed! All synthetic procedural tests passed in isolated environment."
            )
            return True
        else:
            print("❌ Validation Failed:")
            print("STDOUT:", res.stdout)
            print("STDERR:", res.stderr)
            return False


def main() -> None:
    zip_path = package_dataset()
    success = verify_package(zip_path)
    if not success:
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🚀 HOW TO USE IN KAGGLE NOTEBOOK:")
    print("=" * 70)
    print("1. Upload this zip as a Kaggle Private Dataset:")
    print(f"   File: {zip_path}")
    print("\n2. In your Kaggle Submission Notebook, add this in Cell 1:")
    print("   ```python")
    print("   import os, sys, glob")
    print("   for root, dirs, _ in os.walk('/kaggle/input'):")
    print("       if 'kaggle_submission' in dirs and 'hbllm' in dirs:")
    print("           sys.path.insert(0, root)")
    print("           break")
    print("   wheels = glob.glob('/kaggle/input/**/*.whl', recursive=True)")
    print("   if wheels:")
    print("       !pip install --no-index --find-links={os.path.dirname(wheels[0])} arcengine")
    print("   from kaggle_submission.submission import MyAgent")
    print("   agent = MyAgent()")
    print("   ```")
    print("=" * 70)


if __name__ == "__main__":
    main()
