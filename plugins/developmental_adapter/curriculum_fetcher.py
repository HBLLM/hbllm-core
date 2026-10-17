"""Curriculum Fetcher and Ingestion Pipeline for HBLLM Developmental Learning.

Discovers, downloads, standardizes, and caches open-access educational textbooks
and scientific articles (OpenStax, Project Gutenberg, peS2o, Wikibooks) for
embodied developmental learning.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .textbook_curriculum import (
    TextbookChapter,
    TextbookParser,
)

logger = logging.getLogger(__name__)

CURRICULUM_DATA_DIR = Path(__file__).parent / "curriculum_data"


class CurriculumFetcher:
    """Manages educational textbook curricula for developmental training."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else CURRICULUM_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def list_available_chapters(self) -> list[Path]:
        """Return sorted list of all available curriculum markdown files."""
        return sorted(list(self.data_dir.glob("*.md")))

    def load_chapter(self, file_path: Path | str) -> TextbookChapter:
        """Parse a specific chapter markdown file into a typed TextbookChapter."""
        p = Path(file_path)
        if not p.is_absolute():
            p = self.data_dir / p

        if not p.exists():
            raise FileNotFoundError(f"Curriculum chapter file not found: {p}")

        with open(p, encoding="utf-8") as f:
            content = f.read()

        chapter_id = p.stem
        return TextbookParser.parse_markdown(content, chapter_id=chapter_id)

    def load_all_chapters(self) -> list[TextbookChapter]:
        """Load and parse all available curriculum files in pedagogical order."""
        chapter_paths = self.list_available_chapters()
        chapters: list[TextbookChapter] = []
        for path in chapter_paths:
            try:
                ch = self.load_chapter(path)
                chapters.append(ch)
            except Exception as e:
                logger.warning(f"Failed to parse curriculum file {path}: {e}")
        return chapters

    def synthesize_chapter(
        self,
        chapter_id: str,
        title: str,
        grade_level: int,
        source_attribution: str,
        glossary: dict[str, str],
        worked_instruction: str,
        worked_scenario: str,
        analogy_source: str,
        analogy_target: str,
        analogy_mapping: str,
        review_questions: list[str],
        save_to_disk: bool = True,
    ) -> TextbookChapter:
        """Programmatically synthesize a structured textbook chapter adhering to the A23 schema."""
        lines = [
            f"# {title} (Grade {grade_level})",
            "",
            f"Source: {source_attribution}",
            "",
            "## 1. Glossary & Key Definitions",
        ]
        for term, definition in glossary.items():
            lines.append(f"* **{term.capitalize()}**: {definition}")

        lines.extend(
            [
                "",
                "## 2. Worked Problem: Guided Sensorimotor Manipulation",
                "In physical mechanics, learning requires coordinating tools and objects.",
                f"* Instruction: `{worked_instruction}`",
                f"* Physical Scenario: {worked_scenario}",
                "* Mechanics Execution: The student plans and executes the required manipulation steps.",
                "",
                f"## 3. Relational Analogy: {analogy_source} to {analogy_target}",
                f"* Source Domain: {analogy_source}",
                f"* Target Domain: {analogy_target}",
                f"* Structural Mapping: {analogy_mapping}",
                "",
                "## 4. Review & Challenge Exercises",
            ]
        )
        for q in review_questions:
            lines.append(f"* {q}")
        lines.append(
            "* Conceptual Trick Question: Consider an unobserved hypothetical object with zero sensory data."
        )
        lines.append(
            "* Epistemic Defense Mandate: High epistemic uncertainty requires abstaining from unverified guessing."
        )

        md_text = "\n".join(lines)
        if save_to_disk:
            out_file = self.data_dir / f"{chapter_id}.md"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(md_text)

        return TextbookParser.parse_markdown(md_text, chapter_id=chapter_id)
