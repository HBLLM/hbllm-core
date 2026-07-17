"""
Skill Parser — Parses Markdown SKILL.md files into executable Skill objects.

Bridges the gap between human-readable skill definitions (compatible
with agentskills.io format) and HBLLM's internal SkillRegistry.

ProceduralMemory never sees Markdown. Instead::

    SKILL.md  (Markdown + YAML frontmatter)
        ↓
    SkillParser.parse()  (this module)
        ↓
    ParsedSkill  (structured data)
        ↓
    SkillRegistry / ProceduralMemory  (execution)

SKILL.md format::

    ---
    name: deploy-to-staging
    description: Deploy the current branch to the staging environment
    category: devops
    trigger: deploy staging|push to staging|release staging
    tools:
      - shell
      - git
    tags:
      - deployment
      - ci-cd
    version: 1
    author: dumith
    ---

    # Steps

    1. Pull latest changes from the current branch
    2. Run the test suite to ensure no regressions
    3. Build the Docker image with the staging tag
    4. Push the image to the container registry
    5. SSH into the staging server and pull the new image
    6. Restart the service and verify health check

    # Success Criteria

    The staging environment responds with HTTP 200 on the health endpoint.

    # Examples

    User: "Deploy my branch to staging"
    Assistant: "Running deployment pipeline for branch `feat/new-ui`..."

    # Notes

    Always run tests before deploying. If tests fail, abort and report.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Optional YAML support
try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None


# ═══════════════════════════════════════════════════════════════════════════
# Parsed Skill
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ParsedSkill:
    """Structured skill extracted from a SKILL.md file.

    This is the intermediate representation between Markdown and
    the internal Skill dataclass used by SkillRegistry.
    """

    skill_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    description: str = ""
    category: str = "general"
    trigger_patterns: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    version: int = 1
    author: str = ""

    # Parsed body sections
    steps: list[str] = field(default_factory=list)
    success_criteria: str = ""
    examples: list[dict[str, str]] = field(default_factory=list)
    notes: str = ""

    # Source tracking
    source_path: str = ""
    raw_markdown: str = ""

    def to_skill_dict(self) -> dict[str, Any]:
        """Convert to a dict compatible with SkillRegistry.store()."""
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "steps": self.steps,
            "tools_used": self.tools_used,
            "success_criteria": self.success_criteria,
            "examples": self.examples,
            "version": self.version,
            "source": f"skill-md:{self.source_path}" if self.source_path else "skill-md",
        }


# ═══════════════════════════════════════════════════════════════════════════
# Parser
# ═══════════════════════════════════════════════════════════════════════════

# Regex for YAML frontmatter
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Regex for markdown headers
_HEADER_RE = re.compile(r"^#+\s+(.+)$", re.MULTILINE)

# Regex for numbered list items
_NUMBERED_ITEM_RE = re.compile(r"^\s*\d+[\.\)]\s+(.+)$", re.MULTILINE)


class SkillParser:
    """Parses SKILL.md files into ParsedSkill objects.

    Supports the agentskills.io format:
      - YAML frontmatter for metadata
      - Markdown body with # Steps, # Success Criteria, # Examples, # Notes

    Usage::

        parser = SkillParser()

        # Parse a single file
        skill = parser.parse_file(Path("skills/deploy.md"))

        # Parse a string
        skill = parser.parse_string(markdown_content)

        # Parse all .md files in a directory
        skills = parser.parse_directory(Path("skills/"))
    """

    def parse_file(self, path: Path) -> ParsedSkill | None:
        """Parse a single SKILL.md file.

        Args:
            path: Path to the Markdown skill file.

        Returns:
            ParsedSkill, or None if parsing fails.
        """
        if not path.exists():
            logger.warning("Skill file not found: %s", path)
            return None

        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error("Failed to read skill file %s: %s", path, e)
            return None

        skill = self.parse_string(content)
        if skill:
            skill.source_path = str(path)
        return skill

    def parse_string(self, content: str) -> ParsedSkill | None:
        """Parse a SKILL.md string.

        Args:
            content: Raw Markdown content with YAML frontmatter.

        Returns:
            ParsedSkill, or None if parsing fails.
        """
        skill = ParsedSkill(raw_markdown=content)

        # 1. Extract YAML frontmatter
        frontmatter = self._extract_frontmatter(content)
        if frontmatter:
            skill.name = frontmatter.get("name", "")
            skill.description = frontmatter.get("description", "")
            skill.category = frontmatter.get("category", "general")
            skill.version = int(frontmatter.get("version", 1))
            skill.author = frontmatter.get("author", "")
            skill.tools_used = frontmatter.get("tools", [])
            skill.tags = frontmatter.get("tags", [])

            # Parse trigger patterns (pipe-separated or list)
            trigger = frontmatter.get("trigger", "")
            if isinstance(trigger, str):
                skill.trigger_patterns = [t.strip() for t in trigger.split("|") if t.strip()]
            elif isinstance(trigger, list):
                skill.trigger_patterns = trigger

        if not skill.name:
            logger.warning("Skill has no name in frontmatter")
            return None

        # 2. Extract body sections
        body = self._strip_frontmatter(content)
        sections = self._split_sections(body)

        # Steps
        steps_text = sections.get("steps", "")
        skill.steps = self._extract_numbered_items(steps_text)

        # Success criteria
        skill.success_criteria = sections.get("success criteria", "").strip()

        # Examples
        examples_text = sections.get("examples", "")
        skill.examples = self._extract_examples(examples_text)

        # Notes
        skill.notes = sections.get("notes", "").strip()

        logger.debug(
            "Parsed skill '%s': %d steps, %d triggers, %d tools",
            skill.name,
            len(skill.steps),
            len(skill.trigger_patterns),
            len(skill.tools_used),
        )
        return skill

    def parse_directory(self, directory: Path) -> list[ParsedSkill]:
        """Parse all .md files in a directory.

        Args:
            directory: Directory containing SKILL.md files.

        Returns:
            List of successfully parsed skills.
        """
        if not directory.is_dir():
            logger.warning("Skills directory not found: %s", directory)
            return []

        skills: list[ParsedSkill] = []
        for path in sorted(directory.glob("*.md")):
            skill = self.parse_file(path)
            if skill:
                skills.append(skill)

        logger.info("Parsed %d skills from %s", len(skills), directory)
        return skills

    # ── Internal Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _extract_frontmatter(content: str) -> dict[str, Any] | None:
        """Extract YAML frontmatter from Markdown content."""
        match = _FRONTMATTER_RE.match(content)
        if not match:
            return None

        if yaml is None:
            logger.warning("PyYAML not installed — cannot parse SKILL.md frontmatter")
            return None

        try:
            return yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as e:
            logger.error("Failed to parse YAML frontmatter: %s", e)
            return None

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        """Remove YAML frontmatter, leaving only the body."""
        match = _FRONTMATTER_RE.match(content)
        if match:
            return content[match.end():]
        return content

    @staticmethod
    def _split_sections(body: str) -> dict[str, str]:
        """Split Markdown body into sections keyed by header title (lowercase)."""
        sections: dict[str, str] = {}
        current_header: str | None = None
        current_lines: list[str] = []

        for line in body.split("\n"):
            header_match = _HEADER_RE.match(line)
            if header_match:
                # Save previous section
                if current_header is not None:
                    sections[current_header] = "\n".join(current_lines)
                current_header = header_match.group(1).strip().lower()
                current_lines = []
            else:
                current_lines.append(line)

        # Save last section
        if current_header is not None:
            sections[current_header] = "\n".join(current_lines)

        return sections

    @staticmethod
    def _extract_numbered_items(text: str) -> list[str]:
        """Extract numbered list items from text."""
        return _NUMBERED_ITEM_RE.findall(text)

    @staticmethod
    def _extract_examples(text: str) -> list[dict[str, str]]:
        """Extract User/Assistant example pairs from text."""
        examples: list[dict[str, str]] = []
        current: dict[str, str] = {}

        for line in text.strip().split("\n"):
            line = line.strip()
            if line.lower().startswith("user:"):
                if current:
                    examples.append(current)
                current = {"user": line[5:].strip().strip('"')}
            elif line.lower().startswith("assistant:"):
                current["assistant"] = line[10:].strip().strip('"')

        if current:
            examples.append(current)

        return examples
