"""
Unified Core Extractor — parses folder, markdown, python, and json/yaml structures
into stable hierarchical entities and relations.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass, field

from hbllm.core.constants import SKIP_DIRS
from hbllm.graph.types import Entity, Relation

logger = logging.getLogger(__name__)

# Try optional YAML support
try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

# Safeguard Limits
MAX_ENTITIES = 25000
MAX_RELATIONS = 50000


@dataclass
class StructuralResult:
    """The result of extracting structural relationships from a source."""

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)

    def merge(self, other: StructuralResult) -> None:
        """Merge another result into this one."""
        self.entities.extend(other.entities)
        self.relations.extend(other.relations)


class BaseExtractor:
    """Base class for format-specific structural extractors."""

    def __init__(self, root_path: str):
        self.root_path = os.path.abspath(root_path)

    def get_relative_path(self, file_path: str) -> str:
        """Helper to get a normalized relative path from the root path."""
        try:
            rel = os.path.relpath(os.path.abspath(file_path), self.root_path)
        except ValueError:
            rel = os.path.basename(file_path)
        return rel.replace("\\", "/")

    def extract(self, file_path: str) -> StructuralResult:
        """Extract entities and relations from a specific file path."""
        raise NotImplementedError


class FolderExtractor:
    """Walks the folder hierarchy to extract structure-based contains relations."""

    def __init__(
        self, root_path: str, max_depth: int = 20, max_folders: int = 1000, max_files: int = 5000
    ):
        self.root_path = os.path.abspath(root_path)
        self.max_depth = max_depth
        self.max_folders = max_folders
        self.max_files = max_files

    def extract(self) -> StructuralResult:
        result = StructuralResult()
        folders_count = 0
        files_count = 0

        # We construct a folder node for the root itself if it is a directory
        if os.path.isdir(self.root_path):
            root_id = "folder::."
            result.entities.append(
                Entity(
                    id=root_id,
                    label=os.path.basename(self.root_path) or ".",
                    entity_type="folder",
                    attributes={
                        "name": os.path.basename(self.root_path) or ".",
                        "category": "structure",
                    },
                )
            )

        for root, dirs, files in os.walk(self.root_path):
            # Prune skipped directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            # Calculate current depth relative to root
            try:
                rel_root = os.path.relpath(root, self.root_path)
            except ValueError:
                rel_root = ""

            if rel_root == ".":
                depth = 0
            else:
                depth = len(rel_root.split(os.sep))

            if depth > self.max_depth:
                continue

            # Normalized parent directory path & ID
            if rel_root == ".":
                parent_id = "folder::."
            else:
                norm_parent = rel_root.replace("\\", "/")
                parent_id = f"folder::{norm_parent}"

            # Walk child directories
            for d in dirs:
                if folders_count >= self.max_folders:
                    break
                folders_count += 1

                child_rel = os.path.join(rel_root, d) if rel_root != "." else d
                child_rel_norm = child_rel.replace("\\", "/")
                child_id = f"folder::{child_rel_norm}"

                # Create child folder entity
                result.entities.append(
                    Entity(
                        id=child_id,
                        label=d,
                        entity_type="folder",
                        attributes={"name": d, "category": "structure"},
                    )
                )
                # Add contains relation
                result.relations.append(
                    Relation(source_id=parent_id, target_id=child_id, relation_type="contains")
                )

            # Walk files in this directory
            for f in files:
                if files_count >= self.max_files:
                    break
                files_count += 1

                child_rel = os.path.join(rel_root, f) if rel_root != "." else f
                child_rel_norm = child_rel.replace("\\", "/")
                file_id = f"file::{child_rel_norm}"

                # Check if it is a python file to set type as module, otherwise document
                ext = os.path.splitext(f)[1].lower()
                entity_type = "module" if ext == ".py" else "document"

                result.entities.append(
                    Entity(
                        id=file_id,
                        label=f,
                        entity_type=entity_type,
                        attributes={"name": f, "category": "structure"},
                    )
                )
                result.relations.append(
                    Relation(source_id=parent_id, target_id=file_id, relation_type="contains")
                )

        return result


class MarkdownExtractor(BaseExtractor):
    """Parses markdown files to extract headers, tags, wikilinks, and external links."""

    def extract(self, file_path: str) -> StructuralResult:
        result = StructuralResult()
        rel_path = self.get_relative_path(file_path)
        doc_id = f"file::{rel_path}"

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            logger.warning("MarkdownExtractor failed to read %s: %s", file_path, e)
            return result

        # 1. Header structural hierarchy
        # Pattern for headings, e.g. "# Heading Text"
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        headings = list(heading_pattern.finditer(content))

        # We keep a stack of (level, section_id)
        stack: list[tuple[int, str]] = []
        level_counters = {i: 0 for i in range(1, 7)}

        for m in headings:
            level = len(m.group(1))
            heading_text = m.group(2).strip()

            # Increment count for this level
            level_counters[level] += 1
            section_id = f"section::{rel_path}#h{level}_{level_counters[level]}"

            # Add section entity
            result.entities.append(
                Entity(
                    id=section_id,
                    label=heading_text,
                    entity_type="section",
                    attributes={"name": heading_text, "category": "knowledge", "level": level},
                )
            )

            # Pop stack until we find a parent level
            while stack and stack[-1][0] >= level:
                stack.pop()

            parent_id = stack[-1][1] if stack else doc_id
            result.relations.append(
                Relation(source_id=parent_id, target_id=section_id, relation_type="contains")
            )

            stack.append((level, section_id))

        # 2. Extract Tags (e.g. #tag_name)
        # Match word boundaries but exclude code blocks/inline code (simplified regex fallback)
        tag_pattern = re.compile(r"(?<!\w)#([a-zA-Z][a-zA-Z0-9_\-/]*)")
        tags = set(tag_pattern.findall(content))
        for tag in tags:
            tag_id = f"tag::{tag.lower()}"
            result.entities.append(
                Entity(
                    id=tag_id,
                    label=f"#{tag}",
                    entity_type="tag",
                    attributes={"name": f"#{tag}", "category": "knowledge"},
                )
            )
            result.relations.append(
                Relation(source_id=doc_id, target_id=tag_id, relation_type="references")
            )

        # 3. Extract Wikilinks (e.g. [[Target Page]])
        wikilink_pattern = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]*))?\]\]")
        for match in wikilink_pattern.finditer(content):
            target_label = match.group(1).strip()
            # Construct a normalized ID for the target file
            # e.g., replacing spaces with underscores and lowercasing
            slug = target_label.lower().replace(" ", "_")
            if not slug.endswith(".md"):
                slug += ".md"
            target_id = f"file::{slug}"

            # Create placeholder target page if not already exists (will be resolved/checked later)
            result.entities.append(
                Entity(
                    id=target_id,
                    label=target_label,
                    entity_type="document",
                    attributes={"name": target_label, "category": "structure", "placeholder": True},
                )
            )

            result.relations.append(
                Relation(source_id=doc_id, target_id=target_id, relation_type="references")
            )

        # 4. Extract External Links (e.g. [Label](https://...))
        url_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
        for match in url_pattern.finditer(content):
            label = match.group(1).strip()
            url = match.group(2).strip()
            url_id = f"url::{url}"

            result.entities.append(
                Entity(
                    id=url_id,
                    label=url,
                    entity_type="url",
                    attributes={"name": label, "category": "external", "url": url},
                )
            )

            result.relations.append(
                Relation(source_id=doc_id, target_id=url_id, relation_type="links_to")
            )

        return result


class JsonExtractor(BaseExtractor):
    """Parses JSON data to extract structural keys, values, and environment references."""

    def extract(self, file_path: str) -> StructuralResult:
        result = StructuralResult()
        rel_path = self.get_relative_path(file_path)
        doc_id = f"file::{rel_path}"

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("JsonExtractor failed to read/parse %s: %s", file_path, e)
            return result

        if isinstance(data, dict):
            self._parse_dict(data, doc_id, "", rel_path, result)
        elif isinstance(data, list):
            self._parse_list(data, doc_id, "", rel_path, result)

        return result

    def _parse_dict(
        self, d: dict, parent_id: str, key_prefix: str, rel_path: str, result: StructuralResult
    ) -> None:
        env_pattern = re.compile(r"\$\{?([a-zA-Z_][a-zA-Z0-9_]*)\}?")

        for k, v in d.items():
            key_path = f"{key_prefix}.{k}" if key_prefix else k
            prop_id = f"property::{rel_path}#{key_path}"

            # Create property node
            prop_entity = Entity(
                id=prop_id,
                label=k,
                entity_type="property",
                attributes={"name": k, "category": "code", "path": key_path},
            )

            # Check value & set attributes / references
            if isinstance(v, (str, int, float, bool)) or v is None:
                prop_entity.attributes["value"] = str(v)
                if isinstance(v, str):
                    matches = env_pattern.findall(v)
                    for env_var in matches:
                        env_id = f"env::{env_var}"
                        result.entities.append(
                            Entity(
                                id=env_id,
                                label=env_var,
                                entity_type="property",
                                attributes={
                                    "name": env_var,
                                    "category": "external",
                                    "env_var": True,
                                },
                            )
                        )
                        result.relations.append(
                            Relation(source_id=prop_id, target_id=env_id, relation_type="refers_to")
                        )

            result.entities.append(prop_entity)
            result.relations.append(
                Relation(source_id=parent_id, target_id=prop_id, relation_type="contains")
            )

            # Recurse
            if isinstance(v, dict):
                self._parse_dict(v, prop_id, key_path, rel_path, result)
            elif isinstance(v, list):
                self._parse_list(v, prop_id, key_path, rel_path, result)

    def _parse_list(
        self, lst: list, parent_id: str, key_prefix: str, rel_path: str, result: StructuralResult
    ) -> None:
        for idx, item in enumerate(lst):
            key_path = f"{key_prefix}[{idx}]"
            prop_id = f"property::{rel_path}#{key_path}"

            if isinstance(item, dict):
                # Create wrapper entity for list item
                result.entities.append(
                    Entity(
                        id=prop_id,
                        label=f"[{idx}]",
                        entity_type="property",
                        attributes={"name": f"[{idx}]", "category": "code", "path": key_path},
                    )
                )
                result.relations.append(
                    Relation(source_id=parent_id, target_id=prop_id, relation_type="contains")
                )
                self._parse_dict(item, prop_id, key_path, rel_path, result)


class YamlExtractor(BaseExtractor):
    """Parses YAML data using PyYAML and routes structure identically to JSON."""

    def extract(self, file_path: str) -> StructuralResult:
        result = StructuralResult()
        if not _HAS_YAML:
            logger.warning("YamlExtractor active but pyyaml package is missing")
            return result

        rel_path = self.get_relative_path(file_path)
        doc_id = f"file::{rel_path}"

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.warning("YamlExtractor failed to read/parse %s: %s", file_path, e)
            return result

        # Delegate parsing to JsonExtractor logic since structure is identical
        json_ext = JsonExtractor(self.root_path)
        if isinstance(data, dict):
            json_ext._parse_dict(data, doc_id, "", rel_path, result)
        elif isinstance(data, list):
            json_ext._parse_list(data, doc_id, "", rel_path, result)

        return result


class PythonExtractor(BaseExtractor):
    """Parses Python source code using AST to extract modules, classes, functions, and imports."""

    class PythonASTVisitor(ast.NodeVisitor):
        def __init__(self, relative_path: str):
            self.relative_path = relative_path
            self.entities: list[Entity] = []
            self.relations: list[Relation] = []
            # Stack of active container IDs
            self.container_stack = [f"file::{relative_path}"]

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            class_name = node.name
            class_id = f"class::{self.relative_path}#{class_name}"
            parent_id = self.container_stack[-1]

            # Create class entity
            self.entities.append(
                Entity(
                    id=class_id,
                    label=class_name,
                    entity_type="class",
                    attributes={"name": class_name, "category": "code", "line": node.lineno},
                )
            )

            # contains relation
            self.relations.append(
                Relation(
                    source_id=parent_id,
                    target_id=class_id,
                    relation_type="contains",
                    metadata={"line": node.lineno},
                )
            )

            # Inheritance relations
            for base in node.bases:
                base_name = self._resolve_ast_name(base)
                if base_name:
                    # Temporary ID for base class (to be resolved to actual ID in 2nd pass)
                    base_id = f"class::placeholder#{base_name}"
                    self.entities.append(
                        Entity(
                            id=base_id,
                            label=base_name,
                            entity_type="class",
                            attributes={"name": base_name, "category": "code", "placeholder": True},
                        )
                    )
                    self.relations.append(
                        Relation(
                            source_id=class_id,
                            target_id=base_id,
                            relation_type="inherits",
                            metadata={"line": node.lineno},
                        )
                    )

            # Recurse nesting
            self.container_stack.append(class_id)
            self.generic_visit(node)
            self.container_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            func_name = node.name
            parent_id = self.container_stack[-1]

            if parent_id.startswith("class::"):
                class_part = parent_id.split("#")[-1]
                func_id = f"function::{self.relative_path}#{class_part}.{func_name}"
            else:
                func_id = f"function::{self.relative_path}#{func_name}"

            # Create function entity
            self.entities.append(
                Entity(
                    id=func_id,
                    label=func_name,
                    entity_type="function",
                    attributes={"name": func_name, "category": "code", "line": node.lineno},
                )
            )

            self.relations.append(
                Relation(
                    source_id=parent_id,
                    target_id=func_id,
                    relation_type="contains",
                    metadata={"line": node.lineno},
                )
            )

            # Recurse nesting (e.g. inner functions)
            self.container_stack.append(func_id)
            self.generic_visit(node)
            self.container_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.visit_FunctionDef(node)  # type: ignore[arg-type]

        def visit_Import(self, node: ast.Import) -> None:
            parent_id = self.container_stack[-1]
            for alias in node.names:
                name = alias.name
                module_id = f"module::{name}"

                self.entities.append(
                    Entity(
                        id=module_id,
                        label=name,
                        entity_type="module",
                        attributes={"name": name, "category": "structure", "placeholder": True},
                    )
                )
                self.relations.append(
                    Relation(
                        source_id=parent_id,
                        target_id=module_id,
                        relation_type="imports",
                        metadata={"line": node.lineno},
                    )
                )

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            parent_id = self.container_stack[-1]
            module = node.module or ""
            for alias in node.names:
                name = alias.name
                # Store enough details in target_id for the 2nd pass resolver
                symbol_id = f"code::import#{module}.{name}" if module else f"code::import#{name}"

                # Check naming convention to guess type
                est_type = "class" if name and name[0].isupper() else "function"

                self.entities.append(
                    Entity(
                        id=symbol_id,
                        label=name,
                        entity_type=est_type,
                        attributes={
                            "name": name,
                            "category": "code",
                            "placeholder": True,
                            "import_module": module,
                        },
                    )
                )
                self.relations.append(
                    Relation(
                        source_id=parent_id,
                        target_id=symbol_id,
                        relation_type="imports",
                        metadata={"line": node.lineno},
                    )
                )

        def _resolve_ast_name(self, node: ast.AST) -> str | None:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                val = self._resolve_ast_name(node.value)
                return f"{val}.{node.attr}" if val else node.attr
            return None

    def extract(self, file_path: str) -> StructuralResult:
        result = StructuralResult()
        rel_path = self.get_relative_path(file_path)

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            logger.warning("PythonExtractor failed to read %s: %s", file_path, e)
            return result

        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError as se:
            logger.warning("PythonExtractor syntax error in %s: %s", file_path, se)
            return result

        visitor = self.PythonASTVisitor(rel_path)
        visitor.visit(tree)

        result.entities = visitor.entities
        result.relations = visitor.relations
        return result


class KnowledgeGraphExtractor:
    """Orchestrates ingestion of a project folder or file, resolving cross-module references."""

    def __init__(self, root_path: str):
        self.root_path = os.path.abspath(root_path)

    def extract(self) -> StructuralResult:
        """Runs the complete parsing pipeline and returns resolved entities and relations."""
        # Check if root_path is a single file
        if os.path.isfile(self.root_path):
            result = StructuralResult()
            ext = os.path.splitext(self.root_path)[1].lower()
            rel_path = os.path.basename(self.root_path)

            # Create root document node
            result.entities.append(
                Entity(
                    id=f"file::{rel_path}",
                    label=rel_path,
                    entity_type="module" if ext == ".py" else "document",
                    attributes={"name": rel_path, "category": "structure"},
                )
            )

            if ext == ".py":
                py_ext = PythonExtractor(os.path.dirname(self.root_path))
                result.merge(py_ext.extract(self.root_path))
                self._resolve_references(result, [self.root_path])
            elif ext in (".md", ".markdown"):
                md_ext = MarkdownExtractor(os.path.dirname(self.root_path))
                result.merge(md_ext.extract(self.root_path))
                self._resolve_references(result, [])
            elif ext == ".json":
                json_ext = JsonExtractor(os.path.dirname(self.root_path))
                result.merge(json_ext.extract(self.root_path))
            elif ext in (".yaml", ".yml") and _HAS_YAML:
                yaml_ext = YamlExtractor(os.path.dirname(self.root_path))
                result.merge(yaml_ext.extract(self.root_path))

            return result

        # Otherwise, walk the folder hierarchy
        # 1. Gather all files and folder structure
        folder_ext = FolderExtractor(self.root_path)
        all_results = folder_ext.extract()

        # Gather file paths
        python_files = []
        markdown_files = []
        json_files = []
        yaml_files = []

        # Find files locally in root_path (respecting skip dirs)
        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                full_path = os.path.join(root, f)
                ext = os.path.splitext(f)[1].lower()
                if ext == ".py":
                    python_files.append(full_path)
                elif ext in (".md", ".markdown"):
                    markdown_files.append(full_path)
                elif ext == ".json":
                    json_files.append(full_path)
                elif ext in (".yaml", ".yml"):
                    yaml_files.append(full_path)

        # 2. Extractor passes
        py_ext = PythonExtractor(self.root_path)
        for pf in python_files:
            all_results.merge(py_ext.extract(pf))

        md_ext = MarkdownExtractor(self.root_path)
        for mf in markdown_files:
            all_results.merge(md_ext.extract(mf))

        json_ext = JsonExtractor(self.root_path)
        for jf in json_files:
            all_results.merge(json_ext.extract(jf))

        if _HAS_YAML:
            yaml_ext = YamlExtractor(self.root_path)
            for yf in yaml_files:
                all_results.merge(yaml_ext.extract(yf))

        # 3. Two-Pass Reference Resolution
        self._resolve_references(all_results, python_files)

        # Safeguard Limits Check
        if len(all_results.entities) > MAX_ENTITIES:
            logger.warning(
                "Ingestion exceeded node limit. Truncating to %d entities.", MAX_ENTITIES
            )
            all_results.entities = all_results.entities[:MAX_ENTITIES]
        if len(all_results.relations) > MAX_RELATIONS:
            logger.warning(
                "Ingestion exceeded relation limit. Truncating to %d relations.", MAX_RELATIONS
            )
            all_results.relations = all_results.relations[:MAX_RELATIONS]

        return all_results

    def _resolve_references(self, result: StructuralResult, python_files: list[str]) -> None:
        """Second-pass resolution of module imports and class inheritance to stable graph IDs."""
        # Index non-placeholder entities by their label/display name and path
        class_entity_map: dict[str, str] = {}  # class_name -> class_id
        file_label_map: dict[str, str] = {}  # file_name (no ext or lowecase) -> file_id

        # Build indexes
        for e in result.entities:
            if e.attributes.get("placeholder"):
                continue
            if e.entity_type == "class":
                class_entity_map[e.label] = e.id
            elif e.entity_type in ("document", "module") and e.id.startswith("file::"):
                # Map e.g. "knowledge_graph" or "knowledge_graph.md"
                base_name = os.path.splitext(e.label)[0].lower()
                file_label_map[base_name] = e.id
                file_label_map[e.label.lower()] = e.id

        # Mapping functions/methods as well
        defined_ids = {e.id for e in result.entities if not e.attributes.get("placeholder")}

        # Track rewritten entity mappings
        id_rewrites: dict[str, str] = {}

        # Resolve imported symbols
        # e.g. code::import#hbllm.memory.knowledge_graph.KnowledgeGraph -> class::core/hbllm/memory/knowledge_graph.py#KnowledgeGraph
        for e in list(result.entities):
            if not e.attributes.get("placeholder"):
                continue

            # Case A: Imported Symbol
            if e.id.startswith("code::import#"):
                parts = e.id[len("code::import#") :].split(".")
                symbol_name = parts[-1]
                module_path = ".".join(parts[:-1])

                # Find file path for module_path
                resolved_file_rel = self._resolve_module_file(module_path, python_files)
                if resolved_file_rel:
                    # Let's see if we can find the symbol inside that resolved file (class or function)
                    target_class_id = f"class::{resolved_file_rel}#{symbol_name}"
                    target_func_id = f"function::{resolved_file_rel}#{symbol_name}"

                    if target_class_id in defined_ids:
                        id_rewrites[e.id] = target_class_id
                    elif target_func_id in defined_ids:
                        id_rewrites[e.id] = target_func_id
                    else:
                        # Fallback to module level target placeholder
                        id_rewrites[e.id] = f"file::{resolved_file_rel}"
                else:
                    # Standard library or external package, map to a stable generic external module node
                    if module_path:
                        ext_module_id = f"module::{module_path}"
                        id_rewrites[e.id] = ext_module_id
                        # Ensure the module exists in graph
                        result.entities.append(
                            Entity(
                                id=ext_module_id,
                                label=module_path,
                                entity_type="module",
                                attributes={
                                    "name": module_path,
                                    "category": "structure",
                                    "external": True,
                                },
                            )
                        )

            # Case B: Base class placeholders
            elif e.id.startswith("class::placeholder#"):
                base_name = e.id[len("class::placeholder#") :]
                if base_name in class_entity_map:
                    id_rewrites[e.id] = class_entity_map[base_name]

            # Case C: Wikilinks placeholders
            elif e.id.startswith("file::") and e.attributes.get("placeholder"):
                # Check if we have a real file with a matching label
                norm_lbl = e.label.lower()
                if norm_lbl in file_label_map:
                    id_rewrites[e.id] = file_label_map[norm_lbl]
                else:
                    # Try matching slug without .md
                    slug_no_ext = os.path.splitext(norm_lbl)[0]
                    if slug_no_ext in file_label_map:
                        id_rewrites[e.id] = file_label_map[slug_no_ext]

        # Apply rewrites to relations
        for r in result.relations:
            if r.source_id in id_rewrites:
                r.source_id = id_rewrites[r.source_id]
            if r.target_id in id_rewrites:
                r.target_id = id_rewrites[r.target_id]

        # Clean up overridden placeholder entities
        rewritten_source_ids = set(id_rewrites.keys())
        result.entities = [e for e in result.entities if e.id not in rewritten_source_ids]

        # De-duplicate entities by ID
        unique_entities: dict[str, Entity] = {}
        for e in result.entities:
            if e.id not in unique_entities or (
                unique_entities[e.id].attributes.get("placeholder")
                and not e.attributes.get("placeholder")
            ):
                unique_entities[e.id] = e
        result.entities = list(unique_entities.values())

        # De-duplicate relations
        unique_relations: dict[str, Relation] = {}
        for r in result.relations:
            # Recompute key if source or target changed
            key = f"{r.source_id}--{r.relation_type}-->{r.target_id}"
            if key not in unique_relations:
                unique_relations[key] = r
        result.relations = list(unique_relations.values())

    def _resolve_module_file(self, module_path: str, python_files: list[str]) -> str | None:
        """Maps dot-separated import paths (e.g. hbllm.memory) to relative workspace python file paths."""
        parts = module_path.split(".")
        sub_path = "/".join(parts) + ".py"
        sub_path_init = "/".join(parts) + "/__init__.py"

        for f in python_files:
            rel = os.path.relpath(f, self.root_path).replace("\\", "/")
            if rel.endswith(sub_path) or rel.endswith(sub_path_init):
                return rel
        return None
