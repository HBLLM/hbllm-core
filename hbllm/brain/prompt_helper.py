from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from hbllm.network.messages import Message, MessageType

if TYPE_CHECKING:
    from hbllm.network.bus import MessageBus

logger = logging.getLogger(__name__)


async def get_dynamic_system_prompt(bus: MessageBus, tenant_id: str, source_node_id: str) -> str:
    """
    Dynamically retrieve the system prompt, goals, constraints, and active capabilities
    of the system for the given tenant over the message bus.
    """
    persona_name = "Sentra"
    system_prompt_base = "You are Sentra, an advanced cognitive AI assistant powered by the HBLLM modular architecture."
    goals: list[str] = []
    constraints: list[str] = []

    # 1. Query identity profile from IdentityNode
    try:
        id_msg = Message(
            type=MessageType.QUERY,
            source_node_id=source_node_id,
            tenant_id=tenant_id,
            topic="identity.query",
            payload={},
        )
        id_resp = await bus.request("identity.query", id_msg, timeout=2.0)
        if id_resp.payload.get("found"):
            profile = id_resp.payload.get("profile")
            if profile:
                persona_name = profile.get("persona_name", persona_name)
                system_prompt_base = (
                    profile.get("system_prompt")
                    or f"You are {persona_name}, an advanced cognitive AI assistant powered by the HBLLM modular architecture."
                )
                goals = profile.get("goals") or []
                constraints = profile.get("constraints") or []
    except Exception as e:
        logger.warning("Failed to retrieve identity profile: %s. Using default identity.", e)

    # 2. Query active registry capabilities
    has_browser = False
    has_execution = False
    has_logic = False
    has_memory = False
    nodes = []

    try:
        discover_msg = Message(
            type=MessageType.QUERY,
            source_node_id=source_node_id,
            tenant_id=tenant_id,
            topic="registry.discover",
            payload={},
        )
        reg_resp = await bus.request("registry.discover", discover_msg, timeout=2.0)
        nodes = reg_resp.payload.get("nodes", [])
        for n in nodes:
            node_id = n.get("node_id", "")
            caps = n.get("capabilities", [])
            if "web_search" in caps or "browser" in node_id:
                has_browser = True
            if "exec" in node_id or "execution" in node_id or "shell_executor" in node_id:
                has_execution = True
            if "theorem_proving" in caps or "logic" in node_id:
                has_logic = True
            if "memory" in caps or "memory" in node_id:
                has_memory = True
    except Exception as e:
        logger.warning("Failed to discover active nodes from registry: %s", e)

    tools_list = []
    try:
        for n in nodes:
            node_id = n.get("node_id", "")
            caps = n.get("capabilities", [])
            desc = n.get("description", "") or ""

            # Skip base/cognitive/routing nodes
            if node_id in (
                "router",
                "workspace",
                "decision",
                "critic",
                "meta_node",
                "identity",
                "cognitive_awareness",
                "scheduler",
                "load_manager",
                "attention",
                "evaluation",
                "reflection",
                "process_reward",
            ):
                continue

            if not caps:
                continue

            clean_caps = [
                c
                for c in caps
                if c
                not in (
                    "routing",
                    "intent_classification",
                    "task_decomposition",
                    "graph_of_thoughts",
                    "aggregation",
                )
            ]
            if not clean_caps:
                continue

            if not desc:
                desc = f"Serves capabilities: {', '.join(clean_caps)}"

            tools_list.append(f"- `{node_id}`: {desc} (capabilities: {', '.join(clean_caps)})")
    except Exception as e:
        logger.warning("Failed to parse toolbox list from registry: %s", e)

    capabilities_parts = []
    if has_browser:
        capabilities_parts.append(
            "a BrowserNode (which allows you to browse the web and search for real-time information)"
        )
    if has_execution:
        capabilities_parts.append("an ExecutionNode (for running Python code in a secure sandbox)")
    if has_logic:
        capabilities_parts.append("a LogicNode (powered by Z3 for symbolic reasoning)")
    if has_memory:
        capabilities_parts.append("a persistent memory node")

    system_prompt = system_prompt_base
    if capabilities_parts:
        if len(capabilities_parts) == 1:
            caps_str = capabilities_parts[0]
        elif len(capabilities_parts) == 2:
            caps_str = " and ".join(capabilities_parts)
        else:
            caps_str = ", ".join(capabilities_parts[:-1]) + ", and " + capabilities_parts[-1]
        system_prompt += (
            f" You have access to various cognitive and tool modules, including {caps_str}."
        )

    # Append toolbox of dynamically registered tools/plugins
    if tools_list:
        system_prompt += "\n\nAvailable Toolbox & Capabilities:\n" + "\n".join(tools_list)
        system_prompt += (
            "\n\nTo execute a tool call, output an XML-style `<tool_call>` block anywhere in your response:\n"
            '<tool_call name="tool_name">\n'
            "{\n"
            '  "arg_name": "arg_value"\n'
            "}\n"
            "</tool_call>\n"
            "The system will execute the tool and return the output for you to synthesize."
        )

    # Append goals
    if goals:
        system_prompt += "\n\nGoals:\n- " + "\n- ".join(goals)

    # Append constraints
    if constraints:
        system_prompt += "\n\nConstraints:\n- " + "\n- ".join(constraints)

    system_prompt += " Be helpful, precise, and accurate."
    return system_prompt


# ── ChatContext Return Type ──────────────────────────────────────────────────


@dataclass
class ChatContext:
    """Structured return type for the memory recall pipeline.

    Using a dataclass instead of a tuple prevents positional bugs,
    makes test mocking easier, and supports future field additions
    without breaking callers.
    """

    recent_turns: str = ""
    recalled_context: str = ""
    summary_context: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


# ── Memory Recall Constants ──────────────────────────────────────────────────

# Global budget: ~800 tokens (~3200 chars), proportionally allocated
_TOTAL_BUDGET_CHARS = 3200  # ~800 tokens

# Proportional allocation (summaries=20%, recent=20%, semantic=50%, overhead=10%)
_SUMMARY_BUDGET_RATIO = 0.20
_RECENT_BUDGET_RATIO = 0.20
_SEMANTIC_BUDGET_RATIO = 0.50
# 10% reserved for separators, labels, and overhead

_SLIDING_WINDOW_MAX_TURNS = 10
_SEMANTIC_TOP_K = 10
_ADAPTIVE_THRESHOLD_MIN = 0.45
_ADAPTIVE_THRESHOLD_PERCENTILE = 0.70
_DEDUP_SIMILARITY_THRESHOLD = 0.92

# Recency: steeper decay (7-day half-life) with higher max bonus
# Prevents stale memories from staying significant after ~2 weeks
_RECENCY_DECAY_HALFLIFE_DAYS = 7.0
_RECENCY_MAX_BONUS = 0.15

# Low-entropy query detection (short/vague queries)
_LOW_ENTROPY_WORD_THRESHOLD = 3

_MEMORY_TYPE_BONUSES: dict[str, float] = {
    "memory_summary": 0.20,
    "reflection": 0.15,
    "goal_review": 0.15,
    "project_summary": 0.15,
    "project": 0.15,
    "fact": 0.10,
    "conversation": 0.00,
    "dream_journal": -0.05,
}

_SUMMARY_SESSION_IDS = [
    "default_session",  # memory_summary / [CONSOLIDATED MEMORY]
    "dream_journal",
    "reflection",
    "goal_review",
    "project_summary",
]


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


def _compute_recency_bonus(created_at: str | None) -> float:
    """Compute exponential recency bonus from a created_at ISO timestamp.

    Uses a 7-day half-life for faster decay — prevents stale memories
    from accumulating significance over weeks.
    """
    if not created_at:
        return 0.0
    try:
        created_dt = datetime.fromisoformat(created_at)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_since = max(0.0, (now - created_dt).total_seconds() / 86400.0)
        return _RECENCY_MAX_BONUS * math.exp(-days_since / _RECENCY_DECAY_HALFLIFE_DAYS)
    except (ValueError, TypeError, OSError):
        return 0.0


def _get_memory_type_bonus(metadata: dict[str, Any]) -> float:
    """Get the priority bonus for a memory based on its type metadata."""
    mem_type = metadata.get("type", "")
    return _MEMORY_TYPE_BONUSES.get(mem_type, 0.0)


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0.0-1.0) of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[f]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def _truncate_to_budget(text: str, budget_chars: int) -> str:
    """Truncate text to fit within a character budget."""
    if len(text) <= budget_chars:
        return text
    # Try to break at a word boundary
    truncated = text[:budget_chars]
    last_space = truncated.rfind(" ")
    if last_space > budget_chars // 2:
        truncated = truncated[:last_space]
    return truncated + "…"


def _is_low_entropy_query(query: str) -> bool:
    """Detect vague/low-signal queries like 'continue', 'why?', 'what about that?'."""
    words = [w.strip(".,!?\"'()[]{}:;") for w in query.lower().split()]
    meaningful = [w for w in words if len(w) > 2]
    return len(meaningful) < _LOW_ENTROPY_WORD_THRESHOLD


def _find_last_high_signal_message(history: list[dict[str, Any]]) -> str:
    """Find the most recent information-bearing user message from history."""
    for turn in reversed(history):
        if turn.get("role") != "user":
            continue
        content = turn.get("content", "")
        if not _is_low_entropy_query(content):
            return content
    return ""


async def get_chat_memories(
    bus: MessageBus,
    tenant_id: str,
    session_id: str,
    current_query: str,
    history: list[dict[str, Any]],
) -> ChatContext:
    """
    4-layer cognitive memory recall pipeline with unified global budget.

    Implements adaptive thresholding, priority ranking (recency decay + type bonuses),
    semantic near-deduplication, hierarchical token budgeting with hard cap,
    query drift control, and irrelevant memory suppression.

    Args:
        bus: The message bus for memory queries.
        tenant_id: Current tenant ID.
        session_id: Current session ID.
        current_query: The user's current query text.
        history: Pre-fetched list of turn dicts (user/assistant only).

    Returns:
        ChatContext with recent_turns, recalled_context, summary_context, and debug info.
    """

    # ── Budget allocation from global pool ────────────────────────────────

    summary_budget = int(_TOTAL_BUDGET_CHARS * _SUMMARY_BUDGET_RATIO)
    recent_budget = int(_TOTAL_BUDGET_CHARS * _RECENT_BUDGET_RATIO)
    semantic_budget = int(_TOTAL_BUDGET_CHARS * _SEMANTIC_BUDGET_RATIO)

    debug_info: dict[str, Any] = {
        "budgets": {
            "total": _TOTAL_BUDGET_CHARS,
            "summary": summary_budget,
            "recent": recent_budget,
            "semantic": semantic_budget,
        },
    }

    # ── Step 1: Multiple Summary Types ────────────────────────────────────

    summary_parts: list[str] = []
    summary_total_chars = 0

    for sid in _SUMMARY_SESSION_IDS:
        if summary_total_chars >= summary_budget:
            break
        try:
            req = Message(
                type=MessageType.QUERY,
                source_node_id="api_server",
                tenant_id=tenant_id,
                topic="memory.retrieve_recent",
                payload={"session_id": sid, "limit": 2, "tenant_id": tenant_id},
            )
            resp = await bus.request("memory.retrieve_recent", req, timeout=2.0)
            turns = resp.payload.get("turns", [])

            for turn in turns:
                content = turn.get("content", "")
                if not content:
                    continue
                role = turn.get("role", "system")
                if role != "system":
                    continue

                # Match legacy consolidated memory or typed summaries
                meta = turn.get("metadata", {}) or {}
                mem_type = meta.get("type", "")
                is_summary = (
                    mem_type
                    in (
                        "memory_summary",
                        "dream_journal",
                        "reflection",
                        "goal_review",
                        "project_summary",
                    )
                    or "[CONSOLIDATED MEMORY]" in content
                )
                if not is_summary:
                    continue

                remaining = summary_budget - summary_total_chars
                if remaining <= 0:
                    break
                snippet = _truncate_to_budget(content, remaining)
                summary_parts.append(snippet)
                summary_total_chars += len(snippet)
        except Exception as e:
            logger.debug("Summary retrieval for session '%s' failed: %s", sid, e)

    summary_context = "\n".join(summary_parts) if summary_parts else ""
    debug_info["summary_count"] = len(summary_parts)

    # ── Step 2: Sliding Window & Budgeting ────────────────────────────────

    recent_window = history[-_SLIDING_WINDOW_MAX_TURNS:] if history else []
    recent_parts: list[str] = []
    recent_total_chars = 0

    # Build from most recent backwards
    for turn in reversed(recent_window):
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "")
        line = f"{role}: {content}"
        if recent_total_chars + len(line) > recent_budget:
            break
        recent_parts.insert(0, line)
        recent_total_chars += len(line)

    recent_turns = "\n".join(recent_parts) if recent_parts else ""
    debug_info["recent_turns_count"] = len(recent_parts)

    # Initialize final_chars in debug_info with current counts
    debug_info["final_chars"] = {
        "summary": len(summary_context),
        "recent": len(recent_turns),
        "recalled": 0,
        "total": len(summary_context) + len(recent_turns),
    }

    # ── Step 3: Weighted Query with Drift Control ─────────────────────────

    # If the current query is low-entropy (vague), fall back to the last
    # high-signal user message to prevent noisy/useless retrieval
    effective_query = current_query
    if _is_low_entropy_query(current_query):
        fallback = _find_last_high_signal_message(history)
        if fallback:
            logger.debug(
                "[MemoryRecall] Low-entropy query '%s' — falling back to '%s'",
                current_query[:40],
                fallback[:40],
            )
            effective_query = fallback
            debug_info["query_drift_fallback"] = True

    user_turns = [t for t in history if t.get("role") == "user"]
    query_parts = [f"Current:\n{effective_query}"]
    if user_turns:
        last_user = user_turns[-1].get("content", "")
        previous_users = " ".join(t.get("content", "") for t in user_turns[-3:-1])
        recent_block = last_user
        if previous_users:
            recent_block += "\n" + previous_users
        query_parts.append(f"Recent:\n{recent_block}")

    weighted_query = "\n\n".join(query_parts)

    # ── Step 4: Semantic Search ───────────────────────────────────────────

    search_results: list[dict[str, Any]] = []
    try:
        search_msg = Message(
            type=MessageType.QUERY,
            source_node_id="api_server",
            tenant_id=tenant_id,
            topic="memory.search",
            payload={"query_text": weighted_query, "top_k": _SEMANTIC_TOP_K},
        )
        search_resp = await bus.request("memory.search", search_msg, timeout=3.0)
        search_results = search_resp.payload.get("results", [])
    except Exception as e:
        logger.warning("Semantic search failed: %s", e)

    debug_info["search_returned"] = len(search_results)

    if not search_results:
        logger.debug("[MemoryRecall] No search results — returning early")
        return ChatContext(
            recent_turns=recent_turns,
            recalled_context="",
            summary_context=summary_context,
            debug=debug_info,
        )

    # ── Step 5: Adaptive Threshold ────────────────────────────────────────

    scores = [r.get("score", 0.0) for r in search_results]
    threshold = max(
        _percentile(scores, _ADAPTIVE_THRESHOLD_PERCENTILE),
        _ADAPTIVE_THRESHOLD_MIN,
    )
    candidates = [r for r in search_results if r.get("score", 0.0) >= threshold]

    debug_info["threshold"] = round(threshold, 4)
    debug_info["candidates_after_threshold"] = len(candidates)

    if not candidates:
        logger.debug("[MemoryRecall] All results below threshold %.3f", threshold)
        return ChatContext(
            recent_turns=recent_turns,
            recalled_context="",
            summary_context=summary_context,
            debug=debug_info,
        )

    # ── Step 6: Semantic Near-Deduplication ────────────────────────────────

    seen_ids: set[str] = set()
    seen_contents: set[str] = set()
    deduplicated: list[dict[str, Any]] = []

    # Collect reference texts (recent_turns) for cross-dedup
    reference_texts = [t.get("content", "") for t in recent_window]
    # Also include summary content as reference
    reference_texts.extend(summary_parts)

    for candidate in candidates:
        cid = candidate.get("id", "")
        content = candidate.get("content", "")

        # Skip by ID
        if cid in seen_ids:
            continue

        # Skip exact string match
        if content in seen_contents:
            continue

        # Check Jaccard similarity against already-selected memories
        is_dup = False
        for existing in deduplicated:
            if (
                _jaccard_similarity(content, existing.get("content", ""))
                > _DEDUP_SIMILARITY_THRESHOLD
            ):
                is_dup = True
                break

        # Check against recent turns and summaries
        if not is_dup:
            for ref in reference_texts:
                if ref and _jaccard_similarity(content, ref) > _DEDUP_SIMILARITY_THRESHOLD:
                    is_dup = True
                    break

        if is_dup:
            continue

        seen_ids.add(cid)
        seen_contents.add(content)
        deduplicated.append(candidate)

    debug_info["deduplicated_count"] = len(deduplicated)

    if not deduplicated:
        logger.debug("[MemoryRecall] All candidates deduplicated away")
        return ChatContext(
            recent_turns=recent_turns,
            recalled_context="",
            summary_context=summary_context,
            debug=debug_info,
        )

    # ── Step 7: Priority Ranking ──────────────────────────────────────────

    ranked: list[tuple[float, dict[str, Any]]] = []
    for candidate in deduplicated:
        semantic_score = candidate.get("score", 0.0)
        metadata = candidate.get("metadata", {}) or {}
        recency_bonus = _compute_recency_bonus(metadata.get("created_at"))
        type_bonus = _get_memory_type_bonus(metadata)
        priority = semantic_score + recency_bonus + type_bonus
        ranked.append((priority, candidate))

    ranked.sort(key=lambda x: x[0], reverse=True)

    debug_info["ranked_ids"] = [r[1].get("id", "?")[:8] for r in ranked[:5]]

    # ── Step 8: Token Budgeting ───────────────────────────────────────────

    recalled_parts: list[str] = []
    recalled_total_chars = 0

    for _priority, candidate in ranked:
        content = candidate.get("content", "")
        if recalled_total_chars + len(content) > semantic_budget:
            remaining = semantic_budget - recalled_total_chars
            if remaining > 50:  # Only add if we can fit a meaningful snippet
                recalled_parts.append(_truncate_to_budget(content, remaining))
            break
        recalled_parts.append(content)
        recalled_total_chars += len(content)

    recalled_context = "\n---\n".join(recalled_parts) if recalled_parts else ""

    # ── Step 9: Hard Global Cap Enforcement ───────────────────────────────
    # Enforce that the total output (summary + recent + recalled) stays
    # within TOTAL_BUDGET_CHARS. If compression inflated the recalled context,
    # truncate oldest recalled memories first.

    total_chars = len(summary_context) + len(recent_turns) + len(recalled_context)
    if total_chars > _TOTAL_BUDGET_CHARS:
        overflow = total_chars - _TOTAL_BUDGET_CHARS
        logger.debug(
            "[MemoryRecall] Global cap exceeded by %d chars — truncating recalled context",
            overflow,
        )
        recalled_context = _truncate_to_budget(
            recalled_context, max(0, len(recalled_context) - overflow)
        )
        debug_info["global_cap_truncated"] = overflow

    debug_info["final_chars"] = {
        "summary": len(summary_context),
        "recent": len(recent_turns),
        "recalled": len(recalled_context),
        "total": len(summary_context) + len(recent_turns) + len(recalled_context),
    }

    logger.debug(
        "[MemoryRecall] Pipeline complete: %d summaries, %d recent, %d recalled, "
        "%d total chars | threshold=%.3f | query_drift=%s",
        len(summary_parts),
        len(recent_parts),
        len(recalled_parts),
        debug_info["final_chars"]["total"],
        threshold,
        debug_info.get("query_drift_fallback", False),
    )

    return ChatContext(
        recent_turns=recent_turns,
        recalled_context=recalled_context,
        summary_context=summary_context,
        debug=debug_info,
    )
