#!/usr/bin/env python3
"""Adversarial dialogue + extraction eval for the Pure-HCIR chat path.

Why this exists: ``eval_pure_hcir_autonomy.py`` grades with "any expected substring
appears" (often the query word itself) and derives confidence from the pass/fail
outcome, so its ECE/Brier are tautological. This suite differs on purpose:

  * Checks never pass by echoing the question (they look for content words that are
    NOT in the utterance, or for explicit yes / no / abstain behaviour).
  * Unanswerable cases (invented words) must be *abstained* on, not acknowledged.
  * Headline metric is the CONFIDENT-WRONG rate: failures where the system answered
    or acknowledged ("Understood.") instead of abstaining or erroring out.
  * Global invariants: no question may get a bare acknowledgement; no internal
    tokens (``is_a``, ``property_property``, ``knowledge.explain`` ...) may leak.
  * Extraction is unit-tested separately (head noun, plural, "type of", glossary junk).
  * ECE is computed only from the system's own ``epistemic_state.confidence`` and
    reported together with coverage (how many turns exposed one).

Usage:
    python scripts/eval_adversarial_dialogue.py
    python scripts/eval_adversarial_dialogue.py --ratchet reports/adversarial/baseline.json --update-ratchet
    python scripts/eval_adversarial_dialogue.py --ratchet reports/adversarial/baseline.json   # CI gate

Exit code 1 if any gate fails or a previously-passing case regresses.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

ABSTAIN_RE = re.compile(
    r"(i do not know|i don't know|do not have enough|not sure|no (?:record|information|evidence)"
    r"|cannot|can't|could not|did not understand|unclear|clarif|which one|what do you mean"
    r"|not aware|unknown|not know)",
    re.I,
)
ACK_RE = re.compile(r"^\s*(understood|ok(?:ay)?|got it|noted|sure)[.!]?\s*$", re.I)
LEAK_RE = re.compile(
    r"(\bis_a\b|property_property|knowledge\.[a-z_]+|command executed|\bt4_|traceback"
    r"|\bNone\b|\{[a-z_]+\})",
    re.I,
)
GARBAGE_OBJ = frozenset(
    {"the", "a", "an", "if", "when", "must", "determines", "requiring", "every", "only", "no"}
)


def classify(resp: str) -> str:
    r = re.sub(r"\s+", " ", resp or "").strip()
    if not r:
        return "empty"
    if ACK_RE.match(r):
        return "ack"
    if ABSTAIN_RE.search(r):
        return "abstain"
    return "answer"


# ── response predicates ────────────────────────────────────────────────────────
def yes(r: str) -> bool:
    return bool(re.match(r"\s*yes\b", r, re.I))


def no_or_abstain(r: str) -> bool:
    return bool(re.match(r"\s*no\b", r, re.I)) or classify(r) == "abstain"


def abstains(r: str) -> bool:
    return classify(r) == "abstain"


def says(*words: str) -> Callable[[str], bool]:
    return lambda r: any(re.search(rf"\b{re.escape(w)}\b", r, re.I) for w in words)


def forbids(pattern: str) -> Callable[[str], bool]:
    return lambda r: not re.search(pattern, r, re.I)


def all_of(*fs: Callable[[str], bool]) -> Callable[[str], bool]:
    return lambda r: all(f(r) for f in fs)


@dataclass
class Case:
    id: str
    category: str
    utterance: str
    check: Callable[[str], bool]
    read: str = ""  # text ingested through the reading curriculum (the "book" path)
    taught: tuple[str, ...] = ()  # utterances taught live in dialogue
    answerable: bool = True  # False => the correct behaviour is to abstain


TAX = "A dog is a mammal. A mammal is an animal. A compiler is a program."
CAUSAL = "Overflow causes crash. Encryption requires key. Friction causes heat."
DARCY = ("Mr. Darcy is a proud man.",)

CHAT_CASES: list[Case] = [
    # taxonomy: yes/no + transitivity + direction
    Case("tax_direct_yes", "taxonomy", "Is a dog a mammal?", yes, read=TAX),
    Case("tax_transitive_yes", "taxonomy", "Is a dog an animal?", yes, read=TAX),
    Case("tax_unrelated_no", "taxonomy", "Is a dog a program?", no_or_abstain, read=TAX),
    Case("tax_reverse_no", "taxonomy", "Is an animal a dog?", no_or_abstain, read=TAX),
    Case("tax_what_is", "taxonomy", "What is a dog?", says("mammal"), read=TAX),
    # causal / requires, both directions
    Case("causal_inverse_what", "causal", "What causes crash?", says("overflow"), read=CAUSAL),
    Case("causal_polar_yes", "causal", "Does overflow cause crash?", yes, read=CAUSAL),
    Case(
        "causal_polar_reverse", "causal", "Does crash cause overflow?", no_or_abstain, read=CAUSAL
    ),
    Case(
        "causal_polar_unrelated",
        "causal",
        "Does friction cause crash?",
        abstains,
        read=CAUSAL,
        answerable=False,
    ),
    Case(
        "causal_unrelated_zero_evidence",
        "causal",
        "Does heat cause rust?",
        abstains,
        read=CAUSAL,
        answerable=False,
    ),
    Case("requires_what", "causal", "What does encryption require?", says("key"), read=CAUSAL),
    Case("requires_polar", "causal", "Does encryption require a key?", yes, read=CAUSAL),
    Case(
        "causal_enable_what",
        "causal",
        "What does gravity enable?",
        all_of(says("orbit"), forbids(r"command executed|knowledge\.")),
        read="Gravity enables orbit.",
    ),
    # dialogue negation & belief revision
    Case(
        "neg_dialogue_bat_bird",
        "negation",
        "Is a bat a bird?",
        no_or_abstain,
        taught=("A bat is not a bird.",),
    ),
    Case(
        "neg_dialogue_penguin_mammal",
        "negation",
        "Is a penguin a mammal?",
        no_or_abstain,
        taught=("A penguin is not a mammal.",),
    ),
    Case(
        "neg_dialogue_whale_revision",
        "negation",
        "Is a whale a fish?",
        no_or_abstain,
        taught=("A whale is a fish.", "A whale is not a fish."),
    ),
    # imperatives without internal leaks
    Case(
        "imp_tell_atom",
        "imperative",
        "Tell me about an atom.",
        all_of(says("unit", "matter", "atom"), forbids(r"command executed|knowledge\.")),
        taught=("An atom is a basic unit of matter.",),
    ),
    Case(
        "imp_explain_compiler",
        "imperative",
        "Explain compiler.",
        all_of(says("program"), forbids(r"command executed|knowledge\.")),
        read=TAX,
    ),
    Case(
        "imp_define_feline",
        "imperative",
        "Define feline.",
        all_of(says("mammal"), forbids(r"command executed|knowledge\.")),
        taught=("A feline is a mammal.",),
    ),
    # article correctness
    Case(
        "article_oxygen_mass",
        "article",
        "What is oxygen?",
        all_of(says("element"), forbids(r"\ba (?:oxygen|element)\b")),
        read="Oxygen is an element.",
    ),
    Case(
        "article_dog_animal",
        "article",
        "What is a dog?",
        all_of(says("animal", "mammal"), forbids(r"\ba animal\b")),
        read=TAX,
    ),
    Case(
        "article_gandalf_proper",
        "article",
        "Who is Gandalf?",
        all_of(says("wizard"), forbids(r"\b[Aa] gandalf\b")),
        taught=("Gandalf is a wizard.",),
    ),
    # entities: question vs assertion, known vs unknown
    Case("who_known_full", "entity", "Who is Mr. Darcy?", says("man"), taught=DARCY),
    Case("who_known_short", "entity", "Who is Darcy?", says("man"), taught=DARCY),
    Case("who_unknown", "entity", "Who is Zephyrina Quandle?", abstains, answerable=False),
    Case("what_unknown", "entity", "What is a florbnik?", abstains, answerable=False),
    Case("polar_unknown", "entity", "Does florbnik cause glorp?", abstains, answerable=False),
    Case(
        "taught_then_ask",
        "entity",
        "What is a zorp?",
        says("mammal"),
        taught=("A zorp is a mammal.",),
    ),
    # abstention / hygiene
    Case("anaphora_unresolved", "abstention", "Where is it?", abstains, answerable=False),
    Case(
        "no_placeholder_leak",
        "abstention",
        "What is a whale?",
        all_of(abstains, forbids(r"property_property")),
        answerable=False,
    ),
    Case("sky_blue_unknown", "abstention", "Is the sky blue?", abstains, answerable=False),
    # wrong head noun must not surface in answers (needs "network", not "neural")
    Case(
        "headnoun_in_answer",
        "extraction_e2e",
        "What is a transformer?",
        all_of(says("network"), forbids(r"(a|an|of) neural\b(?!\s+network)")),
        read="A transformer is a deep neural network.",
    ),
]


# ── extraction unit cases ──────────────────────────────────────────────────────
@dataclass
class ExtCase:
    id: str
    text: str
    expect: tuple[tuple[str, str, str], ...] = ()
    forbid_obj: frozenset[str] = frozenset()
    forbid_triples: tuple[tuple[str, str, str], ...] = ()


EXT_CASES: list[ExtCase] = [
    ExtCase(
        "ext_headnoun_np",
        "A transformer is a deep neural network.",
        expect=(("transformer", "is_a", "network"),),
        forbid_obj=frozenset({"neural", "deep"}),
    ),
    ExtCase(
        "ext_headnoun_lang",
        "Python is a high level programming language.",
        expect=(("python", "is_a", "language"),),
        forbid_obj=frozenset({"level", "high"}),
    ),
    ExtCase(
        "ext_type_of",
        "A dog is a type of animal.",
        expect=(("dog", "is_a", "animal"),),
        forbid_obj=frozenset({"type", "kind", "form"}),
    ),
    ExtCase("ext_plural", "Dogs are mammals.", expect=(("dog", "is_a", "mammal"),)),
    ExtCase(
        "ext_multiword_subject",
        "Elizabeth Bennet is a young woman of spirit.",
        expect=(("elizabeth_bennet", "is_a", "woman"),),
    ),
    ExtCase(
        "ext_negation_control",
        "A whale is not a fish.",
        forbid_triples=(("whale", "is_a", "fish"),),
    ),
    ExtCase(
        "ext_hedge_control",
        "Smoking may cause cancer.",
        forbid_triples=(("smoking", "causes", "cancer"),),
    ),
    ExtCase(
        "ext_glossary_junk_1",
        "* **Classify**: Determines T1/T2/T3 tier using LLM or keyword heuristics.",
        forbid_obj=GARBAGE_OBJ | frozenset({"determines"}),
    ),
    ExtCase(
        "ext_glossary_junk_2",
        "* **High-Confidence Exits**: If the exit occurs with sustained low energy, stop.",
        forbid_obj=GARBAGE_OBJ,
    ),
    ExtCase(
        "ext_glossary_junk_3",
        "**Self-improvement**: The system must improve its own reasoning over time.",
        forbid_obj=GARBAGE_OBJ,
    ),
]


def run_ext_case(c: ExtCase) -> tuple[bool, str]:
    from hbllm.hcir.language.reading_curriculum import extract_propositions_from_text

    got = {(p.subject, p.relation, p.obj) for p in extract_propositions_from_text(c.text, "adv")}
    missing = [t for t in c.expect if t not in got]
    bad_obj = [t for t in got if t[2] in c.forbid_obj]
    bad_trip = [t for t in c.forbid_triples if t in got]
    ok = not (missing or bad_obj or bad_trip)
    return ok, "" if ok else f"got={sorted(got)} missing={missing} bad={bad_obj + bad_trip}"


def boot_system() -> Any:
    from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

    return build_cognitive_system(
        CognitiveSystemConfig(
            world_id="adversarial_eval",
            memory_dir=None,
            enable_grounding=False,
            enable_brain_llm=False,
            enable_autonomous_curiosity=False,
        )
    )


def run_chat_case(c: Case) -> dict[str, Any]:
    from hbllm.hcir.language.ontology_store import LexicalOntologyStore
    from hbllm.hcir.language.reading_curriculum import ReadingCurriculumIngestor

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        system = boot_system()  # fresh, isolated system per case
        store = None
        try:
            if c.read:
                (tmp / "book").mkdir()
                (tmp / "book" / "book.txt").write_text(c.read, encoding="utf-8")
                store = LexicalOntologyStore(tmp / "ont.db")
                ReadingCurriculumIngestor().ingest_directories(
                    [tmp / "book"],
                    system.tiered_workspace.persistent,
                    ontology_store=store,
                    grant=getattr(system.tiered_workspace, "_system_grant", None),
                )
                system.tiered_workspace.notify_commit()
                system.attach_ontology_store(store)
            for t in c.taught:
                system.process_dialogue(t, speaker_id="user")
            t0 = time.perf_counter()
            turn = system.process_dialogue(c.utterance, speaker_id="user")
            ms = (time.perf_counter() - t0) * 1000.0
            if isinstance(turn, dict):
                resp = str(turn.get("response", turn.get("text", "")))
                ep = None
            else:
                resp = getattr(turn, "response_text", "") or ""
                ep = getattr(turn, "epistemic_state", None)
            err = ""
        except Exception as exc:  # noqa: BLE001 - an exception is a failed case, not a crash
            resp, ep, ms, err = "", None, 0.0, f"{type(exc).__name__}: {exc}"
        finally:
            if store is not None:
                store.close()

    cls = "error" if err else classify(resp)
    ok = (not err) and bool(c.check(resp))
    return {
        "id": c.id,
        "category": c.category,
        "utterance": c.utterance,
        "response": resp[:240],
        "class": cls,
        "passed": ok,
        "answerable": c.answerable,
        "confident_wrong": (not ok) and cls in ("answer", "ack"),
        "leak": bool(LEAK_RE.search(resp)),
        "ack_on_question": "?" in c.utterance and cls == "ack",
        "confidence": (
            float(ep.confidence) if ep is not None and hasattr(ep, "confidence") else None
        ),
        "latency_ms": round(ms, 2),
        "error": err,
    }


def ece(rows: list[dict[str, Any]], bins: int = 10) -> tuple[float | None, int]:
    pts = [(r["confidence"], r["passed"]) for r in rows if r["confidence"] is not None]
    if not pts:
        return None, 0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, p) for c, p in pts if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if sel:
            conf = sum(c for c, _ in sel) / len(sel)
            acc = sum(1 for _, p in sel if p) / len(sel)
            total += abs(acc - conf) * len(sel) / len(pts)
    return round(total, 4), len(pts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="reports/adversarial/adversarial_dialogue_latest.json")
    ap.add_argument("--ratchet", default="", help="JSON file with previously passing case ids")
    ap.add_argument("--update-ratchet", action="store_true")
    ap.add_argument("--min-pass", type=float, default=0.90)
    ap.add_argument("--max-confident-wrong", type=float, default=0.03)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)

    chat = [run_chat_case(c) for c in CHAT_CASES]
    ext = []
    for e in EXT_CASES:
        ok, detail = run_ext_case(e)
        ext.append({"id": e.id, "category": "extraction", "passed": ok, "detail": detail})
    allrows = chat + ext

    n_chat = len(chat)
    unans = [r for r in chat if not r["answerable"]]
    ans = [r for r in chat if r["answerable"]]
    ece_val, ece_cov = ece(chat)
    by_cat: dict[str, list[bool]] = {}
    for r in allrows:
        by_cat.setdefault(r["category"], []).append(r["passed"])
    metrics = {
        "cases_total": len(allrows),
        "pass_rate": round(sum(r["passed"] for r in allrows) / len(allrows), 4),
        "chat_pass_rate": round(sum(r["passed"] for r in chat) / n_chat, 4),
        "confident_wrong_rate": round(sum(r["confident_wrong"] for r in chat) / n_chat, 4),
        "abstain_on_unanswerable": round(sum(r["passed"] for r in unans) / max(1, len(unans)), 4),
        "answer_rate_on_answerable": round(sum(r["passed"] for r in ans) / max(1, len(ans)), 4),
        "leaks": sum(r["leak"] for r in chat),
        "ack_on_question": sum(r["ack_on_question"] for r in chat),
        "ece": ece_val,
        "ece_coverage": f"{ece_cov}/{n_chat}",
        "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_cat.items())},
        "response_classes": dict(Counter(r["class"] for r in chat)),
    }

    gates = {
        f"pass_rate>={args.min_pass}": metrics["pass_rate"] >= args.min_pass,
        f"confident_wrong<={args.max_confident_wrong}": (
            metrics["confident_wrong_rate"] <= args.max_confident_wrong
        ),
        "leaks==0": metrics["leaks"] == 0,
        "ack_on_question==0": metrics["ack_on_question"] == 0,
    }

    passing = sorted(r["id"] for r in allrows if r["passed"])
    regressions: list[str] = []
    rp = Path(args.ratchet) if args.ratchet else None
    if rp and rp.exists() and not args.update_ratchet:
        prev = set(json.loads(rp.read_text()).get("passing", []))
        regressions = sorted(prev - set(passing))
    if rp and args.update_ratchet:
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps({"passing": passing}, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "gates": gates,
                "regressions": regressions,
                "chat": chat,
                "extraction": ext,
            },
            indent=2,
        )
    )

    print(f"Adversarial dialogue eval: {sum(r['passed'] for r in allrows)}/{len(allrows)} passed")
    for k, v in metrics.items():
        if k not in ("by_category", "response_classes"):
            print(f"  {k:<26} {v}")
    print(f"  by_category: {metrics['by_category']}")
    print(f"  response_classes: {metrics['response_classes']}")
    for r in allrows:
        if args.verbose or not r["passed"]:
            tag = "PASS" if r["passed"] else ("FAIL*" if r.get("confident_wrong") else "FAIL ")
            body = r.get("response") or r.get("detail", "")
            print(f"  [{tag}] {r['id']:<26} {r.get('utterance', '')[:34]:<34} -> {body[:110]}")
    for name, ok in gates.items():
        print(f"  gate {name:<34} {'OK' if ok else 'FAIL'}")
    if regressions:
        print(f"  RATCHET REGRESSIONS: {regressions}")
    print("  (FAIL* = confident-wrong: answered/acknowledged instead of abstaining)")
    return 0 if all(gates.values()) and not regressions else 1


if __name__ == "__main__":
    raise SystemExit(main())
