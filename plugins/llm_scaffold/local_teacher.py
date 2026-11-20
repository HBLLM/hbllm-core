"""Local offline HuggingFace teacher adapter (strictly outside hbllm/hcir/).

Loads a local checkpoint (e.g. ``checkpoints/Llama-3.2-1B``) with
``local_files_only=True`` and produces structured JSON lessons for
verifier-gated distillation into HCIR.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM_PROMPT = (
    "You are a structured semantic parser teaching a symbolic Cognitive OS (HCIR). "
    "Given a user utterance, output ONLY a single JSON object with keys: "
    '"response_text" (short factual reply), '
    '"claim" (concise canonical proposition), '
    '"entities" (list of {"name": str, "type": str} containing EVERY concrete noun in the utterance using its exact singular lowercase lemma as "name"), '
    '"lexicon" (list of {"lemma": str, "pos": "noun"|"verb"|"adj", "definition": str, "hypernyms": [str]}), '
    '"confidence" (float 0.0-1.0).'
)


def parse_teacher_json(raw_text: str) -> dict[str, Any]:
    """Extract and validate a structured lesson dict from raw LLM output."""
    match = _JSON_BLOCK_RE.search(raw_text)
    if not match:
        return {
            "response_text": raw_text.strip(),
            "claim": "",
            "confidence": 0.0,
            "verified": False,
        }
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "response_text": raw_text.strip(),
            "claim": "",
            "confidence": 0.0,
            "verified": False,
        }
    if not isinstance(data, dict):
        return {
            "response_text": raw_text.strip(),
            "claim": "",
            "confidence": 0.0,
            "verified": False,
        }

    claim = str(data.get("claim", "")).strip()
    response_text = str(data.get("response_text", "")).strip()
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    entities = [
        {
            "name": str(e["name"]).strip().lower(),
            "type": str(e.get("type", e["name"])).strip().lower(),
        }
        for e in (data.get("entities") or [])
        if isinstance(e, dict) and e.get("name")
    ]
    lexicon = [
        {
            "lemma": str(item["lemma"]).strip().lower(),
            "pos": str(item.get("pos", "noun")).strip().lower(),
            "definition": str(item.get("definition", "")).strip(),
            "hypernyms": [str(h).strip().lower() for h in (item.get("hypernyms") or []) if h],
        }
        for item in (data.get("lexicon") or [])
        if isinstance(item, dict) and item.get("lemma")
    ]

    # Structural verification: must have a concrete claim or entities/lexicon
    verified = bool(claim or entities or lexicon) and confidence >= 0.5
    return {
        "response_text": response_text,
        "claim": claim,
        "entities": entities,
        "lexicon": lexicon,
        "confidence": confidence,
        "verified": verified,
    }


class LocalHFTeacher:
    """Lazy offline loader for local HuggingFace checkpoints (e.g. Llama-3.2-1B)."""

    def __init__(
        self,
        checkpoint_path: Path | str = Path("checkpoints/Llama-3.2-1B"),
        max_new_tokens: int = 160,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Local teacher checkpoint not found: {self.checkpoint_path}")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._device is None:
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"

        dtype = torch.float16 if self._device in ("mps", "cuda") else torch.float32
        logger.info("Loading local teacher scaffold %s on %s", self.checkpoint_path, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.checkpoint_path),
            local_files_only=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        loaded_model: Any = AutoModelForCausalLM.from_pretrained(
            str(self.checkpoint_path),
            local_files_only=True,
            torch_dtype=dtype,
        )
        self._model = loaded_model.to(self._device)
        self._model.eval()

    def __call__(self, utterance: str) -> dict[str, Any]:
        """Generate a structured JSON lesson for *utterance*."""
        self._ensure_loaded()
        import torch

        assert self._tokenizer is not None and self._model is not None
        prompt = f"{_SYSTEM_PROMPT}\n\nUtterance: {utterance}\nJSON:\n"
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        raw = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        return parse_teacher_json(raw)
