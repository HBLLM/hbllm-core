"""Multilingual language packages and runtime for HBLLM Brain."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _register_language_lexicons() -> None:
    try:
        from hbllm.brain.language.sinhala.lexicon import SinhalaLexicon
        from hbllm.brain.language.tamil.lexicon import TamilLexicon
        from hbllm.hcir.kernel.governance.governance_engine import register_lexicon_provider

        register_lexicon_provider("si", SinhalaLexicon)
        register_lexicon_provider("ta", TamilLexicon)
    except Exception as e:
        logger.debug("Failed registering default language lexicons: %s", e)


_register_language_lexicons()
