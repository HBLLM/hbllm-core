"""
Unit test helper: MockLLM for deterministic testing.

Provides a deterministic LLMInterface-compatible object that returns 
pre-configured JSON responses based on prompt keyword matching.
This is NOT mock data in the production code — it is exclusively 
a test utility for verifiable CI assertions.
"""

from __future__ import annotations
from typing import Any


class MockLLM:
    """
    Deterministic LLM stand-in for unit tests.
    
    Routes generate_json/generate calls to keyword-matched responses 
    so test assertions can be deterministic without a real model.
    """
    
    def __init__(self, overrides: dict[str, Any] | None = None):
        self._overrides = overrides or {}

    async def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Return a deterministic string based on prompt patterns."""
        prompt_lower = prompt.lower()
        
        # Z3 translation
        if "z3" in prompt_lower or "theorem prover" in prompt_lower:
            return (
                "from z3 import *\n"
                "x = Int('x')\n"
                "s = Solver()\n"
                "s.add(x > 0, x < 10)\n"
                "if s.check() == sat:\n"
                "    result = f'x = {s.model()[x]}'\n"
                "else:\n"
                "    result = 'unsatisfiable'\n"
            )
        
        # OpenAPI Schema
        if "openapi" in prompt_lower or "json schema" in prompt_lower or "tool" in prompt_lower:
            return '{"name": "test_tool", "parameters": {"type": "object", "properties": {"input": {"type": "string"}}}}'
        
        # Generic fallback
        if key := self._overrides.get("generate"):
            return key
        return "Generated response from MockLLM."
    
    async def generate_stream(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7):
        """Yield tokens one at a time from the generate result."""
        result = await self.generate(prompt, max_tokens, temperature)
        for word in result.split(" "):
            yield word + " "
    
    async def generate_json(self, prompt: str, max_tokens: int = 256) -> dict[str, Any]:
        """Return deterministic JSON based on prompt content."""
        prompt_lower = prompt.lower()
        
        # API request classification — must be checked BEFORE generic patterns
        if "api schema" in prompt_lower or "json payload" in prompt_lower or "openapi specification" in prompt_lower or "tool definition" in prompt_lower:
            # Extract the actual query text between the quotes after "Query:"
            import re
            query_match = re.search(r'query:\s*"([^"]+)"', prompt_lower)
            if query_match:
                actual_query = query_match.group(1)
                if any(kw in actual_query for kw in ["openapi", "api", "rest", "json", "schema", "endpoint", "tool"]):
                    return {"is_api_request": True, "request_type": "schema"}
            return {"is_api_request": False, "request_type": "none"}
        
        # Intent classification
        if "classify" in prompt_lower and "intent" in prompt_lower:
            if "biology" in prompt_lower or "quantum" in prompt_lower:
                return {"domain": "science", "intent": "unknown_topic", "confidence": 0.1}
            if "code" in prompt_lower or "python" in prompt_lower:
                return {"domain": "coding", "intent": "code_generation", "confidence": 0.85}
            return {"domain": "general", "intent": "general_knowledge", "confidence": 0.6}
        
        # Logic classification
        if "logical deduction" in prompt_lower or "theorem proving" in prompt_lower:
            if "prove" in prompt_lower or "constraint" in prompt_lower or "greater" in prompt_lower:
                return {"is_logical": True, "reason": "Contains formal constraint"}
            return {"is_logical": False, "reason": "Not a logic problem"}
        
        # Fuzzy classification
        if "fuzzy" in prompt_lower and "subjective" in prompt_lower:
            if "quality" in prompt_lower or "rate" in prompt_lower or "somewhat" in prompt_lower:
                return {"is_fuzzy": True, "reason": "Subjective judgment detected"}
            return {"is_fuzzy": False, "reason": "Not fuzzy"}
        
        # Fuzzy extraction
        if "fuzzy logic expert" in prompt_lower:
            return {
                "antecedents": [{"name": "quality", "range": [0, 10], "value": 7}],
                "consequent": {"name": "score", "range": [0, 25]},
                "rules": [{"if": "quality is high", "then": "high"}]
            }
        
        # Critic evaluation — inspect ONLY the Proposed Response, not the full prompt
        if "qa evaluator" in prompt_lower:
            import re
            response_match = re.search(r'proposed response:\s*"([^"]+)"', prompt_lower)
            if response_match:
                response_text = response_match.group(1)
                if "as an ai" in response_text or "i don't know" in response_text:
                    return {"verdict": "FAIL", "reason": "AI disclaimer deflection"}
            return {"verdict": "PASS", "reason": "Response is relevant and grounded"}
        
        # Safety classification
        if "safety classifier" in prompt_lower:
            if "dangerous" in prompt_lower or "harmful" in prompt_lower:
                return {"safe": False, "reason": "Content contains harmful instructions"}
            return {"safe": True, "reason": "Content is safe"}
        
        # Topic extraction
        if "academic" in prompt_lower or "domain" in prompt_lower:
            if "biology" in prompt_lower:
                return {"topic": "biology"}
            return {"topic": "general"}
        
        # Overrides
        if override := self._overrides.get("generate_json"):
            return override
        
        return {"result": "mock"}
