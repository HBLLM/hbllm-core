"""
Test for Grammar-Constrained Sampler.
"""

import torch

from hbllm.model.grammar import GrammarState


def test_grammar_incremental():
    vocab_words = {
        0: "{",
        1: "}",
        2: "[",
        3: "]",
        4: ":",
        5: ",",
        6: '"',
        7: "key",
        8: "value",
        9: " ",
        10: "true",
    }

    state = GrammarState(vocab_words)
    device = torch.device("cpu")

    # 1. Initially expecting a key or close brace
    mask = state.get_logit_mask(len(vocab_words), device)
    # `{` should not be blocked since it is valid to nest or open
    # We should ensure structural characters are masked correctly
    # Let's advance state with `{`
    state.advance(0)
    assert "}" in state.stack

    # 2. Expecting a key string quote
    mask = state.get_logit_mask(len(vocab_words), device)
    assert mask[7] == float("-inf")  # "key" should be blocked because it has no quote
    assert mask[6] == 0.0  # quote is allowed

    # Open key string
    state.advance(6)
    assert state.inside_string

    # Content of key string is allowed
    state.advance(7)

    # Close key string
    state.advance(6)
    assert not state.inside_string

    # 3. Now we expect a colon
    mask = state.get_logit_mask(len(vocab_words), device)
    assert mask[4] == 0.0  # colon is allowed
    assert mask[5] == float("-inf")  # comma is blocked

    state.advance(4)  # colon

    # 4. Expecting a value
    mask = state.get_logit_mask(len(vocab_words), device)
    assert mask[6] == 0.0  # quote for value is allowed
    assert mask[10] == 0.0  # "true" is allowed
