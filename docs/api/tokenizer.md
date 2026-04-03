---
title: "API Reference — Tokenizer"
description: "API documentation for HBLLMTokenizer with Rust BPE and tiktoken fallback."
---

# Tokenizer API

The `HBLLMTokenizer` provides a unified interface for encoding/decoding with special token support and chat templates.

## Creating a Tokenizer

```python
from hbllm.model.tokenizer import HBLLMTokenizer

# Default: tiktoken fallback
tok = HBLLMTokenizer()

# From Rust BPE vocabulary
tok = HBLLMTokenizer.from_vocab("path/to/vocab.json")

# Explicit tiktoken
tok = HBLLMTokenizer.from_tiktoken()
```

## Encoding / Decoding

```python
# Encode text to token IDs
ids = tok.encode("Hello, world!")

# With special tokens
ids = tok.encode("Hello", add_bos=True, add_eos=True)

# Decode back to text
text = tok.decode(ids)
```

## Chat Templates

Format messages into ChatML-style prompts:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]

prompt = tok.apply_chat_template(messages)

# Encode chat directly to IDs
ids = tok.encode_chat(messages)
```

## Special Tokens

| Token | Symbol | Purpose |
|---|---|---|
| BOS | `<\|bos\|>` | Beginning of sequence |
| EOS | `<\|eos\|>` | End of sequence |
| PAD | `<\|pad\|>` | Padding |
| SYSTEM | `<\|system\|>` | System message marker |
| USER | `<\|user\|>` | User message marker |
| ASSISTANT | `<\|assistant\|>` | Assistant message marker |

## Properties

| Property | Type | Description |
|---|---|---|
| `vocab_size` | `int` | Total vocabulary size including special tokens |
| `bos_id` | `int` | Token ID for beginning of sequence |
| `eos_id` | `int` | Token ID for end of sequence |
| `pad_id` | `int` | Token ID for padding |
