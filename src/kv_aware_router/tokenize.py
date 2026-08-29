"""Turning a chat request into the token sequence the router matches on.

Why the router needs a tokenizer at all: prefix reuse happens at token
granularity, and two prompts sharing a character prefix can tokenise
differently at the boundary. Matching on text would report reuse the backend
cannot deliver.

How closely it must match the backend's tokenizer:

  for routing decisions   approximately. The router only needs the ordering to
                          hold -- a longer shared prefix here should mean a
                          longer shared prefix there. Small mismatches move the
                          match length a little and rarely change which replica
                          wins.
  for reported hit rates  exactly. If you publish a hit rate computed under a
                          different tokenizer than the backend runs, the number
                          is fiction.

So a mismatched tokenizer degrades quietly rather than failing, which is the
dangerous kind of wrong. Configure it to the model you are actually serving.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class ByteTokenizer:
    """Deterministic, offline, and not any real model's tokenizer.

    Default so the gateway runs with no downloads and tests stay hermetic. It
    produces plausible prefix structure -- identical text yields identical
    tokens -- which is enough to exercise routing, and wrong for reporting hit
    rates against a real backend.
    """

    name = "byte"

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        # fold to ~4 bytes per token so sequence lengths land in a realistic
        # range rather than being 4x too long
        return [
            int.from_bytes(data[i : i + 4].ljust(4, b"\0"), "big")
            for i in range(0, len(data), 4)
        ]


class HFTokenizer:
    """The real thing. Requires the tokenizer of the model being served."""

    def __init__(self, model_id: str) -> None:
        from tokenizers import Tokenizer as _HF

        self.name = model_id
        self._tok = _HF.from_pretrained(model_id)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=True).ids


def render_messages(messages: list[dict]) -> str:
    """Flatten chat messages into one string, stably.

    Not any model's chat template. It only has to be deterministic and
    prefix-preserving: appending a turn must extend the string rather than
    rewrite it, or the prefix tree sees a different conversation every turn.
    """
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):  # multimodal content blocks
            content = "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        parts.append(f"<|{role}|>{content}")
    return "".join(parts)
