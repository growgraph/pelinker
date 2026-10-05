"""The embedding corpus stream re-attaches the peeked first chunk without self-reference."""

from __future__ import annotations

from pelinker.embed.corpus import prepend_first


def test_prepend_first_yields_every_chunk_once() -> None:
    source = iter(["a", "b", "c"])
    first = next(source)

    assert list(prepend_first(first, source)) == ["a", "b", "c"]


def test_rebinding_the_source_name_does_not_make_the_stream_consume_itself() -> None:
    """The bug this guards: a closure over a rebound name raised at the second chunk."""
    chunks = iter(range(4))
    first = next(chunks)
    chunks = prepend_first(first, chunks)

    assert list(chunks) == [0, 1, 2, 3]
