"""Tests for embed-on-write on direct-write corpora.

Direct-write corpora (todos, flashcards, notes, wiki, memories,
conversations) previously stored blocks with ``embedding = NULL``.  The
:class:`~acatome_store.store.Store` now computes embeddings on
:meth:`create_ref`, :meth:`update_block_text`, and :meth:`add_block`
automatically, and exposes a :meth:`backfill_embeddings` pass to
populate legacy rows.

These tests use a fake embedder so they never pull down
``sentence-transformers`` weights and run in under a second.  The fake
records the texts it was asked to embed so we can assert both the
*shape* (correct blocks embedded, skip-types omitted) and the *timing*
(embed happens on write, not lazily at query time).
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from acatome_store.models import Block, Ref

# ---------------------------------------------------------------------------
# Fake embedder
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Deterministic stand-in for a real embedding function.

    Returns a 384-dim vector per text (matches the default pgvector
    column dimension from ``_ensure_embedding_column``).  Records every
    call in ``.calls`` for assertions.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t) % 7) / 7.0] * self.dim for t in texts]

    @property
    def total_texts(self) -> int:
        return sum(len(batch) for batch in self.calls)


@pytest.fixture
def fake_embedder(store):
    """Install a fake embedder on the store and return it for inspection."""
    emb = _FakeEmbedder(dim=store._config.embed_dim or 384)
    # Bypass the lazy property — prime the cache with the fake.
    store.__dict__["_embedder_cache"] = emb
    return emb


# ---------------------------------------------------------------------------
# create_ref embeds blocks on write
# ---------------------------------------------------------------------------


class TestCreateRefEmbeds:
    def test_single_block_embedded(self, store, fake_embedder):
        ref_id = store.create_ref(
            slug="todo-test-1",
            corpus_id="todos",
            title="Ship the datasheet handler",
            blocks=[{"text": "ship the handler by friday", "block_type": "text"}],
        )
        assert isinstance(ref_id, int)
        # Embedder was called exactly once with exactly one text.
        assert fake_embedder.total_texts == 1
        assert fake_embedder.calls[0] == ["ship the handler by friday"]

    def test_multiple_blocks_batched(self, store, fake_embedder):
        store.create_ref(
            slug="wiki-test-1",
            corpus_id="wiki",
            title="Notes on X",
            blocks=[
                {"text": "first paragraph", "block_type": "text"},
                {"text": "second paragraph", "block_type": "text"},
                {"text": "third paragraph", "block_type": "text"},
            ],
        )
        # One batched call, three texts.
        assert len(fake_embedder.calls) == 1
        assert fake_embedder.calls[0] == [
            "first paragraph",
            "second paragraph",
            "third paragraph",
        ]

    def test_skip_embed_types_omitted(self, store, fake_embedder):
        store.create_ref(
            slug="mixed-1",
            corpus_id="notes",
            title="Mixed block types",
            blocks=[
                {"text": "real prose", "block_type": "text"},
                {"text": "## heading", "block_type": "section_header"},
                {"text": "E = mc^2", "block_type": "equation"},
                {"text": "copyright 2024", "block_type": "junk"},
                {"text": "more prose", "block_type": "text"},
            ],
        )
        # Only the 'text' blocks get embedded.
        assert fake_embedder.total_texts == 2
        assert fake_embedder.calls[0] == ["real prose", "more prose"]

    def test_empty_text_skipped(self, store, fake_embedder):
        store.create_ref(
            slug="empty-1",
            corpus_id="notes",
            title="Has an empty block",
            blocks=[
                {"text": "", "block_type": "text"},
                {"text": "   ", "block_type": "text"},
                {"text": "real content", "block_type": "text"},
            ],
        )
        # Only the non-empty block.
        assert fake_embedder.total_texts == 1
        assert fake_embedder.calls[0] == ["real content"]

    def test_no_blocks_no_embedder_call(self, store, fake_embedder):
        store.create_ref(
            slug="refonly-1",
            corpus_id="todos",
            title="A todo with no body",
            blocks=None,
        )
        # No blocks → no embedding work.
        assert fake_embedder.calls == []

    def test_embedder_failure_does_not_rollback_write(self, store):
        """If the embedder raises, the ref + blocks still commit."""

        class _BrokenEmbedder:
            def __call__(self, texts):
                raise RuntimeError("simulated embedder outage")

        store.__dict__["_embedder_cache"] = _BrokenEmbedder()

        ref_id = store.create_ref(
            slug="todo-broken-embed",
            corpus_id="todos",
            title="Ref survives embedder failure",
            blocks=[{"text": "hello world", "block_type": "text"}],
        )
        # Ref was created despite embedder raising.
        assert isinstance(ref_id, int)
        got = store.get("todo-broken-embed")
        assert got is not None
        assert got["slug"] == "todo-broken-embed"

    def test_embedder_none_does_not_rollback_write(self, store):
        """If the embedder is unavailable (None), writes still succeed."""
        store.__dict__["_embedder_cache"] = None

        ref_id = store.create_ref(
            slug="todo-no-embed",
            corpus_id="todos",
            title="Ref survives missing embedder",
            blocks=[{"text": "hello world", "block_type": "text"}],
        )
        assert isinstance(ref_id, int)


# ---------------------------------------------------------------------------
# update_block_text re-embeds
# ---------------------------------------------------------------------------


class TestUpdateBlockTextReembeds:
    def test_edit_triggers_reembed(self, store, fake_embedder):
        store.create_ref(
            slug="todo-edit-1",
            corpus_id="todos",
            title="Editable todo",
            blocks=[{"text": "original text", "block_type": "text"}],
        )
        # First call was the initial embed.
        assert fake_embedder.calls == [["original text"]]

        # Find the block we just created.
        with store._Session() as session:
            ref = session.execute(
                select(Ref).where(Ref.slug == "todo-edit-1")
            ).scalar_one()
            block = session.execute(
                select(Block).where(Block.ref_id == ref.id)
            ).scalar_one()
            node_id = block.node_id

        store.update_block_text("todo-edit-1", node_id, "revised text")

        # A second call with the new text.
        assert fake_embedder.calls[-1] == ["revised text"]
        assert fake_embedder.total_texts == 2


# ---------------------------------------------------------------------------
# add_block — new API
# ---------------------------------------------------------------------------


class TestAddBlock:
    def test_appends_and_embeds(self, store, fake_embedder):
        store.create_ref(
            slug="fc-context-1",
            corpus_id="flashcards",
            title="Flashcard with context",
            blocks=[{"text": "front text", "block_type": "text"}],
        )
        baseline = fake_embedder.total_texts

        node_id = store.add_block("fc-context-1", text="context note about this card")
        assert node_id.startswith("fc-context-1-b")

        # Embedder ran for the new block.
        assert fake_embedder.total_texts == baseline + 1
        assert fake_embedder.calls[-1] == ["context note about this card"]

    def test_explicit_node_id_honored(self, store, fake_embedder):
        store.create_ref(
            slug="fc-context-2",
            corpus_id="flashcards",
            title="Flashcard 2",
            blocks=[{"text": "front", "block_type": "text"}],
        )
        node_id = store.add_block(
            "fc-context-2",
            text="explicit",
            node_id="fc-context-2-note-a",
        )
        assert node_id == "fc-context-2-note-a"

    def test_duplicate_node_id_rejected(self, store, fake_embedder):
        store.create_ref(
            slug="fc-dup-1",
            corpus_id="flashcards",
            title="Dup test",
            blocks=[{"text": "front", "block_type": "text"}],
        )
        store.add_block("fc-dup-1", text="first", node_id="fc-dup-1-note-x")
        with pytest.raises(ValueError, match="already exists"):
            store.add_block("fc-dup-1", text="second", node_id="fc-dup-1-note-x")

    def test_unknown_ref_raises(self, store, fake_embedder):
        with pytest.raises(ValueError, match="Ref not found"):
            store.add_block("does-not-exist", text="hi")

    def test_skip_embed_type_not_embedded(self, store, fake_embedder):
        store.create_ref(
            slug="eq-1",
            corpus_id="notes",
            title="Equation holder",
            blocks=[{"text": "intro", "block_type": "text"}],
        )
        baseline = fake_embedder.total_texts

        store.add_block("eq-1", text="E = mc^2", block_type="equation")
        # Skip-type → embedder not called.
        assert fake_embedder.total_texts == baseline


# ---------------------------------------------------------------------------
# backfill_embeddings — legacy migration pass for pre-embed-on-write rows
# ---------------------------------------------------------------------------


class TestBackfillEmbeddings:
    def test_backfill_populates_null_rows(self, store, fake_embedder):
        # Create a ref with an embedder — rows get embedded immediately.
        store.create_ref(
            slug="already-embedded",
            corpus_id="notes",
            title="Already embedded",
            blocks=[{"text": "first row", "block_type": "text"}],
        )

        # Now simulate a legacy write: directly null out the embedding.
        with store._Session() as session:
            rows = (
                session.execute(select(Block).where(Block.ref_id == Ref.id))
                .scalars()
                .all()
            )
            for r in rows:
                r.embedding = None
            session.commit()

        # Reset the fake so we can tell backfill made the call.
        fake_embedder.calls.clear()

        result = store.backfill_embeddings()
        assert result["scanned"] >= 1
        assert result["embedded"] >= 1
        assert result["failed"] == 0
        assert fake_embedder.total_texts >= 1

    def test_backfill_dry_run(self, store, fake_embedder):
        store.create_ref(
            slug="dry-run-ref",
            corpus_id="notes",
            title="For dry run",
            blocks=[{"text": "hello", "block_type": "text"}],
        )
        # Null out so backfill has something to see.
        with store._Session() as session:
            for r in session.execute(select(Block)).scalars().all():
                r.embedding = None
            session.commit()

        before = fake_embedder.total_texts
        result = store.backfill_embeddings(dry_run=True)
        # Dry run never calls the embedder.
        assert fake_embedder.total_texts == before
        assert result["embedded"] == 0
        assert result["scanned"] >= 1

    def test_backfill_corpus_scope(self, store, fake_embedder):
        store.create_ref(
            slug="todo-scoped-1",
            corpus_id="todos",
            title="A todo",
            blocks=[{"text": "todo text", "block_type": "text"}],
        )
        store.create_ref(
            slug="note-scoped-1",
            corpus_id="notes",
            title="A note",
            blocks=[{"text": "note text", "block_type": "text"}],
        )
        # Null out both.
        with store._Session() as session:
            for r in session.execute(select(Block)).scalars().all():
                r.embedding = None
            session.commit()

        fake_embedder.calls.clear()
        result = store.backfill_embeddings(corpus_id="todos")
        # Only the todo block is scanned.
        assert result["scanned"] == 1
        assert result["embedded"] == 1
        assert fake_embedder.calls[0] == ["todo text"]
