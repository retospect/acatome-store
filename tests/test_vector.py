"""Tests for PgVectorIndex — the only vector backend since v1.0.0.

These tests run against a live Postgres+pgvector via the ``store``
fixture (see ``conftest.py``).  The fake embedder fixture seeds the
block-write path; a dedicated ``_FakeQueryModel`` monkeypatched onto
``PgVectorIndex._st_model`` keeps query-time SentenceTransformer
downloads out of the test run.
"""

from __future__ import annotations

import numpy as np
import pytest

from acatome_store.vector import PgVectorIndex, VectorIndex, create_index


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Deterministic stand-in for the write-path embedder."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def __call__(self, texts: list[str]) -> list[list[float]]:
        # Constant vector — we're testing plumbing, not ranking quality.
        return [[0.1] * self.dim for _ in texts]


def _install_fake_query_embedder(store) -> None:
    """Replace the query-time SentenceTransformer with a numpy fake.

    Dim must match ``store._config.embed_dim`` because the DB column
    is typed ``Vector(embed_dim)``.
    """
    dim = store._config.embed_dim or 384

    class _FakeQueryModel:
        def encode(self, text):  # noqa: ARG002 — query text unused by fake
            return np.array([0.1] * dim, dtype=np.float32)

    store.index._st_model = _FakeQueryModel()


@pytest.fixture
def fake_embedder(store):
    emb = _FakeEmbedder(dim=store._config.embed_dim or 1024)
    store.__dict__["_embedder_cache"] = emb
    return emb


# ---------------------------------------------------------------------------
# Construction / factory
# ---------------------------------------------------------------------------


class TestCreateIndex:
    def test_returns_pgvector_index(self, store):
        """Factory returns a PgVectorIndex — the only concrete backend."""
        assert isinstance(store.index, PgVectorIndex)
        assert isinstance(store.index, VectorIndex)

    def test_requires_session_factory(self):
        """Direct ``create_index()`` without a session factory raises.

        The store hands its own factory over at property-access time;
        third-party callers who want a standalone index must supply one.
        """
        from acatome_store.config import StoreConfig

        cfg = StoreConfig(_db_url="postgresql+psycopg://x/y")
        with pytest.raises(ValueError, match="session_factory"):
            create_index(cfg)


# ---------------------------------------------------------------------------
# add_blocks — writing embeddings onto existing Block rows
# ---------------------------------------------------------------------------


class TestAddBlocks:
    """``PgVectorIndex.add_blocks`` updates the embedding column on
    already-inserted block rows.  Rows must exist before — the index
    never creates them."""

    def _seed_ref_with_blocks(self, store):
        """Create a ref whose blocks have NULL embeddings."""
        dim = store._config.embed_dim or 1024
        ref_id = store.create_ref(
            slug="vec-test-1",
            corpus_id="memories",
            title="vector test",
            blocks=[{"text": "first block text", "block_type": "text"}],
        )
        # Return the dim so the test can build matching embeddings.
        return ref_id, dim

    def test_add_blocks_updates_embedding_column(self, store, fake_embedder):
        """create_ref already populates embeddings via the fake_embedder
        fixture — verify the row now has a non-NULL embedding."""
        self._seed_ref_with_blocks(store)
        # One block with an embedding → count == 1.
        from acatome_store.models import Block

        with store._Session() as session:
            n_with_emb = (
                session.query(Block).filter(Block.embedding.isnot(None)).count()
            )
        assert n_with_emb == 1

    def test_add_blocks_accepts_corpus_id_kwarg(self, store):
        """``corpus_id`` is accepted for API parity with legacy Chroma
        callers; pgvector ignores it (corpus lives on refs)."""
        # Should not raise even though corpus_id is passed.
        result = store.index.add_blocks(
            "999",  # non-existent ref — nothing to update
            blocks=[],
            corpus_id="papers",
        )
        assert result == 0

    def test_add_blocks_skips_missing_profile(self, store):
        """A block without the requested profile in its ``embeddings``
        dict is skipped — the index doesn't fabricate vectors."""
        blocks = [
            {
                "node_id": "nonexistent",
                "embeddings": {"other_profile": [0.1] * 1024},
            }
        ]
        result = store.index.add_blocks("1", blocks, profile="default")
        assert result == 0

    def test_empty_blocks_returns_zero(self, store):
        assert store.index.add_blocks("1", []) == 0


# ---------------------------------------------------------------------------
# search_text — the main query path
# ---------------------------------------------------------------------------


class TestSearchText:
    """End-to-end: seed a couple of refs, run search_text with the
    fake query embedder, verify ranking + metadata shape."""

    def _seed(self, store):
        store.create_ref(
            slug="search-alpha",
            corpus_id="memories",
            title="Alpha memory",
            blocks=[{"text": "alpha content here", "block_type": "text"}],
        )
        store.create_ref(
            slug="search-beta",
            corpus_id="memories",
            title="Beta memory",
            blocks=[{"text": "beta content there", "block_type": "text"}],
        )

    def test_search_text_returns_hits_with_metadata(self, store, fake_embedder):
        self._seed(store)
        _install_fake_query_embedder(store)
        hits = store.index.search_text("alpha", top_k=5)
        assert len(hits) >= 1
        h = hits[0]
        # Required metadata keys present
        for key in (
            "corpus_id",
            "slug",
            "ref_title",
            "node_id",
            "block_index",
            "paper_id",
            "ref_id",
        ):
            assert key in h["metadata"], f"missing {key} in metadata"

    def test_search_text_returns_empty_on_empty_index(self, store):
        _install_fake_query_embedder(store)
        hits = store.index.search_text("nothing-here", top_k=5)
        assert hits == []

    def test_search_text_respects_top_k(self, store, fake_embedder):
        self._seed(store)
        _install_fake_query_embedder(store)
        hits = store.index.search_text("alpha", top_k=1)
        assert len(hits) <= 1

    def test_search_text_default_where_filters_profile(self, store, fake_embedder):
        """With ``where=None``, the default filter restricts to
        ``profile='default'`` so we don't leak other profiles'
        vectors into the ranking."""
        self._seed(store)
        _install_fake_query_embedder(store)
        hits = store.index.search_text("alpha", top_k=5)
        for h in hits:
            assert h["metadata"]["profile"] == "default"


# ---------------------------------------------------------------------------
# count / delete_paper
# ---------------------------------------------------------------------------


class TestCount:
    def test_count_empty(self, store):
        assert store.index.count() == 0

    def test_count_reflects_embedded_blocks(self, store, fake_embedder):
        store.create_ref(
            slug="count-1",
            corpus_id="memories",
            title="count test",
            blocks=[
                {"text": "block one", "block_type": "text"},
                {"text": "block two", "block_type": "text"},
            ],
        )
        # Fake embedder populates both blocks.
        assert store.index.count() == 2


class TestDeletePaper:
    def test_delete_is_noop(self, store):
        """``delete_paper`` is a no-op on pgvector — the CASCADE from
        the papers table handles cleanup.  Test just verifies it
        doesn't raise."""
        store.index.delete_paper("anything")  # no assertion — shouldn't raise
