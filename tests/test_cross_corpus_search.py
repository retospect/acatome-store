"""Tests for cross-corpus semantic search (pgvector, v1.0.0+).

The store supports filtering :meth:`Store.search_text` by one or
more corpora, either via an explicit ``where={'corpus_id': ...}``
dict or the ``corpora=[...]`` convenience kwarg.  The filter is a
JOIN to ``refs`` at query time.

Two test layers:

1. **Plumbing** — unit tests that a mocked vector index receives the
   right ``where`` dict for each caller shape.  Fast.
2. **pgvector integration** — round-trip over Postgres+pgvector
   through the ``store`` fixture.

Chroma is gone (v1.0.0 removed both Chroma and the SQLite fallback),
so the old Layer-2 Chroma-metadata tests have been deleted; the
sqlite-skip branches in Layer-3 are likewise dead and are now
plain integration tests that assume a live Postgres (the ``store``
fixture skips the whole test module if no test DB is configured).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Shared fake embedder — reused for direct-write tests in this file so
# the assertions only depend on the embedding layer being exercised, not
# on any particular model's vector.
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Deterministic stand-in that returns a 384-dim vector per text."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        # Vary the vector by text length so similar-length texts cluster
        # and we can make weak but deterministic relevance assertions.
        return [[float(len(t) % 7) / 7.0] * self.dim for t in texts]


@pytest.fixture
def fake_embedder(store):
    emb = _FakeEmbedder(dim=store._config.embed_dim or 384)
    store.__dict__["_embedder_cache"] = emb
    return emb


# ===========================================================================
# Layer 1 — Plumbing: Store.search_text corpora= kwarg
# ===========================================================================


class TestStoreSearchCorporaPlumbing:
    """Unit tests with a mocked vector index — verify the ``where`` dict
    is built correctly from the ``corpora=`` kwarg."""

    def _install_mock_index(self, store):
        mock = MagicMock()
        mock.search_text.return_value = []
        # ``Store.index`` is a lazy-init property — bypass it by seeding
        # the private cache attribute the property reads from.
        store._index = mock
        return mock

    def test_single_corpus_scalar_filter(self, store):
        mock = self._install_mock_index(store)
        store.search_text("q", top_k=5, corpora=["memories"])
        _, kwargs = mock.search_text.call_args
        assert kwargs["where"] == {"corpus_id": "memories"}

    def test_multiple_corpora_in_filter(self, store):
        mock = self._install_mock_index(store)
        store.search_text("q", top_k=5, corpora=["papers", "memories", "websites"])
        _, kwargs = mock.search_text.call_args
        assert kwargs["where"] == {
            "corpus_id": {"$in": ["papers", "memories", "websites"]}
        }

    def test_empty_corpora_list_ignored(self, store):
        mock = self._install_mock_index(store)
        store.search_text("q", top_k=5, corpora=[])
        _, kwargs = mock.search_text.call_args
        # Empty list shouldn't build a filter (would match nothing).
        assert kwargs["where"] is None

    def test_none_corpora_ignored(self, store):
        mock = self._install_mock_index(store)
        store.search_text("q", top_k=5, corpora=None)
        _, kwargs = mock.search_text.call_args
        assert kwargs["where"] is None

    def test_corpora_merged_into_existing_where(self, store):
        mock = self._install_mock_index(store)
        store.search_text(
            "q",
            top_k=5,
            where={"block_type": "text"},
            corpora=["memories"],
        )
        _, kwargs = mock.search_text.call_args
        assert kwargs["where"] == {
            "block_type": "text",
            "corpus_id": "memories",
        }

    def test_explicit_where_corpus_id_wins(self, store):
        """Caller pinned corpus_id explicitly — corpora= must not clobber it."""
        mock = self._install_mock_index(store)
        store.search_text(
            "q",
            top_k=5,
            where={"corpus_id": "papers"},
            corpora=["memories", "websites"],
        )
        _, kwargs = mock.search_text.call_args
        assert kwargs["where"] == {"corpus_id": "papers"}

    def test_top_k_forwarded(self, store):
        mock = self._install_mock_index(store)
        store.search_text("q", top_k=42, corpora=["papers"])
        _, kwargs = mock.search_text.call_args
        assert kwargs["top_k"] == 42

    def test_corpora_does_not_mutate_caller_list(self, store):
        mock = self._install_mock_index(store)
        corpora = ["papers", "memories"]
        store.search_text("q", corpora=corpora)
        # The stored filter should be an independent list.
        assert corpora == ["papers", "memories"]


# ===========================================================================
# Layer 2 — pgvector integration: JOIN produces corpus_id + slug + ref_title,
# and the ``corpora=`` filter is a real SQL IN clause.
# ===========================================================================


class TestPgVectorJoinMetadata:
    """The JOIN to Ref must hydrate corpus_id + slug + ref_title on
    every hit, and the ``corpora=`` filter must be a real SQL IN
    clause (not a client-side post-filter).

    These tests rely on the ``store`` fixture which skips the whole
    module if no test Postgres is configured — so no per-test skip
    branch is needed.
    """

    def _install_fake_query_embedder(self, store):
        """Replace the query-time SentenceTransformer with a fake so
        tests don't download weights.  Dim must match the store's
        ``embed_dim`` since the DB column is typed ``Vector(embed_dim)``
        — a mismatched fake would raise at query time.
        """
        import numpy as np

        dim = store._config.embed_dim or 384

        class _FakeModel:
            def encode(self, text):
                return np.array([0.1] * dim, dtype=np.float32)

        store.index._st_model = _FakeModel()

    def test_join_returns_corpus_id_slug_title(self, store, fake_embedder):
        store.create_ref(
            slug="note-beta",
            corpus_id="notes",
            title="Note beta title",
            blocks=[{"text": "beta notes content", "block_type": "text"}],
        )
        self._install_fake_query_embedder(store)
        hits = store.search_text("beta", top_k=5, corpora=["notes"])
        assert hits, "pgvector JOIN should return the seeded note"
        h = hits[0]
        meta = h["metadata"]
        assert meta["corpus_id"] == "notes"
        assert meta["slug"] == "note-beta"
        assert meta["ref_title"] == "Note beta title"

    def test_join_corpus_id_in_filter(self, store, fake_embedder):
        store.create_ref(
            slug="n-1",
            corpus_id="notes",
            title="n1",
            blocks=[{"text": "alpha n1", "block_type": "text"}],
        )
        store.create_ref(
            slug="t-1",
            corpus_id="todos",
            title="t1",
            blocks=[{"text": "alpha t1", "block_type": "text"}],
        )
        store.create_ref(
            slug="w-1",
            corpus_id="wiki",
            title="w1",
            blocks=[{"text": "alpha w1", "block_type": "text"}],
        )
        self._install_fake_query_embedder(store)

        hits = store.search_text(
            "alpha",
            top_k=10,
            corpora=["notes", "todos"],  # wiki excluded
        )
        corpora_hit = {h["metadata"]["corpus_id"] for h in hits}
        assert corpora_hit <= {"notes", "todos"}
        assert "wiki" not in corpora_hit

    def test_scalar_corpus_id_filter(self, store, fake_embedder):
        """Scalar corpus_id in where dict (not a list) is an equality."""
        store.create_ref(
            slug="mem-single",
            corpus_id="memories",
            title="single memory",
            blocks=[{"text": "gamma content", "block_type": "text"}],
        )
        store.create_ref(
            slug="note-single",
            corpus_id="notes",
            title="single note",
            blocks=[{"text": "gamma content", "block_type": "text"}],
        )
        self._install_fake_query_embedder(store)

        hits = store.search_text(
            "gamma",
            top_k=10,
            where={"corpus_id": "memories"},
        )
        corpora_hit = {h["metadata"]["corpus_id"] for h in hits}
        assert corpora_hit == {"memories"}
