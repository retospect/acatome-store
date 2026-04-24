"""Vector index for semantic search over blocks.

Postgres + pgvector only as of v1.0.0.  The old Chroma backend is gone
— embeddings live on the ``blocks.embedding`` column via pgvector and
ANN search is a normal SQLAlchemy query.  Zero text duplication, one
real database.

If you need to run against a non-Postgres target (tests, local toy
setups), stand up a Postgres container with pgvector and point
``StoreConfig._db_url`` at it.  See ``CHANGELOG.md`` for the 0.9→1.0
migration.
"""

from __future__ import annotations

from typing import Any

from acatome_store.config import StoreConfig


# ---------------------------------------------------------------------------
# Interface — kept as an abstract base so tests and alternate backends
# can still satisfy the contract.
# ---------------------------------------------------------------------------


class VectorIndex:
    """Backend-agnostic vector index for block-level search."""

    def add_blocks(
        self,
        paper_id: str,
        blocks: list[dict[str, Any]],
        profile: str = "default",
        corpus_id: str | None = None,
    ) -> int:
        raise NotImplementedError

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        profile: str = "default",
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def delete_paper(self, paper_id: str) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Postgres backend (SQLAlchemy + pgvector)
# ---------------------------------------------------------------------------


class PgVectorIndex(VectorIndex):
    """pgvector-backed vector index — queries the blocks table directly.

    Zero text duplication: embeddings live alongside block text in one
    table, ANN search is a normal SQLAlchemy query.
    """

    def __init__(self, session_factory, embed_model: str = "all-MiniLM-L6-v2"):
        self._session_factory = session_factory
        self._embed_model_name = embed_model

    def add_blocks(
        self,
        paper_id: str,
        blocks: list[dict[str, Any]],
        profile: str = "default",
        corpus_id: str | None = None,  # noqa: ARG002 — kept for API parity
    ) -> int:
        """Store embeddings on the blocks table.

        Blocks must already exist in the DB (inserted by Store.ingest).
        This method only updates the ``embedding`` column.

        ``corpus_id`` is accepted for signature parity with prior
        backends but ignored — the corpus lives on ``refs.corpus_id``
        and is JOINed at query time.
        """
        from acatome_store.models import Block

        count = 0
        with self._session_factory() as session:
            for block in blocks:
                emb_dict = block.get("embeddings", {})
                if profile not in emb_dict:
                    continue
                emb = emb_dict[profile]
                if not emb:
                    continue

                node_id = block["node_id"]
                row = (
                    session.query(Block)
                    .filter_by(node_id=node_id, profile=profile)
                    .first()
                )
                if row:
                    row.embedding = emb
                    count += 1
            session.commit()
        return count

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        profile: str = "default",
    ) -> list[dict[str, Any]]:
        from acatome_store.models import Block

        with self._session_factory() as session:
            rows = (
                session.query(
                    Block,
                    Block.embedding.cosine_distance(query_embedding).label("distance"),
                )
                .filter(
                    Block.profile == profile,
                    Block.embedding.isnot(None),
                )
                .order_by("distance")
                .limit(top_k)
                .all()
            )
            return [
                {
                    "id": f"{row.Block.node_id}:{row.Block.profile}",
                    "text": row.Block.text,
                    "distance": float(row.distance),
                    "metadata": {
                        "paper_id": row.Block.paper_id,
                        "node_id": row.Block.node_id,
                        "block_index": row.Block.block_index or 0,
                        "page": row.Block.page,
                        "type": row.Block.block_type,
                        "profile": row.Block.profile,
                        "section_path": row.Block.section_path or "[]",
                    },
                }
                for row in rows
            ]

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search by text: embed query via sentence-transformers, then ANN.

        The result set always JOINs to ``refs`` so every hit carries its
        ``corpus_id`` + ``slug`` + ``title`` — enough for cross-corpus
        callers (``type='all'``) to render hits without a per-hit
        follow-up ``get()``.

        The ``where`` dict accepts Chroma-style filter keys (legacy
        naming kept for API parity with pre-1.0 callers):

        - ``paper_id`` — filters by ``Block.ref_id`` (legacy name).
        - ``profile`` / ``block_type`` / ``type`` — Block columns.
        - ``corpus_id`` — filters by ``Ref.corpus_id``.  Scalar
          (``corpus_id='papers'``) or list (``{'$in': [...]}``).

        When ``where`` is omitted, the default filter
        ``Block.profile == 'default'`` is applied so we only search the
        default embedding profile.
        """
        from sentence_transformers import SentenceTransformer

        if not hasattr(self, "_st_model"):
            self._st_model = SentenceTransformer(self._embed_model_name)
        emb = self._st_model.encode(query).tolist()

        # Build SQLAlchemy filters from the where dict.
        from acatome_store.models import Block, Ref

        filters = [Block.embedding.isnot(None)]
        if where:
            for key, val in where.items():
                if key == "corpus_id":
                    # Ref-level filter — handled via the JOIN below.
                    if isinstance(val, dict) and "$in" in val:
                        items = [str(x) for x in val["$in"]]
                        filters.append(Ref.corpus_id.in_(items))
                    else:
                        filters.append(Ref.corpus_id == str(val))
                    continue
                col = self._where_col(key)
                if col is None:
                    continue
                if isinstance(val, dict) and "$in" in val:
                    items = val["$in"]
                    if key == "paper_id":
                        items = [int(x) for x in items]
                    filters.append(col.in_(items))
                else:
                    if key == "paper_id":
                        val = int(val)
                    filters.append(col == val)
        else:
            filters.append(Block.profile == "default")

        with self._session_factory() as session:
            # Always JOIN Block → Ref so every hit carries ``corpus_id``
            # + ``slug`` + ``title``.  Cheap (Block.ref_id is indexed
            # via ``idx_blocks_ref_page`` + ``idx_refs_slug``) and
            # removes an N+1 lookup at the Store.search_text layer.
            rows = (
                session.query(
                    Block,
                    Block.embedding.cosine_distance(emb).label("distance"),
                    Ref.corpus_id.label("corpus_id"),
                    Ref.slug.label("slug"),
                    Ref.title.label("ref_title"),
                )
                .join(Ref, Block.ref_id == Ref.id)
                .filter(*filters)
                .order_by("distance")
                .limit(top_k)
                .all()
            )
            return [
                {
                    "id": f"{row.Block.node_id}:{row.Block.profile}",
                    "text": row.Block.text,
                    "distance": float(row.distance),
                    "metadata": {
                        "paper_id": str(row.Block.ref_id),
                        "ref_id": row.Block.ref_id,
                        "node_id": row.Block.node_id,
                        "block_index": row.Block.block_index or 0,
                        "page": row.Block.page,
                        "block_type": row.Block.block_type,
                        "type": row.Block.block_type,
                        "profile": row.Block.profile,
                        "section_path": row.Block.section_path or "[]",
                        "corpus_id": row.corpus_id,
                        "slug": row.slug,
                        "ref_title": row.ref_title,
                    },
                }
                for row in rows
            ]

    @staticmethod
    def _where_col(key: str):
        """Map where keys to Block columns.

        ``corpus_id`` is handled separately in :meth:`search_text` via
        the Block→Ref JOIN — it is *not* a Block column.
        """
        from acatome_store.models import Block

        mapping = {
            "paper_id": Block.ref_id,
            "profile": Block.profile,
            "block_type": Block.block_type,
            "type": Block.block_type,
        }
        return mapping.get(key)

    def delete_paper(self, paper_id: str) -> None:
        # CASCADE from papers table handles this — no-op
        pass

    def count(self) -> int:
        from acatome_store.models import Block

        with self._session_factory() as session:
            return session.query(Block).filter(Block.embedding.isnot(None)).count()


# ---------------------------------------------------------------------------
# Factory — single backend since v1.0.0.
# ---------------------------------------------------------------------------


def create_index(
    config: StoreConfig,
    collection_name: str = "blocks",  # noqa: ARG001 — ignored, kept for API parity
    session_factory=None,
) -> VectorIndex:
    """Create a pgvector-backed VectorIndex.

    The ``collection_name`` kwarg is accepted for source compatibility
    with callers that hardcoded it during the Chroma era; it has no
    effect since there's no separate collection namespace on pgvector
    (the blocks table is the single index).
    """
    if session_factory is None:
        raise ValueError(
            "session_factory is required — the pgvector index shares the "
            "store's SQLAlchemy session factory.  Use Store(...).index "
            "instead of calling create_index() directly if you don't have "
            "one handy."
        )
    return PgVectorIndex(session_factory, embed_model=config.embed_model)
