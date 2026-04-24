"""Core Store class: ingest, search, metadata queries.

Persistence:
  - Relational data (corpora, refs, papers, blocks, links) via SQLAlchemy
    ORM (portable across SQLite, Postgres, MySQL via connection string)
  - Vector search via Chroma (LlamaIndex) or pgvector (SQLAlchemy column)

Identity model:
  - ``Corpus`` = document type registry with write policy.
  - ``Ref`` = one row per known document (paper, note, todo, wiki, etc.).
    Holds slug, DOI, S2 ID, arxiv ID, title, etc.  Auto-int PK.
    Polymorphic on ``corpus_id``.
  - ``Paper`` = ingestion receipt (1:1 optional on Ref). PDF hash,
    bundle path. Only exists when content is ingested.
  - ``Block`` = text chunks from ingested documents (FK to Ref).
  - ``Link`` = slug-based edge between refs or blocks. Replaces Citation.
  - ``LinkType`` = relation registry (cites/cited_by, annotates, etc.).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from acatome_meta.literature import (
    SKIP_EMBED_TYPES,
    EmbedderUnavailableError,
)
from acatome_meta.literature import (
    make_slug as _lit_make_slug,
)
from acatome_meta.pdf import is_garbage_title as _is_garbage_title
from precis_summary import pick_best_summary
from precis_summary.rake import telegram_precis
from sqlalchemy import create_engine, select
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session, joinedload, sessionmaker

# Pure-function helpers live in _helpers.py; re-exported here under
# their original private names for backward compatibility.
from acatome_store._helpers import (
    assert_safe_ident as _assert_safe_ident,
)
from acatome_store._helpers import (
    get_embedder as _get_embedder,
)
from acatome_store._helpers import (
    read_bundle as _read_bundle,
)
from acatome_store._helpers import (
    reembed_blocks as _reembed_blocks,
)
from acatome_store._helpers import (
    update_bundle_embeddings as _update_bundle_embeddings,
)
from acatome_store.config import StoreConfig
from acatome_store.models import (
    Base,
    Block,
    Corpus,
    Link,
    LinkType,
    Note,
    Paper,
    Ref,
    create_blocks_view,
    seed_block_types,
    seed_corpora,
    seed_link_types,
)
from acatome_store.vector import VectorIndex, create_index

log = logging.getLogger(__name__)

# Sentinel for distinguishing "not yet computed" from a cached ``None``
# in the lazy embedder property.  A plain ``None`` means "embedder is
# unavailable, don't try again"; ``_MISSING`` means "haven't tried yet".
_MISSING: Any = object()


class Store:
    """Acatome paper store.

    - SQLAlchemy ORM for relational CRUD (refs, papers, blocks, citations)
    - LlamaIndex ChromaVectorStore for Chroma vector search
    - pgvector column on blocks table for Postgres vector search
    """

    def __init__(self, config: StoreConfig | None = None):
        self._config = config or StoreConfig.from_global()
        self._index: VectorIndex | None = None
        self._ensure_dirs()
        self._init_db()

    def _ensure_dirs(self) -> None:
        self._config.store_path.mkdir(parents=True, exist_ok=True)

    def _init_db(self) -> None:
        """Create SQLAlchemy engine, session factory, and tables."""
        url = self._config.db_url
        is_pg = url.startswith("postgresql")

        if is_pg:
            try:
                from acatome_store.models import add_pgvector_column

                add_pgvector_column(self._config.embed_dim)
            except ImportError:
                pass

        self._engine = create_engine(url)

        # SQLite needs this for ON DELETE CASCADE to work
        if url.startswith("sqlite"):
            from sqlalchemy import event

            @event.listens_for(self._engine, "connect")
            def _set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

        # Auto-migrate: add missing columns for existing databases
        if is_pg:
            self._ensure_embedding_column()
        self._ensure_missing_columns(is_pg)

        # Seed lookup tables and create convenience view
        with self._Session() as session:
            seed_corpora(session)
            seed_block_types(session)
            seed_link_types(session)
        try:
            create_blocks_view(self._engine)
        except (OperationalError, ProgrammingError) as exc:
            # SQLite pre-3.44 lacks CREATE OR REPLACE VIEW; the view is a
            # convenience for human SQL queries and not required for runtime.
            log.debug("Skipped blocks_v view creation: %s", exc)

    def _ensure_missing_columns(self, is_pg: bool) -> None:
        """Add missing columns to existing tables (Postgres and SQLite).

        Identifiers are all hardcoded in the migrations list below and
        validated against :data:`_SAFE_IDENT_RE` before being used in
        SQL string interpolation.  ``text()`` with f-strings is a tool
        that could become dangerous if this function ever learns to
        read migrations from external data; the validator guards against
        that future mistake.
        """
        # (table, column, pg_type, sqlite_type)
        # SQLite doesn't support UNIQUE in ALTER TABLE ADD COLUMN,
        # so we add a separate CREATE UNIQUE INDEX step below.
        migrations = [
            ("refs", "tags", "TEXT", "TEXT"),
            (
                "refs",
                "corpus_id",
                "VARCHAR DEFAULT 'papers' REFERENCES corpora(id)",
                "VARCHAR DEFAULT 'papers'",
            ),
            ("refs", "slug", "VARCHAR UNIQUE", "VARCHAR"),
            ("refs", "published_date", "DATE", "DATE"),
            ("refs", "metadata", "TEXT", "TEXT"),
            (
                "corpora",
                "write_policy",
                "VARCHAR DEFAULT 'ingestion'",
                "VARCHAR DEFAULT 'ingestion'",
            ),
            ("notes", "origin", "VARCHAR", "VARCHAR"),
        ]
        # Validate identifiers once up-front so a typo in the migrations
        # list (or a future data-driven call site) fails loudly instead
        # of smuggling metacharacters into the SQL.
        for table, column, *_ in migrations:
            _assert_safe_ident(table, kind="table")
            _assert_safe_ident(column, kind="column")
        try:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                for table, column, pg_type, sqlite_type in migrations:
                    if is_pg:
                        row = conn.execute(
                            text(
                                "SELECT 1 FROM information_schema.columns "
                                "WHERE table_name = :table "
                                "AND column_name = :column"
                            ),
                            {"table": table, "column": column},
                        ).fetchone()
                    else:
                        rows = conn.execute(
                            text(f"PRAGMA table_info('{table}')")
                        ).fetchall()
                        if not rows:
                            continue  # table doesn't exist, skip
                        row = any(r[1] == column for r in rows)
                    if not row:
                        col_type = pg_type if is_pg else sqlite_type
                        conn.execute(
                            text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                        )
                # SQLite: add unique index for slug (replaces inline UNIQUE)
                if not is_pg:
                    try:
                        conn.execute(
                            text(
                                "CREATE UNIQUE INDEX IF NOT EXISTS "
                                "idx_refs_slug_unique ON refs(slug)"
                            )
                        )
                    except OperationalError as exc:
                        log.debug("Slug index already present: %s", exc)
                conn.commit()
        except (OperationalError, ProgrammingError) as exc:
            # Column already present or table missing — auto-migration is
            # best-effort for legacy databases; real schema errors surface
            # from create_all() above.
            log.debug("Schema auto-migration skipped: %s", exc)

    def _ensure_embedding_column(self) -> None:
        """Add embedding column and HNSW index to blocks table if missing (Postgres only)."""
        try:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                row = conn.execute(
                    text(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_name='blocks' AND column_name='embedding'"
                    )
                ).fetchone()
                if not row:
                    conn.execute(
                        text("ALTER TABLE blocks ADD COLUMN embedding vector(384)")
                    )
                # Ensure HNSW index exists for ANN search (without it,
                # every vector query does a sequential scan).
                hnsw = conn.execute(
                    text(
                        "SELECT 1 FROM pg_indexes "
                        "WHERE tablename = 'blocks' "
                        "AND indexdef LIKE '%hnsw%'"
                    )
                ).fetchone()
                if not hnsw:
                    log.info(
                        "Creating HNSW index on blocks.embedding — "
                        "this may take a few minutes on large tables"
                    )
                    conn.execute(
                        text(
                            "CREATE INDEX idx_blocks_embedding_hnsw "
                            "ON blocks USING hnsw (embedding vector_cosine_ops) "
                            "WITH (m = 16, ef_construction = 64)"
                        )
                    )
                conn.commit()
        except (OperationalError, ProgrammingError) as exc:
            # pgvector extension unavailable on this server; vector search
            # will fall back to a non-pg backend or fail explicitly when used.
            log.warning(
                "Could not provision pgvector embedding column: %s. "
                "Install the pgvector extension on the Postgres server, or "
                "set vector_backend='chroma' in acatome config.",
                exc,
            )

    @property
    def index(self) -> VectorIndex:
        """Lazy-init vector index."""
        if self._index is None:
            sf = self._Session if self._config.vector_backend == "postgres" else None
            self._index = create_index(self._config, session_factory=sf)
        return self._index

    # ------------------------------------------------------------------
    # Embedding helpers (for direct-write corpora)
    # ------------------------------------------------------------------

    @property
    def _embedder(self):
        """Lazy-init, cached embedding function.  ``None`` if unavailable.

        Builds the embedder on first access from :attr:`_config`.  If the
        embedding backend is missing (e.g. ``sentence-transformers`` not
        installed in this venv) we log a single warning and cache ``None``
        so subsequent direct-writes proceed without embeddings rather
        than raising — writes succeeding with NULL embedding is exactly
        the pre-existing behaviour.
        """
        cached = self.__dict__.get("_embedder_cache", _MISSING)
        if cached is not _MISSING:
            return cached
        try:
            fn = _get_embedder(self._config)
        except EmbedderUnavailableError as exc:
            log.warning(
                "Embedder unavailable; direct-write blocks will not be "
                "embedded automatically (use backfill_embeddings() later "
                "after installing the provider): %s",
                exc,
            )
            fn = None
        self.__dict__["_embedder_cache"] = fn
        return fn

    def _compute_block_embeddings(
        self,
        block_specs: list[tuple[str, str, str]],
    ) -> list[dict[str, Any]]:
        """Embed a batch of direct-write blocks.

        Args:
            block_specs: list of ``(node_id, text, block_type)`` tuples.

        Returns:
            List of block dicts in the shape :meth:`VectorIndex.add_blocks`
            consumes: ``{"node_id", "text", "type", "embeddings": {"default": [...]}}``.
            Blocks whose ``block_type`` is in :data:`SKIP_EMBED_TYPES` or
            whose text is empty are omitted from the result.  Returns an
            empty list if the embedder is unavailable or raises.
        """
        embedder = self._embedder
        if embedder is None or not block_specs:
            return []

        texts: list[str] = []
        keep: list[tuple[str, str, str]] = []
        for node_id, text, block_type in block_specs:
            if block_type in SKIP_EMBED_TYPES:
                continue
            t = (text or "").strip()
            if not t:
                continue
            texts.append(t)
            keep.append((node_id, text, block_type))

        if not texts:
            return []

        try:
            embeddings = embedder(texts)
        except Exception as exc:  # pragma: no cover — best-effort
            log.warning(
                "Embedding computation failed for %d direct-write block(s): %s",
                len(texts),
                exc,
            )
            return []

        return [
            {
                "node_id": node_id,
                "text": text,
                "type": block_type,
                "embeddings": {"default": emb},
            }
            for (node_id, text, block_type), emb in zip(keep, embeddings)
        ]

    def _index_direct_blocks(
        self,
        ref_id: int,
        block_specs: list[tuple[str, str, str]],
        corpus_id: str | None = None,
    ) -> int:
        """Compute + push embeddings for direct-write blocks (best-effort).

        Wraps :meth:`_compute_block_embeddings` + :meth:`VectorIndex.add_blocks`
        in try/except so indexing failures never propagate out of a write.

        ``corpus_id`` is forwarded to the vector index so per-block
        metadata (used for cross-corpus filters on Chroma) is stamped
        at index time.  The pgvector backend ignores the kwarg — it
        already joins ``blocks`` → ``refs`` at query time for the
        corpus filter.

        Returns the number of blocks successfully indexed (0 on any failure).
        """
        embedded = self._compute_block_embeddings(block_specs)
        if not embedded:
            return 0
        try:
            return self.index.add_blocks(str(ref_id), embedded, corpus_id=corpus_id)
        except Exception as exc:  # pragma: no cover — best-effort
            log.warning(
                "Failed to index %d direct-write embedding(s) for ref %s: %s",
                len(embedded),
                ref_id,
                exc,
            )
            return 0

    # ------------------------------------------------------------------
    # Ref helpers
    # ------------------------------------------------------------------

    def _find_ref(self, session: Session, header: dict) -> Ref | None:
        """Find an existing Ref by DOI, S2 ID, or arxiv ID."""
        doi = header.get("doi")
        if doi:
            ref = session.execute(
                select(Ref).where(Ref.doi == doi)
            ).scalar_one_or_none()
            if ref:
                return ref

        s2_id = header.get("s2_id")
        if s2_id:
            ref = session.execute(
                select(Ref).where(Ref.s2_id == s2_id)
            ).scalar_one_or_none()
            if ref:
                return ref

        arxiv_id = header.get("arxiv_id")
        if arxiv_id:
            ref = session.execute(
                select(Ref).where(Ref.arxiv_id == arxiv_id)
            ).scalar_one_or_none()
            if ref:
                return ref

        return None

    @staticmethod
    def _should_upgrade_ref(ref: Ref, header: dict) -> bool:
        """Decide whether stale Ref fields should be overwritten by the new bundle.

        Upgrade (replace non-null stale fields) is allowed when any of:
        - No Paper yet (first-time ingest of content for this Ref)
        - Existing Paper is ``verified=False`` and new bundle is ``verified=True``
        - Current ``title`` looks like garbage (e.g. InDesign filename artefact)
          while the new bundle provides a real title

        Otherwise we preserve existing fields (fill-blanks-only policy).
        """
        new_verified = bool(header.get("verified"))
        paper = ref.paper
        if paper is None:
            return True
        if new_verified and not bool(paper.verified):
            return True
        new_title = header.get("title") or ""
        cur_title = ref.title or ""
        if (
            cur_title
            and _is_garbage_title(cur_title)
            and new_title
            and not _is_garbage_title(new_title)
        ):
            return True
        return False

    def _refresh_ref_metadata(self, ref: Ref, header: dict) -> None:
        """Refresh Ref header-derived fields from a bundle header.

        Policy:
        - Always fill blanks (existing behavior)
        - If :meth:`_should_upgrade_ref` returns True, additionally overwrite
          stale non-null fields. This lets a later verified ingest repair a
          Ref that was created earlier with garbage/unverified metadata.

        Slug is NOT touched here — callers handle slug separately with
        collision-aware logic.
        """
        upgrade = self._should_upgrade_ref(ref, header)

        # Scalar columns: doi, s2_id, arxiv_id, title
        for field in ("doi", "s2_id", "arxiv_id", "title"):
            new_val = header.get(field)
            if not new_val:
                continue
            cur_val = getattr(ref, field, None)
            if not cur_val:
                setattr(ref, field, new_val)
            elif upgrade:
                # Title-specific guard: don't downgrade a clean title to
                # garbage just because the new bundle happens to be verified.
                if (
                    field == "title"
                    and _is_garbage_title(new_val)
                    and not _is_garbage_title(cur_val)
                ):
                    continue
                setattr(ref, field, new_val)

        # authors (stored as JSON array)
        new_authors = header.get("authors")
        if new_authors and (not ref.authors or upgrade):
            ref.authors = json.dumps(new_authors)

        # year
        new_year = header.get("year")
        if new_year and (not ref.year or upgrade):
            ref.year = new_year

        # keywords (stored as JSON array; fill-blanks only — user may curate)
        if header.get("keywords") and not ref.keywords:
            ref.keywords = json.dumps(header["keywords"])

        # Metadata JSON: journal, entry_type, source
        meta = ref._meta.copy() if ref.meta else {}
        changed = False
        for meta_field in ("journal", "entry_type", "source"):
            val = header.get(meta_field)
            if not val:
                continue
            if not meta.get(meta_field) or upgrade:
                meta[meta_field] = val
                changed = True
        if changed or (meta and not ref.meta):
            ref.meta = json.dumps(meta)

        # Upgrade Paper verification flag when the new bundle confirms
        if (
            upgrade
            and ref.paper is not None
            and header.get("verified")
            and not ref.paper.verified
        ):
            ref.paper.verified = True

    def _upsert_ref(self, session: Session, header: dict) -> Ref:
        """Find or create a Ref from bundle header, refreshing fields.

        See :meth:`_refresh_ref_metadata` for the fill-blanks + upgrade policy.
        Slug is NOT set here — ingest() handles it after collision check.
        """
        ref = self._find_ref(session, header)
        if ref is None:
            ref = Ref()
            session.add(ref)

        self._refresh_ref_metadata(ref, header)
        session.flush()  # ensure ref.id is assigned
        return ref

    @staticmethod
    def _merge_tags(ref: Ref, new_tags: list[str]) -> None:
        """Additively merge tags into a Ref (deduped, sorted)."""
        existing: list[str] = json.loads(ref.tags) if ref.tags else []
        merged = sorted(set(existing) | set(new_tags))
        ref.tags = json.dumps(merged)

    _make_slug = staticmethod(_lit_make_slug)

    @staticmethod
    def _disambiguate_slug(session: Session, base_slug: str) -> str:
        """Append a/b/c… suffix to make a slug unique."""
        for suffix in "abcdefghijklmnopqrstuvwxyz":
            candidate = f"{base_slug}{suffix}"
            exists = session.execute(
                select(Ref).where(Ref.slug == candidate)
            ).scalar_one_or_none()
            if not exists:
                return candidate
        # Extremely unlikely: 26 collisions — fall back to numeric
        n = 2
        while True:
            candidate = f"{base_slug}{n}"
            exists = session.execute(
                select(Ref).where(Ref.slug == candidate)
            ).scalar_one_or_none()
            if not exists:
                return candidate
            n += 1

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, bundle_path: str | Path, *, tags: list[str] | None = None) -> int:
        """Ingest a .acatome bundle into the store.

        Dedup rules:
        - Same pdf_hash → skip (return existing ref_id)
        - Same DOI, different PDF → skip (return existing ref_id)
        - Same slug, different ref → raise ValueError
        - Same ref, re-ingest → atomic replace of Paper + blocks

        Tags are always merged additively, even on dedup skip.

        Returns:
            ref_id (int) of the ingested paper.
        """
        bundle_path = Path(bundle_path)
        data = _read_bundle(bundle_path)
        header = data["header"]
        slug = (
            header.get("slug", "").replace("~", "").replace("\u203a", "")
        )  # separators reserved
        pdf_hash = header["pdf_hash"]

        with self._Session() as session:
            # Dedup: same pdf_hash already ingested
            existing = session.execute(
                select(Paper).where(Paper.pdf_hash == pdf_hash)
            ).scalar_one_or_none()
            if existing:
                ref = session.get(Ref, existing.ref_id)
                if ref:
                    # Refresh stale metadata (e.g. garbage title from an
                    # earlier unverified ingest) from the fresh bundle header.
                    upgrade = self._should_upgrade_ref(ref, header)
                    self._refresh_ref_metadata(ref, header)
                    # Upgrade slug when the new bundle is clearly better
                    # (unverified→verified or garbage→clean title).
                    if upgrade and slug and slug != ref.slug:
                        collision = session.execute(
                            select(Ref).where(Ref.slug == slug, Ref.id != ref.id)
                        ).scalar_one_or_none()
                        if not collision:
                            ref.slug = slug
                    if tags:
                        self._merge_tags(ref, tags)
                    session.commit()
                return existing.ref_id

            # Find or create the Ref (identity)
            ref = self._upsert_ref(session, header)

            # Dedup: ref has a Paper with a different PDF hash
            if ref.paper and ref.paper.pdf_hash != pdf_hash:
                return ref.id

            # Generate slug from metadata if bundle didn't provide one
            if not slug:
                slug = self._make_slug(
                    header.get("authors", []),
                    header.get("year"),
                    header.get("title", ""),
                )

            # Slug collision: auto-disambiguate with a/b/c… suffix
            existing_ref = session.execute(
                select(Ref).where(Ref.slug == slug)
            ).scalar_one_or_none()
            if existing_ref and existing_ref.id != ref.id:
                slug = self._disambiguate_slug(session, slug)
                log.info("slug collision → %s", slug)
            ref.slug = slug

            # Atomic replace: delete old Paper + blocks if re-ingesting
            if ref.paper:
                session.delete(ref.paper)
            # Blocks FK to ref, so delete them explicitly
            for old_block in list(ref.blocks):
                session.delete(old_block)
            session.flush()

            # Insert Paper (ingestion receipt)
            paper = Paper(
                ref_id=ref.id,
                pdf_hash=pdf_hash,
                bundle_path=str(bundle_path.resolve()),
                verified=bool(header.get("verified")),
            )
            session.add(paper)
            session.flush()

            # Insert abstract as a block (if present)
            abstract_text = header.get("abstract", "")
            if abstract_text:
                session.add(
                    Block(
                        node_id=f"ref:{ref.id}:abstract",
                        profile="default",
                        ref_id=ref.id,
                        page=None,
                        block_index=None,
                        block_type="abstract",
                        text=abstract_text,
                    )
                )

            # Insert document summary as a block (if present)
            enrich_meta = data.get("enrichment_meta") or {}
            # New format: paper_summaries dict; old format: paper_summary string
            paper_summaries = enrich_meta.get("paper_summaries") or {}
            doc_summary = pick_best_summary(paper_summaries) or enrich_meta.get(
                "paper_summary", ""
            )
            if doc_summary:
                session.add(
                    Block(
                        node_id=f"ref:{ref.id}:document_summary",
                        profile="default",
                        ref_id=ref.id,
                        page=None,
                        block_index=None,
                        block_type="document_summary",
                        text=doc_summary,
                    )
                )

            # Insert blocks (block_index is a global sequential counter)
            blocks = data.get("blocks", [])
            for block_index, b in enumerate(blocks):
                node_id = b["node_id"]

                bbox = b.get("bbox")
                block = Block(
                    node_id=node_id,
                    profile="default",
                    ref_id=ref.id,
                    page=b.get("page", 0),
                    block_index=block_index,
                    block_type=b.get("type", "text"),
                    text=b.get("text", ""),
                    summary=pick_best_summary(b.get("summaries")) or b.get("summary"),
                    section_path=json.dumps(b.get("section_path", [])),
                    bbox_x0=bbox[0] if bbox else None,
                    bbox_y0=bbox[1] if bbox else None,
                    bbox_x1=bbox[2] if bbox else None,
                    bbox_y1=bbox[3] if bbox else None,
                )
                session.add(block)

            # Aggregate block-level RAKE keywords into paper-level keywords
            if not ref.keywords:
                rake_parts = []
                for b in blocks:
                    summaries = b.get("summaries") or {}
                    rake = summaries.get("rake", "")
                    if rake:
                        rake_parts.append(rake)
                if rake_parts:
                    combined = "; ".join(rake_parts)
                    paper_kw = telegram_precis(combined, min_n=3, max_n=8)
                    ref.keywords = json.dumps(paper_kw.split("; "))

            if tags:
                self._merge_tags(ref, tags)

            session.commit()
            ref_id = ref.id

        # Check embedding model compatibility and re-embed if needed
        if blocks:
            bundle_models = enrich_meta.get("embedding_models", {})
            bundle_default = bundle_models.get("default", {})
            bundle_model = bundle_default.get("model", "")

            needs_reembed = (
                not bundle_model  # no enrichment metadata → embed from scratch
                or bundle_model != self._config.embed_model
            )

            if needs_reembed:
                log.warning(
                    "Embedding model mismatch: bundle has '%s', system expects '%s' "
                    "— re-embedding %d blocks",
                    bundle_model,
                    self._config.embed_model,
                    len(blocks),
                )
                try:
                    embedder = _get_embedder(self._config)
                    blocks = _reembed_blocks(blocks, embedder)
                    _update_bundle_embeddings(
                        bundle_path,
                        data,
                        blocks,
                        self._config.embed_model,
                        self._config.embed_dim,
                    )
                except EmbedderUnavailableError as exc:
                    raise EmbedderUnavailableError(
                        f"Cannot re-embed bundle for vector indexing: {exc}"
                    ) from exc

            self.index.add_blocks(str(ref_id), blocks, corpus_id="papers")

        return ref_id

    # ------------------------------------------------------------------
    # Direct ref creation (for direct-write corpora: todos, wiki, notes)
    # ------------------------------------------------------------------

    def create_ref(
        self,
        slug: str,
        *,
        corpus_id: str = "papers",
        title: str = "",
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        blocks: list[dict[str, Any]] | None = None,
    ) -> int:
        """Create a ref directly (no .acatome bundle needed).

        For use by direct-write corpora (todos, wiki, notes, journal).
        Raises ValueError if slug already exists or corpus not found.

        Args:
            slug: Unique slug for the ref.
            corpus_id: Corpus this ref belongs to.
            title: Human-readable title.
            metadata: JSON-serialisable dict stored in ref.metadata.
            tags: Optional list of tags.
            blocks: Optional list of block dicts with keys:
                text, block_type (default "text"), section_path (default []).

        Returns:
            ref_id (int) of the created ref.
        """
        if not slug:
            raise ValueError("slug is required for create_ref()")
        if "~" in slug or "\u203a" in slug:
            raise ValueError(
                f"slug must not contain '~' or '\u203a' (reserved as URI selector separator): {slug}"
            )

        with self._Session() as session:
            # Validate corpus exists
            corpus = session.get(Corpus, corpus_id)
            if not corpus:
                raise ValueError(f"Unknown corpus: {corpus_id}")

            # Check slug uniqueness
            existing = session.execute(
                select(Ref).where(Ref.slug == slug)
            ).scalar_one_or_none()
            if existing:
                raise ValueError(f"Slug already exists: {slug}")

            ref = Ref(
                corpus_id=corpus_id,
                slug=slug,
                title=title or None,
            )
            if metadata:
                ref.meta = json.dumps(metadata)
            session.add(ref)
            session.flush()  # get ref.id

            if tags:
                self._merge_tags(ref, tags)

            # Insert blocks (embeddings are computed after commit below —
            # we need ref.id + committed rows for index.add_blocks to find
            # them, and we don't want embedding failures to roll back the
            # ref/block inserts themselves).
            block_specs: list[tuple[str, str, str]] = []
            for i, b in enumerate(blocks or []):
                node_id = f"{slug}-b{i:04d}"
                block_type = b.get("block_type", "text")
                text = b.get("text", "")
                block = Block(
                    node_id=node_id,
                    profile="default",
                    ref_id=ref.id,
                    page=0,
                    block_index=i,
                    block_type=block_type,
                    text=text,
                    section_path=json.dumps(b.get("section_path", [])),
                )
                session.add(block)
                block_specs.append((node_id, text, block_type))

            session.commit()
            ref_id = ref.id

        # Best-effort embedding pass — direct-write corpora (todos,
        # flashcards, memories, notes, wiki, conversations) previously
        # wrote blocks with NULL embedding and were invisible to
        # semantic search.  We now populate them on write so they
        # participate in cross-corpus search immediately.  Failures
        # here are logged but do not propagate: the ref already exists.
        if block_specs:
            self._index_direct_blocks(ref_id, block_specs, corpus_id=corpus_id)
        return ref_id

    def update_ref_metadata(
        self,
        slug: str,
        metadata: dict[str, Any],
        *,
        merge: bool = True,
    ) -> None:
        """Update the metadata JSON for a ref.

        Args:
            slug: Ref slug to update.
            metadata: Dict of keys to set.
            merge: If True (default), merge with existing metadata.
                   If False, replace entirely.

        Raises ValueError if ref not found.
        """
        with self._Session() as session:
            ref = session.execute(
                select(Ref).where(Ref.slug == slug)
            ).scalar_one_or_none()
            if not ref:
                raise ValueError(f"Ref not found: {slug}")

            if merge and ref.meta:
                try:
                    existing = json.loads(ref.meta)
                except (ValueError, TypeError):
                    existing = {}
                existing.update(metadata)
                ref.meta = json.dumps(existing)
            else:
                ref.meta = json.dumps(metadata)
            session.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        corpora: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic text search via vector index.

        Args:
            query: Natural-language query to embed and rank against.
            top_k: Max number of hits to return.
            where: Chroma-style filter dict forwarded to the vector
                index.  See :meth:`VectorIndex.search_text` for the
                supported keys (``paper_id``, ``profile``,
                ``block_type``, ``corpus_id``).  Scalar or
                ``{'$in': [...]}``.
            corpora: Convenience kwarg for cross-corpus search —
                pass a list of corpus ids (``['papers', 'memories',
                'websites']``) and the call is dispatched with
                ``where={'corpus_id': {'$in': corpora}}``.  When set
                alongside ``where``, the corpus filter is merged into
                ``where``; explicit ``where['corpus_id']`` wins.

        Returns:
            list of result dicts with ``text``, ``metadata`` (now
            always including ``corpus_id`` + ``slug`` + ``ref_title``
            via the Block→Ref JOIN), ``distance``, and enriched
            ``paper`` info for papers.
        """
        # Merge ``corpora`` into ``where`` unless the caller already
        # pinned ``corpus_id`` explicitly.
        if corpora:
            where = dict(where) if where else {}
            if "corpus_id" not in where:
                where["corpus_id"] = (
                    corpora[0] if len(corpora) == 1 else {"$in": list(corpora)}
                )

        hits = self.index.search_text(query, top_k=top_k, where=where)

        for hit in hits:
            pid = hit.get("metadata", {}).get("paper_id")
            if pid:
                paper = self.get(pid)
                if paper:
                    hit["paper"] = {
                        "slug": paper.get("slug"),
                        "title": paper.get("title"),
                        "year": paper.get("year"),
                        "doi": paper.get("doi"),
                    }

        # Batch-fetch block summaries
        lookups = []
        for hit in hits:
            meta = hit.get("metadata", {})
            pid = meta.get("paper_id")
            bi = meta.get("block_index")
            if pid is not None and bi is not None:
                lookups.append((int(pid), int(bi)))
        if lookups:
            summaries = self._batch_block_summaries(lookups)
            for hit in hits:
                meta = hit.get("metadata", {})
                pid = meta.get("paper_id")
                bi = meta.get("block_index")
                if pid is not None and bi is not None:
                    s = summaries.get((int(pid), int(bi)))
                    if s:
                        hit["summary"] = s

        return hits

    def _batch_block_summaries(
        self, keys: list[tuple[int, int]]
    ) -> dict[tuple[int, int], str]:
        """Fetch block summaries for a list of (ref_id, block_index) pairs."""
        from sqlalchemy import tuple_

        with self._Session() as session:
            stmt = select(Block.ref_id, Block.block_index, Block.summary).where(
                tuple_(Block.ref_id, Block.block_index).in_(keys),
                Block.summary.isnot(None),
                Block.profile == "default",
            )
            rows = session.execute(stmt).all()
            return {(r[0], r[1]): r[2] for r in rows if r[2]}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get(self, identifier) -> dict[str, Any] | None:
        """Get document by ref_id (int), slug, or DOI.

        Returns merged ref + paper dict, or None.
        """
        with self._Session() as session:
            ref = None

            # Try int ref_id
            if isinstance(identifier, int):
                ref = session.get(Ref, identifier)
            elif isinstance(identifier, str):
                # Try slug (on Ref)
                ref = session.execute(
                    select(Ref)
                    .options(joinedload(Ref.paper))
                    .where(Ref.slug == identifier)
                ).scalar_one_or_none()

                # Try DOI
                if not ref:
                    ref = session.execute(
                        select(Ref).where(Ref.doi == identifier)
                    ).scalar_one_or_none()

                # Try as int string
                if not ref:
                    try:
                        ref = session.get(Ref, int(identifier))
                    except (ValueError, TypeError):
                        pass

            if ref:
                return ref.to_dict()
        return None

    def get_blocks(
        self,
        identifier,
        block_type: str | None = None,
        supplement: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get blocks for a paper by ref_id, slug, or DOI.

        Args:
            identifier: ref_id (int), slug, or DOI.
            block_type: Optional filter (e.g. "text", "abstract").
            supplement: If set, only blocks for that supplement.
                        If None (default), only main-paper blocks.
                        Use "*" for all blocks (main + supplements).
        """
        paper = self.get(identifier)
        if not paper or "ref_id" not in paper:
            return []
        ref_id = paper["ref_id"]
        with self._Session() as session:
            stmt = select(Block).where(Block.ref_id == ref_id)
            if block_type:
                stmt = stmt.where(Block.block_type == block_type)
            if supplement == "*":
                pass  # no filter — return all
            elif supplement is not None:
                stmt = stmt.where(Block.supplement == supplement)
            else:
                stmt = stmt.where(Block.supplement.is_(None))
            stmt = stmt.order_by(Block.page, Block.block_index)
            rows = session.execute(stmt).scalars().all()
            return [r.to_dict() for r in rows]

    def update_block_text(
        self,
        identifier,
        node_id: str,
        text: str,
    ) -> None:
        """Update the text content of a specific block.

        Args:
            identifier: ref_id (int), slug, or DOI.
            node_id: The block's node_id.
            text: New text content.

        Raises ValueError if ref or block not found.

        The block's embedding is refreshed best-effort after the text
        change so semantic search stays consistent with the new content.
        Indexing failures are logged but never propagate.
        """
        paper = self.get(identifier)
        # ``Ref.to_dict()`` surfaces ``id`` for unpapered refs (todos,
        # flashcards, wiki, notes, memories) and ``ref_id`` only when
        # a :class:`Paper` ingestion receipt is merged in.  Accept
        # either so direct-write corpora work here.
        ref_id = paper.get("ref_id") if paper else None
        if ref_id is None and paper:
            ref_id = paper.get("id")
        if ref_id is None:
            raise ValueError(f"Ref not found: {identifier}")
        with self._Session() as session:
            block = session.execute(
                select(Block).where(
                    Block.ref_id == ref_id,
                    Block.node_id == node_id,
                )
            ).scalar_one_or_none()
            if not block:
                raise ValueError(f"Block not found: {node_id} in ref {identifier}")
            block.text = text
            block_type = block.block_type
            session.commit()

        # Refresh embedding for the changed text.
        corpus_id = paper.get("corpus_id") if paper else None
        self._index_direct_blocks(
            ref_id, [(node_id, text, block_type)], corpus_id=corpus_id
        )

    def add_block(
        self,
        identifier,
        *,
        text: str,
        block_type: str = "text",
        node_id: str | None = None,
        section_path: list | None = None,
        page: int = 0,
    ) -> str:
        """Append a single block to an existing ref.

        Public API for direct-write corpora (flashcards, todos, memories,
        conversations, etc.) that need to add secondary blocks — context
        notes on a flashcard, turn-by-turn appends to a conversation,
        annotation blocks on a memory — without reaching into
        :class:`~acatome_store.models.Block` or :attr:`_Session` directly.

        The block's embedding is computed automatically (best-effort).

        Args:
            identifier: ref_id, slug, or DOI of the target ref.
            text: Block text content.  Empty strings are permitted but
                will not be embedded.
            block_type: One of the seeded block types.  Defaults to
                ``"text"``.  Types in :data:`SKIP_EMBED_TYPES` are
                skipped by the embedder.
            node_id: Optional explicit node_id.  When omitted, a
                sequentially-numbered id is derived from the ref's slug
                (``{slug}-b{NNNN}`` where NNNN is the next free index).
            section_path: Optional list of section breadcrumb strings.
            page: Page number if meaningful for this block_type; the
                default of 0 is appropriate for synthetic/direct-write
                blocks that have no source page.

        Returns:
            The block's ``node_id`` (generated or passed-in).

        Raises:
            ValueError: the ref does not exist, or the requested
                ``node_id`` collides with an existing block.
        """
        ref = self.get(identifier)
        # See :meth:`update_block_text` for the ``id``/``ref_id`` rationale.
        ref_id = ref.get("ref_id") if ref else None
        if ref_id is None and ref:
            ref_id = ref.get("id")
        if ref_id is None:
            raise ValueError(f"Ref not found: {identifier}")
        slug = ref.get("slug") or f"ref{ref_id}"

        with self._Session() as session:
            # Find the next free block_index for this ref.
            existing = (
                session.execute(
                    select(Block)
                    .where(Block.ref_id == ref_id)
                    .order_by(Block.block_index.desc())
                )
                .scalars()
                .all()
            )
            next_idx = 0
            for b in existing:
                if b.block_index is not None:
                    next_idx = b.block_index + 1
                    break
            # Fallback: if no block has block_index set, use count.
            if next_idx == 0 and existing:
                next_idx = len(existing)

            chosen_node_id = node_id or f"{slug}-b{next_idx:04d}"

            # Collision check — node_id must be unique within the ref.
            clash = session.execute(
                select(Block).where(
                    Block.ref_id == ref_id,
                    Block.node_id == chosen_node_id,
                )
            ).scalar_one_or_none()
            if clash:
                raise ValueError(
                    f"Block node_id already exists on ref {slug!r}: {chosen_node_id}"
                )

            block = Block(
                node_id=chosen_node_id,
                profile="default",
                ref_id=ref_id,
                page=page,
                block_index=next_idx,
                block_type=block_type,
                text=text,
                section_path=json.dumps(section_path or []),
            )
            session.add(block)
            session.commit()

        # Embed the new block (best-effort).
        corpus_id = ref.get("corpus_id") if ref else None
        self._index_direct_blocks(
            ref_id, [(chosen_node_id, text, block_type)], corpus_id=corpus_id
        )
        return chosen_node_id

    # ------------------------------------------------------------------
    # Embedding backfill (one-time migration for legacy direct-writes)
    # ------------------------------------------------------------------

    def backfill_embeddings(
        self,
        *,
        batch_size: int = 64,
        corpus_id: str | None = None,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """Populate missing embeddings for blocks written before embed-on-write.

        Direct-write corpora (todos, flashcards, notes, wiki, memories,
        conversations) historically inserted blocks with ``embedding =
        NULL`` because :meth:`create_ref` did not compute embeddings.
        After upgrading to the embed-on-write code path, existing rows
        remain unembedded; this method scans them, skips the usual
        :data:`SKIP_EMBED_TYPES`, and pushes freshly computed vectors
        into the pgvector column in batches.

        Only supported against the ``postgres`` vector backend — the
        Chroma backend has no concept of a NULL embedding row to find.

        Args:
            batch_size: Texts per call to the embedder.  Larger batches
                amortise the model's per-call overhead but keep a bigger
                peak memory footprint.
            corpus_id: If set, restrict to refs in this corpus only.
                Useful for staged rollouts (``corpus_id="todos"`` before
                ``corpus_id="flashcards"`` etc.).
            limit: If set, stop after scheduling this many blocks.  Use
                to dry-run capacity planning.
            dry_run: If True, log what *would* be embedded and return
                counts without touching the embedder or the database.

        Returns:
            Dict with keys:
              * ``scanned`` — blocks matching the NULL-embedding filter
              * ``embedded`` — blocks that were successfully embedded
              * ``skipped`` — blocks omitted (empty text / skip-type)
              * ``failed`` — blocks where the embedder raised

        Raises:
            NotImplementedError: vector backend is not postgres.
            RuntimeError: pgvector column is not available.
        """
        if self._config.vector_backend != "postgres":
            raise NotImplementedError(
                "backfill_embeddings() requires vector_backend='postgres' "
                f"(current: {self._config.vector_backend!r})"
            )
        if not hasattr(Block, "embedding"):
            raise RuntimeError(
                "pgvector 'embedding' column is not present on blocks — "
                "ensure the pgvector extension is installed and the store "
                "was initialised with a postgres DSN."
            )

        scanned = embedded = skipped = failed = 0

        with self._Session() as session:
            stmt = (
                select(Block).where(Block.embedding.is_(None)).where(Block.text != "")
            )
            if corpus_id:
                stmt = stmt.join(Ref, Block.ref_id == Ref.id).where(
                    Ref.corpus_id == corpus_id
                )
            if limit:
                stmt = stmt.limit(limit)

            blocks = session.execute(stmt).scalars().all()
            scanned = len(blocks)

            # Filter out skip-types client-side so the count reflects
            # intent, not just SQL reach.
            work: list[Block] = []
            for b in blocks:
                if b.block_type in SKIP_EMBED_TYPES:
                    skipped += 1
                    continue
                if not (b.text or "").strip():
                    skipped += 1
                    continue
                work.append(b)

            if dry_run:
                log.info(
                    "backfill_embeddings dry-run: scanned=%d would_embed=%d skipped=%d",
                    scanned,
                    len(work),
                    skipped,
                )
                return {
                    "scanned": scanned,
                    "embedded": 0,
                    "skipped": skipped,
                    "failed": 0,
                }

            embedder = self._embedder
            if embedder is None:
                raise EmbedderUnavailableError(
                    "No embedder available for backfill — install the "
                    "configured embed provider or adjust embed_model in "
                    "the store config."
                )

            # Batch through the embedder, committing after each batch so
            # partial progress survives a mid-run failure.
            for i in range(0, len(work), batch_size):
                batch = work[i : i + batch_size]
                texts = [b.text for b in batch]
                try:
                    vectors = embedder(texts)
                except Exception as exc:  # pragma: no cover — best-effort
                    log.warning(
                        "backfill batch failed at offset %d (%d blocks): %s",
                        i,
                        len(batch),
                        exc,
                    )
                    failed += len(batch)
                    continue

                for b, vec in zip(batch, vectors):
                    b.embedding = vec
                    embedded += 1

                session.commit()
                log.info("backfill: %d/%d embedded", embedded, len(work))

        return {
            "scanned": scanned,
            "embedded": embedded,
            "skipped": skipped,
            "failed": failed,
        }

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    _FIG_NUM_RE = re.compile(r"(?:Fig(?:ure)?\.?\s*|Scheme\s*)(\d+)", re.IGNORECASE)

    def get_figures(self, identifier) -> list[dict[str, Any]]:
        """Get figure metadata for a paper, with parsed figure numbers.

        Returns list of dicts: {fig_num, caption, page, block_index, node_id}.
        Figures are numbered by caption label (Fig 1, Figure 2, Scheme 3).
        Figures without a parseable number are assigned sequentially.
        """
        blocks = self.get_blocks(identifier, block_type="figure")
        figures: list[dict[str, Any]] = []
        next_auto = 1
        used_nums: set[int] = set()

        # First pass: extract explicit numbers
        for b in blocks:
            caption = b.get("text", "")
            m = self._FIG_NUM_RE.search(caption)
            num = int(m.group(1)) if m else None
            if num is not None:
                used_nums.add(num)
            figures.append(
                {
                    "fig_num": num,
                    "caption": caption,
                    "page": b.get("page"),
                    "block_index": b.get("block_index"),
                    "node_id": b.get("node_id", ""),
                }
            )

        # Second pass: assign auto numbers to figures without explicit labels
        for fig in figures:
            if fig["fig_num"] is None:
                while next_auto in used_nums:
                    next_auto += 1
                fig["fig_num"] = next_auto
                used_nums.add(next_auto)
                next_auto += 1

        return figures

    def get_figure_image(self, identifier, fig_num: int) -> dict[str, Any] | None:
        """Get figure image data from the bundle.

        Args:
            identifier: slug, ref_id, or DOI.
            fig_num: Figure number (from caption label).

        Returns:
            Dict with keys: fig_num, caption, page, image_bytes, image_ext,
            or None if not found.
        """
        # Find the figure metadata
        figures = self.get_figures(identifier)
        fig_meta = next((f for f in figures if f["fig_num"] == fig_num), None)
        if fig_meta is None:
            return None

        # Read the bundle to get image bytes
        paper = self.get(identifier)
        if not paper:
            return None

        with self._Session() as session:
            paper_row = session.execute(
                select(Paper).where(Paper.ref_id == paper["ref_id"])
            ).scalar_one_or_none()
            if not paper_row or not paper_row.bundle_path:
                return None
            bundle_path = Path(paper_row.bundle_path)

        if not bundle_path.exists():
            return None

        data = _read_bundle(bundle_path)
        # Match by node_id (reliable across bundle and store)
        target_node_id = fig_meta["node_id"]
        for block in data.get("blocks", []):
            if block.get("node_id") == target_node_id and block.get("type") == "figure":
                b64 = block.get("image_base64", "")
                mime = block.get("image_mime", "image/png")
                if not b64:
                    return None
                import base64

                ext = ".png" if "png" in mime else ".jpg"
                return {
                    "fig_num": fig_num,
                    "caption": fig_meta["caption"],
                    "page": fig_meta["page"],
                    "image_bytes": base64.b64decode(b64),
                    "image_ext": ext,
                }

        return None

    def get_toc(
        self, identifier, supplement: str | None = None
    ) -> list[dict[str, Any]]:
        """Get a lightweight table-of-contents for a paper.

        Returns one entry per block: node_id, page, block_type,
        summary (if available, else truncated text), section_path.
        """
        paper = self.get(identifier)
        if not paper or "ref_id" not in paper:
            return []
        ref_id = paper["ref_id"]
        with self._Session() as session:
            stmt = (
                select(Block)
                .where(Block.ref_id == ref_id)
                .order_by(Block.page, Block.block_index)
            )
            if supplement == "*":
                pass
            elif supplement is not None:
                stmt = stmt.where(Block.supplement == supplement)
            else:
                stmt = stmt.where(Block.supplement.is_(None))
            rows = session.execute(stmt).scalars().all()
            toc = []
            for r in rows:
                preview = r.summary or (
                    r.text[:120] + "…" if len(r.text) > 120 else r.text
                )
                entry = {
                    "node_id": r.node_id,
                    "block_index": r.block_index,
                    "page": r.page,
                    "block_type": r.block_type,
                    "section_path": r.section_path,
                    "preview": preview,
                }
                if r.summary:
                    entry["has_summary"] = True
                toc.append(entry)
            return toc

    def reindex_blocks(self, identifier=None) -> int:
        """Reassign block_index as a global sequential counter per paper.

        Fixes papers ingested with the old per-page block_index scheme.
        If *identifier* is given, reindex that paper only; otherwise all.
        Returns the number of papers reindexed.
        """
        papers = (
            [self.get(identifier)] if identifier else self.list_papers(limit=10_000)
        )
        count = 0
        with self._Session() as session:
            for paper in papers:
                if not paper or "ref_id" not in paper:
                    continue
                ref_id = paper["ref_id"]
                rows = (
                    session.execute(
                        select(Block)
                        .where(
                            Block.ref_id == ref_id,
                            Block.supplement.is_(None),
                        )
                        .order_by(Block.page, Block.block_index, Block.node_id)
                    )
                    .scalars()
                    .all()
                )
                for new_idx, row in enumerate(rows):
                    if row.block_index != new_idx:
                        row.block_index = new_idx
                count += 1
            session.commit()
        return count

    def delete(self, identifier) -> bool:
        """Delete the ingested Paper + blocks (keeps the Ref stub)."""
        paper_dict = self.get(identifier)
        if not paper_dict or "ref_id" not in paper_dict:
            return False
        ref_id = paper_dict["ref_id"]
        with self._Session() as session:
            ref = session.get(Ref, ref_id)
            if not ref:
                return False
            # Blocks FK to ref — delete explicitly
            for block in list(ref.blocks):
                session.delete(block)
            if ref.paper:
                session.delete(ref.paper)
            session.commit()
        try:
            self.index.delete_paper(str(ref_id))
        except (KeyError, ValueError) as exc:
            # Vector index may not contain this paper (indexing was skipped
            # during ingest, or backend switched). Metadata row is still deleted.
            log.debug("Vector index delete skipped for ref_id=%s: %s", ref_id, exc)
        return True

    def list_papers(
        self,
        limit: int = 100,
        offset: int = 0,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """List ingested papers (joined with ref metadata).

        Args:
            limit: Max papers to return.
            offset: Pagination offset.
            since: Only return papers ingested on or after this datetime (UTC, naive).
        """
        with self._Session() as session:
            stmt = (
                select(Ref)
                .join(Paper)
                .options(joinedload(Ref.paper))
                .order_by(Paper.ingested_at.desc())
            )
            if since is not None:
                stmt = stmt.where(Paper.ingested_at >= since)
            rows = (
                session.execute(stmt.limit(limit).offset(offset))
                .unique()
                .scalars()
                .all()
            )
            result = []
            for r in rows:
                block_count = (
                    session.query(Block)
                    .filter(Block.ref_id == r.id, Block.profile == "default")
                    .count()
                )
                kw = None
                if r.keywords:
                    try:
                        kw = json.loads(r.keywords)
                    except (json.JSONDecodeError, TypeError):
                        kw = r.keywords
                tags: list[str] = []
                if r.tags:
                    try:
                        tags = json.loads(r.tags)
                    except (json.JSONDecodeError, TypeError):
                        pass
                result.append(
                    {
                        "ref_id": r.id,
                        "slug": r.slug,
                        "title": r.title,
                        "authors": r.authors,
                        "year": r.year,
                        "doi": r.doi,
                        "corpus_id": r.corpus_id,
                        "block_count": block_count,
                        "keywords": kw,
                        "tags": tags,
                        "ingested_at": r.paper.ingested_at if r.paper else None,
                    }
                )
            return result

    def list_refs_by_corpus(
        self,
        corpus_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List refs filtered by corpus, ordered by first_seen_at desc.

        Use for direct-write corpora (todos, wiki, notes, flashcards, etc.)
        where the refs table is the source of truth and there is no Paper
        row to join against.

        Args:
            corpus_id: Corpus to filter by (e.g. ``"todos"``, ``"flashcards"``).
            limit: Max refs to return.
            offset: Pagination offset.
        """
        with self._Session() as session:
            stmt = (
                select(Ref)
                .where(Ref.corpus_id == corpus_id)
                .order_by(Ref.first_seen_at.desc())
                .limit(limit)
                .offset(offset)
            )
            rows = session.execute(stmt).scalars().all()
            return [r.to_dict() for r in rows]

    def stats(self) -> dict[str, Any]:
        """Return store statistics and connection info."""
        cfg = self._config
        with self._Session() as session:
            total_refs = session.query(Ref).count()
            total_ingested = session.query(Paper).count()
            total_blocks = session.query(Block).count()
            verified = session.query(Paper).filter(Paper.verified.is_(True)).count()
        indexed = self.index.count()

        # Connection info
        info: dict[str, Any] = {
            "metadata_backend": cfg.metadata_backend,
        }
        if cfg.metadata_backend == "postgres":
            info["pg_host"] = cfg.pg_host
            info["pg_port"] = cfg.pg_port
            info["pg_database"] = cfg.pg_database
            info["pg_schema"] = cfg.pg_schema
            info["pg_user"] = cfg.pg_user
        else:
            info["db_path"] = str(cfg.store_path / "acatome.db")

        info.update(
            {
                "vector_backend": cfg.vector_backend,
                "embed_model": cfg.embed_model,
                "embed_dim": cfg.embed_dim,
                "store_path": str(cfg.store_path),
                "total_refs": total_refs,
                "total_papers": total_ingested,
                "total_blocks": total_blocks,
                "verified": verified,
                "indexed_blocks": indexed,
            }
        )
        return info

    # ------------------------------------------------------------------
    # Links
    # ------------------------------------------------------------------

    def create_link(
        self,
        src_slug: str,
        dst_slug: str,
        relation: str = "cites",
        *,
        src_node_id: str | None = None,
        dst_node_id: str | None = None,
    ) -> Link:
        """Create a link between two refs or blocks.

        Args:
            src_slug: Source ref slug.
            dst_slug: Target ref slug.
            relation: Link type name (must exist in link_types table).
            src_node_id: Optional source block node_id.
            dst_node_id: Optional target block node_id.

        Returns:
            The created Link object.

        Raises:
            ValueError: If relation is not a valid link type, or if
                src_slug/dst_slug don't resolve to existing refs.
        """
        with self._Session() as session:
            # Validate relation
            lt = session.get(LinkType, relation)
            if not lt:
                valid = [r.name for r in session.execute(select(LinkType)).scalars()]
                raise ValueError(
                    f"Unknown link relation '{relation}'. "
                    f"Valid types: {', '.join(sorted(valid))}"
                )
            # Validate source ref exists
            src_ref = session.execute(
                select(Ref).where(Ref.slug == src_slug)
            ).scalar_one_or_none()
            if not src_ref:
                raise ValueError(f"Source ref not found: '{src_slug}'")
            # Validate target ref exists
            dst_ref = session.execute(
                select(Ref).where(Ref.slug == dst_slug)
            ).scalar_one_or_none()
            if not dst_ref:
                raise ValueError(f"Target ref not found: '{dst_slug}'")

            link = Link(
                src_slug=src_slug,
                src_node_id=src_node_id,
                dst_slug=dst_slug,
                dst_node_id=dst_node_id,
                relation=relation,
            )
            session.add(link)
            session.commit()
            session.refresh(link)
            return link

    def get_links(
        self,
        slug: str,
        *,
        node_id: str | None = None,
        relation: str | None = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Get links for a ref or block.

        Args:
            slug: Ref slug.
            node_id: Optional block node_id (filters to block-level links).
            relation: Optional filter by link type name.
            direction: 'outbound', 'inbound', or 'both'.

        Returns:
            List of link dicts, each with an extra 'display_relation' key
            showing the natural-language label from the viewing side.

        Raises:
            ValueError: If direction is not one of 'outbound', 'inbound', 'both'.
        """
        if direction not in ("outbound", "inbound", "both"):
            raise ValueError(
                f"Invalid direction '{direction}', must be 'outbound', 'inbound', or 'both'"
            )
        results = []
        with self._Session() as session:
            # Build outbound query (this slug is src)
            if direction in ("outbound", "both"):
                q = select(Link).where(Link.src_slug == slug)
                if node_id is not None:
                    q = q.where(Link.src_node_id == node_id)
                if relation:
                    q = q.where(Link.relation == relation)
                q = q.order_by(Link.created_at)
                for link in session.execute(q).scalars():
                    d = link.to_dict()
                    d["display_relation"] = link.relation
                    d["direction"] = "outbound"
                    results.append(d)

            # Build inbound query (this slug is dst)
            if direction in ("inbound", "both"):
                q = select(Link).where(Link.dst_slug == slug)
                if node_id is not None:
                    q = q.where(Link.dst_node_id == node_id)
                if relation:
                    q = q.where(Link.relation == relation)
                q = q.order_by(Link.created_at)
                for link in session.execute(q).scalars():
                    d = link.to_dict()
                    # Use inverse label for inbound links
                    lt = session.get(LinkType, link.relation)
                    d["display_relation"] = lt.inverse if lt else link.relation
                    d["direction"] = "inbound"
                    results.append(d)

        return results

    def get_link_count(self, slug: str) -> dict[str, int]:
        """Get link counts for a ref, grouped by display relation.

        Returns:
            Dict mapping display_relation to count.
            E.g. {"cites": 3, "cited_by": 5, "annotated_by": 2}
        """
        links = self.get_links(slug)
        counts: dict[str, int] = {}
        for link in links:
            rel = link["display_relation"]
            counts[rel] = counts.get(rel, 0) + 1
        return counts

    def delete_link(self, link_id: int) -> bool:
        """Delete a link by ID.

        Returns:
            True if deleted, False if link not found.
        """
        with self._Session() as session:
            link = session.get(Link, link_id)
            if not link:
                return False
            session.delete(link)
            session.commit()
            return True

    def delete_links_for_slug(self, slug: str) -> int:
        """Delete all links where slug is src or dst.

        Returns:
            Number of links deleted.
        """
        from sqlalchemy import delete as sa_delete
        from sqlalchemy import or_

        with self._Session() as session:
            stmt = sa_delete(Link).where(
                or_(Link.src_slug == slug, Link.dst_slug == slug)
            )
            result = session.execute(stmt)
            session.commit()
            return result.rowcount

    # ------------------------------------------------------------------
    # Notes (DEPRECATED — use links with relation='annotates')
    # ------------------------------------------------------------------

    def add_note(
        self,
        content: str,
        *,
        ref_id: int | None = None,
        block_node_id: str | None = None,
        block_profile: str = "default",
        title: str | None = None,
        tags: list[str] | None = None,
        origin: str | None = None,
    ) -> int:
        """Add a note on a paper (ref_id) or block (block_node_id).

        Returns note id.
        """
        with self._Session() as session:
            note = Note(
                ref_id=ref_id,
                block_node_id=block_node_id,
                block_profile=block_profile if block_node_id else None,
                title=title,
                content=content,
                tags=json.dumps(tags) if tags else None,
                origin=origin,
            )
            session.add(note)
            session.commit()
            return note.id

    def get_notes(
        self,
        *,
        ref_id: int | None = None,
        block_node_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get notes for a paper or block."""
        with self._Session() as session:
            q = select(Note)
            if ref_id is not None:
                q = q.where(Note.ref_id == ref_id)
            if block_node_id is not None:
                q = q.where(Note.block_node_id == block_node_id)
            q = q.order_by(Note.created_at.desc())
            rows = session.execute(q).scalars().all()
            return [r.to_dict() for r in rows]

    def update_note(self, note_id: int, **kwargs) -> bool:
        """Update a note. Pass content=, title=, tags= as kwargs."""
        with self._Session() as session:
            note = session.get(Note, note_id)
            if not note:
                return False
            for key in ("content", "title"):
                if key in kwargs:
                    setattr(note, key, kwargs[key])
            if "tags" in kwargs:
                note.tags = json.dumps(kwargs["tags"]) if kwargs["tags"] else None
            session.commit()
            return True

    def delete_note(self, note_id: int) -> bool:
        """Delete a note by id."""
        with self._Session() as session:
            note = session.get(Note, note_id)
            if not note:
                return False
            session.delete(note)
            session.commit()
            return True

    # ------------------------------------------------------------------
    # Supplements
    # ------------------------------------------------------------------

    def ingest_supplement(
        self,
        parent_identifier,
        bundle_path: str | Path,
        supplement_name: str,
    ) -> int:
        """Ingest a supplement PDF bundle into an existing paper.

        The supplement blocks are stored with ``supplement=name`` on the
        parent paper's ref_id.  The paper's ``supplements`` JSON list is
        updated to include the new name.

        Args:
            parent_identifier: ref_id (int), slug, or DOI of the parent.
            bundle_path: Path to the .acatome bundle for the supplement.
            supplement_name: Lowercase string id (e.g. "s1", "methods").

        Returns:
            ref_id (int) of the parent paper.
        """
        parent = self.get(parent_identifier)
        if not parent or "ref_id" not in parent:
            raise ValueError(f"Parent paper not found: {parent_identifier}")

        ref_id = parent["ref_id"]
        supplement_name = supplement_name.lower()
        bundle_path = Path(bundle_path)
        data = _read_bundle(bundle_path)

        with self._Session() as session:
            paper = session.execute(
                select(Paper).where(Paper.ref_id == ref_id)
            ).scalar_one_or_none()
            if not paper:
                raise ValueError(f"No ingested paper for ref_id={ref_id}")

            # Delete old supplement blocks with same name (re-ingest)
            old = (
                session.execute(
                    select(Block).where(
                        Block.ref_id == ref_id,
                        Block.supplement == supplement_name,
                    )
                )
                .scalars()
                .all()
            )
            for b in old:
                session.delete(b)
            session.flush()

            # Insert supplement blocks (block_index is a global sequential counter)
            blocks = data.get("blocks", [])
            for block_index, b in enumerate(blocks):
                node_id = b["node_id"]

                bbox = b.get("bbox")
                block = Block(
                    node_id=f"{node_id}:supp:{supplement_name}",
                    profile="default",
                    ref_id=ref_id,
                    page=b.get("page", 0),
                    block_index=block_index,
                    block_type=b.get("type", "text"),
                    text=b.get("text", ""),
                    summary=pick_best_summary(b.get("summaries")) or b.get("summary"),
                    supplement=supplement_name,
                    section_path=json.dumps(b.get("section_path", [])),
                    bbox_x0=bbox[0] if bbox else None,
                    bbox_y0=bbox[1] if bbox else None,
                    bbox_x1=bbox[2] if bbox else None,
                    bbox_y1=bbox[3] if bbox else None,
                )
                session.add(block)

            # Update supplements list on paper
            existing = json.loads(paper.supplements) if paper.supplements else []
            if supplement_name not in existing:
                existing.append(supplement_name)
                existing.sort()
            paper.supplements = json.dumps(existing)

            session.commit()

        if blocks:
            self.index.add_blocks(str(ref_id), blocks, corpus_id="papers")

        return ref_id

    def get_supplements(self, identifier) -> list[str]:
        """Return list of supplement names for a paper, or empty list."""
        paper = self.get(identifier)
        if not paper:
            return []
        supps = paper.get("supplements")
        if not supps:
            return []
        try:
            return json.loads(supps)
        except (json.JSONDecodeError, TypeError):
            return []

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def add_tags(self, identifier, tags: list[str]) -> bool:
        """Add tags to a paper (additive, deduped).

        Args:
            identifier: ref_id (int), slug, or DOI.
            tags: Tag strings to add.

        Returns:
            True if the ref was found and updated.
        """
        paper_dict = self.get(identifier)
        if not paper_dict or "id" not in paper_dict:
            return False
        with self._Session() as session:
            ref = session.get(Ref, paper_dict["id"])
            if not ref:
                return False
            self._merge_tags(ref, tags)
            session.commit()
        return True

    def remove_tags(self, identifier, tags: list[str]) -> bool:
        """Remove tags from a paper.

        Args:
            identifier: ref_id (int), slug, or DOI.
            tags: Tag strings to remove.

        Returns:
            True if the ref was found and updated.
        """
        paper_dict = self.get(identifier)
        if not paper_dict or "id" not in paper_dict:
            return False
        with self._Session() as session:
            ref = session.get(Ref, paper_dict["id"])
            if not ref:
                return False
            existing: list[str] = json.loads(ref.tags) if ref.tags else []
            remaining = sorted(set(existing) - set(tags))
            ref.tags = json.dumps(remaining) if remaining else None
            session.commit()
        return True

    def get_tags(self, identifier) -> list[str]:
        """Return tags for a paper, or empty list."""
        paper_dict = self.get(identifier)
        if not paper_dict:
            return []
        raw = paper_dict.get("tags")
        if not raw:
            return []
        try:
            return json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return []

    def find_by_tag(self, tag: str) -> list[dict[str, Any]]:
        """Find all papers with a given tag.

        Returns list of dicts with ref_id, slug, title, year, tags.
        """
        with self._Session() as session:
            refs = session.execute(select(Ref)).scalars().all()
            results = []
            for r in refs:
                tags: list[str] = json.loads(r.tags) if r.tags else []
                if tag in tags:
                    results.append(
                        {
                            "ref_id": r.id,
                            "slug": r.slug,
                            "title": r.title,
                            "year": r.year,
                            "tags": tags,
                        }
                    )
            return results

    def list_tags(self) -> dict[str, int]:
        """Return all tags with counts, sorted by count descending."""
        with self._Session() as session:
            refs = session.execute(select(Ref)).scalars().all()
            counts: dict[str, int] = {}
            for r in refs:
                tags: list[str] = json.loads(r.tags) if r.tags else []
                for t in tags:
                    counts[t] = counts.get(t, 0) + 1
            return dict(sorted(counts.items(), key=lambda x: -x[1]))

    # ------------------------------------------------------------------
    # Retractions
    # ------------------------------------------------------------------

    def retract(self, identifier, note: str = "") -> bool:
        """Mark a document as retracted.

        Args:
            identifier: ref_id (int), slug, or DOI.
            note: Optional retraction note.

        Returns:
            True if the ref was found and updated.
        """
        paper_dict = self.get(identifier)
        if not paper_dict or "id" not in paper_dict:
            return False
        with self._Session() as session:
            ref = session.get(Ref, paper_dict["id"])
            if not ref:
                return False
            ref._set_meta_field("retracted", True)
            if note:
                ref._set_meta_field("retraction_note", note)
            session.commit()
        return True

    def unretract(self, identifier) -> bool:
        """Remove retraction flag from a document."""
        paper_dict = self.get(identifier)
        if not paper_dict or "id" not in paper_dict:
            return False
        with self._Session() as session:
            ref = session.get(Ref, paper_dict["id"])
            if not ref:
                return False
            ref._set_meta_field("retracted", False)
            ref._set_meta_field("retraction_note", None)
            session.commit()
        return True

    def reset_schema(self) -> None:
        """Drop all tables and recreate from the current model.

        WARNING: destroys all data. Use for dev/reingest workflows only.
        """
        # Drop dependent views first (Postgres blocks DROP TABLE otherwise)
        if self._config.db_url.startswith("postgresql"):
            from sqlalchemy import text

            with self._engine.connect() as conn:
                conn.execute(text("DROP VIEW IF EXISTS blocks_v CASCADE"))
                conn.commit()
        Base.metadata.drop_all(self._engine)
        self._init_db()

    def close(self) -> None:
        """Dispose of the engine."""
        if hasattr(self, "_engine"):
            self._engine.dispose()
