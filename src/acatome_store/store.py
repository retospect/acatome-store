"""Core Store class: ingest, search, metadata queries.

Persistence:
  - Relational data (refs, papers, blocks, citations) → SQLAlchemy ORM
    (portable across SQLite, Postgres, MySQL via connection string)
  - Vector search → Chroma (LlamaIndex) or pgvector (SQLAlchemy column)

Identity model:
  - ``Ref`` = one row per known paper (ingested or stub). Holds DOI,
    S2 ID, arxiv ID, title, etc.  Auto-int PK.
  - ``Paper`` = ingested content (1:1 optional on Ref). Slug, PDF hash,
    bundle path. Only exists when we have a PDF.
  - ``Block`` = text chunks from ingested papers (FK → Paper).
  - ``Citation`` = directed edge (citing_ref → cited_ref). Both FKs
    point to Ref, so the graph spans ingested + stub papers.
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, joinedload, sessionmaker

from precis_summary import pick_best_summary
from precis_summary.rake import telegram_precis

from acatome_store.config import StoreConfig
from acatome_store.models import (
    Base,
    Block,
    Note,
    Paper,
    Ref,
    create_blocks_view,
    seed_block_types,
)
from acatome_store.vector import VectorIndex, create_index

log = logging.getLogger(__name__)

# Block types that skip embedding (same as acatome_extract.enrich)
_SKIP_EMBED_TYPES = {"section_header", "title", "author", "equation", "junk"}


def _get_embedder(
    config: StoreConfig,
) -> Callable[[list[str]], list[list[float]]] | None:
    """Build an embedding function from the store's configured profile.

    Returns None if the embedding backend is unavailable.
    """
    if config.embed_provider == "chroma":
        try:
            from chromadb.utils.embedding_functions import (
                DefaultEmbeddingFunction,
            )

            ef = DefaultEmbeddingFunction()

            def _chroma_embed(texts: list[str]) -> list[list[float]]:
                results = ef(texts)
                return [
                    e.tolist() if hasattr(e, "tolist") else list(e) for e in results
                ]

            return _chroma_embed
        except Exception:
            return None

    if config.embed_provider == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(config.embed_model)
            dim = config.embed_index_dim or config.embed_dim

            def _st_embed(texts: list[str]) -> list[list[float]]:
                embs = model.encode(texts, normalize_embeddings=True)
                return [e[:dim].tolist() for e in embs]

            return _st_embed
        except Exception:
            return None

    return None


def _reembed_blocks(
    blocks: list[dict[str, Any]],
    embedder: Callable[[list[str]], list[list[float]]],
    profile: str = "default",
) -> list[dict[str, Any]]:
    """Re-embed blocks using the given embedder, replacing existing embeddings."""
    texts = []
    indices = []
    for i, b in enumerate(blocks):
        if b.get("type") in _SKIP_EMBED_TYPES:
            continue
        text = b.get("text", "").strip()
        if not text:
            continue
        texts.append(text)
        indices.append(i)

    if not texts:
        return blocks

    embeddings = embedder(texts)
    for idx, emb in zip(indices, embeddings):
        if "embeddings" not in blocks[idx]:
            blocks[idx]["embeddings"] = {}
        blocks[idx]["embeddings"][profile] = emb

    return blocks


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

                add_pgvector_column()
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

        # Auto-migrate: add embedding column if Postgres and missing
        if is_pg:
            self._ensure_embedding_column()

        # Seed block_types lookup and create convenience view
        with self._Session() as session:
            seed_block_types(session)
        try:
            create_blocks_view(self._engine)
        except Exception:
            pass  # SQLite doesn't support CREATE OR REPLACE VIEW in all versions

    def _ensure_embedding_column(self) -> None:
        """Add embedding column to blocks table if missing (Postgres only)."""
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
                conn.commit()
        except Exception:
            pass  # pgvector not installed or other issue — non-fatal

    @property
    def index(self) -> VectorIndex:
        """Lazy-init vector index."""
        if self._index is None:
            sf = self._Session if self._config.vector_backend == "postgres" else None
            self._index = create_index(self._config, session_factory=sf)
        return self._index

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

    def _upsert_ref(self, session: Session, header: dict) -> Ref:
        """Find or create a Ref from bundle header, updating sparse fields."""
        ref = self._find_ref(session, header)
        if ref is None:
            ref = Ref()
            session.add(ref)

        # Update fields (fill in blanks, never overwrite with None)
        for field in (
            "doi",
            "s2_id",
            "arxiv_id",
            "title",
            "journal",
            "entry_type",
            "source",
        ):
            val = header.get(field)
            if val and not getattr(ref, field, None):
                setattr(ref, field, val)

        if header.get("authors"):
            if not ref.authors:
                ref.authors = json.dumps(header["authors"])
        if header.get("year") and not ref.year:
            ref.year = header["year"]
        if header.get("keywords"):
            if not ref.keywords:
                ref.keywords = json.dumps(header["keywords"])

        session.flush()  # ensure ref.id is assigned
        return ref

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, bundle_path: str | Path) -> int:
        """Ingest a .acatome bundle into the store.

        Dedup rules:
        - Same pdf_hash → skip (return existing ref_id)
        - Same DOI, different PDF → skip (return existing ref_id)
        - Same slug, different ref → raise ValueError
        - Same ref, re-ingest → atomic replace of Paper + blocks

        Returns:
            ref_id (int) of the ingested paper.
        """
        bundle_path = Path(bundle_path)
        data = _read_bundle(bundle_path)
        header = data["header"]
        slug = header.get("slug", "")
        pdf_hash = header["pdf_hash"]

        with self._Session() as session:
            # Dedup: same pdf_hash already ingested
            existing = session.execute(
                select(Paper).where(Paper.pdf_hash == pdf_hash)
            ).scalar_one_or_none()
            if existing:
                return existing.ref_id

            # Find or create the Ref (identity)
            ref = self._upsert_ref(session, header)

            # Dedup: ref has a Paper with a different PDF hash
            if ref.paper and ref.paper.pdf_hash != pdf_hash:
                return ref.id

            # Slug collision: slug taken by a different ref
            if slug:
                existing = session.execute(
                    select(Paper).where(Paper.slug == slug)
                ).scalar_one_or_none()
                if existing and existing.ref_id != ref.id:
                    raise ValueError(
                        f"Slug collision: '{slug}' already belongs to "
                        f"ref_id={existing.ref_id}. "
                        f"Resolve manually or rename."
                    )

            # Atomic replace: delete old Paper + blocks if re-ingesting
            if ref.paper:
                session.delete(ref.paper)
            # Blocks FK to ref, so delete them explicitly
            for old_block in list(ref.blocks):
                session.delete(old_block)
            session.flush()

            # Insert Paper
            paper = Paper(
                ref_id=ref.id,
                slug=slug or None,
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

            # Insert paper summary as a block (if present)
            enrich_meta = data.get("enrichment_meta") or {}
            # New format: paper_summaries dict; old format: paper_summary string
            paper_summaries = enrich_meta.get("paper_summaries") or {}
            paper_summary = pick_best_summary(paper_summaries) or enrich_meta.get(
                "paper_summary", ""
            )
            if paper_summary:
                session.add(
                    Block(
                        node_id=f"ref:{ref.id}:paper_summary",
                        profile="default",
                        ref_id=ref.id,
                        page=None,
                        block_index=None,
                        block_type="paper_summary",
                        text=paper_summary,
                    )
                )

            # Insert blocks
            blocks = data.get("blocks", [])
            for b in blocks:
                node_id = b["node_id"]
                try:
                    block_index = int(node_id.rsplit("-", 1)[-1])
                except (ValueError, IndexError):
                    block_index = 0

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
                embedder = _get_embedder(self._config)
                if embedder:
                    try:
                        blocks = _reembed_blocks(blocks, embedder)
                    except Exception:
                        log.warning("Re-embedding failed, skipping vector index")
                        blocks = []  # skip indexing
                else:
                    log.warning(
                        "No embedder available for '%s', skipping vector index",
                        self._config.embed_provider,
                    )
                    blocks = []  # skip indexing

            # Index embeddings in vector store (best-effort)
            try:
                self.index.add_blocks(str(ref_id), blocks)
            except Exception:
                pass

        return ref_id

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic text search via vector index.

        Returns list of result dicts with text, metadata, distance,
        and enriched paper info.
        """
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
            stmt = (
                select(Block.ref_id, Block.block_index, Block.summary)
                .where(
                    tuple_(Block.ref_id, Block.block_index).in_(keys),
                    Block.summary.isnot(None),
                    Block.profile == "default",
                )
            )
            rows = session.execute(stmt).all()
            return {(r[0], r[1]): r[2] for r in rows if r[2]}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get(self, identifier) -> dict[str, Any] | None:
        """Get paper by ref_id (int), slug, or DOI.

        Returns merged ref + paper dict, or None.
        """
        with self._Session() as session:
            ref = None

            # Try int ref_id
            if isinstance(identifier, int):
                ref = session.get(Ref, identifier)
            elif isinstance(identifier, str):
                # Try slug
                paper = session.execute(
                    select(Paper)
                    .options(joinedload(Paper.ref))
                    .where(Paper.slug == identifier)
                ).scalar_one_or_none()
                if paper:
                    ref = paper.ref

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
        except Exception:
            pass
        return True

    def list_papers(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """List ingested papers (joined with ref metadata)."""
        with self._Session() as session:
            rows = (
                session.execute(
                    select(Ref)
                    .join(Paper)
                    .options(joinedload(Ref.paper))
                    .order_by(Paper.ingested_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
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
                result.append(
                    {
                        "ref_id": r.id,
                        "slug": r.paper.slug if r.paper else None,
                        "title": r.title,
                        "year": r.year,
                        "doi": r.doi,
                        "block_count": block_count,
                        "keywords": kw,
                    }
                )
            return result

    def stats(self) -> dict[str, Any]:
        """Return store statistics and connection info."""
        cfg = self._config
        with self._Session() as session:
            total_refs = session.query(Ref).count()
            total_ingested = session.query(Paper).count()
            total_blocks = session.query(Block).count()
            verified = session.query(Paper).filter(Paper.verified.is_(True)).count()
        indexed = 0
        try:
            indexed = self.index.count()
        except Exception:
            pass

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
    # Notes
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

            # Insert supplement blocks
            blocks = data.get("blocks", [])
            for b in blocks:
                node_id = b["node_id"]
                try:
                    block_index = int(node_id.rsplit("-", 1)[-1])
                except (ValueError, IndexError):
                    block_index = 0

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

        # Index embeddings (best-effort)
        if blocks:
            try:
                self.index.add_blocks(str(ref_id), blocks)
            except Exception:
                pass

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
    # Retractions
    # ------------------------------------------------------------------

    def retract(self, identifier, note: str = "") -> bool:
        """Mark a paper as retracted.

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
            ref.retracted = True
            ref.retraction_note = note or None
            session.commit()
        return True

    def unretract(self, identifier) -> bool:
        """Remove retraction flag from a paper."""
        paper_dict = self.get(identifier)
        if not paper_dict or "id" not in paper_dict:
            return False
        with self._Session() as session:
            ref = session.get(Ref, paper_dict["id"])
            if not ref:
                return False
            ref.retracted = False
            ref.retraction_note = None
            session.commit()
        return True

    def close(self) -> None:
        """Dispose of the engine."""
        if hasattr(self, "_engine"):
            self._engine.dispose()


def _read_bundle(path: str | Path) -> dict[str, Any]:
    """Read a .acatome bundle (gzipped JSON)."""
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
