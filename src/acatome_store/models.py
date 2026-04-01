"""SQLAlchemy ORM models for acatome-store.

Schema overview (v3):
  ``corpora``     — Document type registry (string PK).  Each corpus defines
                    a handler, slug pattern, default tags, and write policy.
  ``block_types`` — Lookup table for block kinds (string PK).  Carries a
                    ``provenance`` flag ("original" | "generated") so
                    consumers always know what is original vs. LLM-generated.
  ``refs``        — Document identity (auto int PK).  One row per known
                    document (paper, note, todo, wiki page, etc.), ingested
                    or stub.  DOI/S2/arxiv are nullable UNIQUE lookup handles.
                    Polymorphic on ``corpus_id``.
  ``papers``      — Ingestion receipt (1:1 FK to refs).  PDF hash, bundle
                    path, verified flag.  Only exists when content is ingested.
  ``blocks``      — Text chunks + synthetic blocks (FK → refs).  Abstract
                    and document summary are blocks with special ``block_type``.
  ``link_types``  — Relation registry (name PK, inverse label).  Defines
                    valid edge types for the link graph.
  ``links``       — Slug-based edges between refs or blocks.  Nullable
                    node_id fields allow doc→doc, doc→block, block→block.
                    Replaces the older ``citations`` table.

Deprecated (kept for migration compatibility):
  ``citations``   — Use ``links`` with relation='cites' instead.
  ``notes``       — Use refs in the 'notes' corpus with 'annotates' links.

Portable across SQLite, Postgres, and MySQL via connection string.
On Postgres, ``blocks.embedding`` uses pgvector for native ANN search.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# block_types — lookup table (string PK, provenance flag)
# ---------------------------------------------------------------------------

# Seed data: inserted by seed_block_types() on first create_all.
BLOCK_TYPE_SEEDS = [
    ("text", "original", "Extracted text chunk"),
    ("section_header", "original", "Extracted section heading"),
    ("table", "original", "Extracted table"),
    ("figure", "original", "Extracted figure / caption"),
    ("list", "original", "Extracted list block"),
    ("equation", "original", "Extracted equation"),
    ("abstract", "original", "Abstract or executive summary"),
    ("document_summary", "generated", "LLM-generated document summary"),
    ("block_summary", "generated", "LLM-generated block summary"),
    ("junk", "original", "Frontmatter boilerplate (copyright, reviewer info, etc.)"),
    # Legal / regulatory block types
    ("provision", "original", "Statutory provision or rule"),
    ("definition", "original", "Defined term"),
    ("worksheet", "original", "Tax worksheet or calculation"),
    ("example", "original", "Worked example or illustration"),
    ("form_ref", "original", "Reference to a specific form"),
    ("amendment", "original", "Amendment or revision note"),
]


class BlockType(Base):
    __tablename__ = "block_types"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    provenance: Mapped[str] = mapped_column(
        String, nullable=False
    )  # original|generated
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


def seed_block_types(session: Session) -> None:
    """Insert seed rows into block_types if they don't exist."""
    for name, prov, desc in BLOCK_TYPE_SEEDS:
        if not session.get(BlockType, name):
            session.add(BlockType(name=name, provenance=prov, description=desc))
    session.commit()


# ---------------------------------------------------------------------------
# corpora — document type registry (string PK, config table)
# ---------------------------------------------------------------------------

CORPUS_SEEDS = [
    (
        "papers",
        "Scientific Papers",
        "paper",
        "ingestion",
        "{author}{year}{keyword}",
        '["papers"]',
        "Peer-reviewed scientific literature",
    ),
    (
        "irs",
        "IRS Publications",
        "reference",
        "ingestion",
        "irs-pub{number}-{year}",
        '["irs", "tax", "us"]',
        "US Internal Revenue Service publications and form instructions",
    ),
    (
        "us-code",
        "US Code",
        "statute",
        "ingestion",
        "usc{title}-s{section}",
        '["us-code", "law", "us"]',
        "United States Code \u2014 federal statutes",
    ),
    (
        "ie-acts",
        "Irish Acts",
        "statute",
        "ingestion",
        "ie-{abbrev}{year}",
        '["ie-acts", "law", "ie"]',
        "Acts of the Oireachtas (Irish primary legislation)",
    ),
    (
        "ie-revenue",
        "Irish Revenue Manuals",
        "reference",
        "ingestion",
        "ie-tdm-{id}",
        '["ie-revenue", "tax", "ie"]',
        "Revenue Tax and Duty Manuals (Ireland)",
    ),
    (
        "notes",
        "Notes & Annotations",
        "note",
        "direct",
        "note:{hash}",
        '["notes"]',
        "User and agent annotations on documents and blocks",
    ),
    (
        "todos",
        "Todo Items",
        "todo",
        "direct",
        "todo:{title}",
        '["todos"]',
        "Task items with state, priority, and due dates",
    ),
    (
        "wiki",
        "Knowledge Wiki",
        "wiki",
        "direct",
        "wiki:{title}",
        '["wiki"]',
        "Agent-writable knowledge pages",
    ),
    (
        "journal",
        "Journal",
        "journal",
        "direct",
        "journal:{title}",
        '["journal"]',
        "Conversation summaries and decisions",
    ),
]


class Corpus(Base):
    __tablename__ = "corpora"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    label: Mapped[str] = mapped_column(String, nullable=False)
    handler: Mapped[str] = mapped_column(String, nullable=False, default="paper")
    write_policy: Mapped[str] = mapped_column(
        String, nullable=False, default="ingestion"
    )  # ingestion | direct | system
    slug_pattern: Mapped[str | None] = mapped_column(String, nullable=True)
    default_tags: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


def seed_corpora(session: Session) -> None:
    """Insert or update seed rows in corpora."""
    for id_, label, handler, write_policy, pattern, tags, desc in CORPUS_SEEDS:
        existing = session.get(Corpus, id_)
        if existing:
            # Update write_policy for existing corpora (migration)
            if not existing.write_policy or existing.write_policy == "ingestion":
                existing.write_policy = write_policy
        else:
            session.add(
                Corpus(
                    id=id_,
                    label=label,
                    handler=handler,
                    write_policy=write_policy,
                    slug_pattern=pattern,
                    default_tags=tags,
                    description=desc,
                )
            )
    session.commit()


# ---------------------------------------------------------------------------
# refs — document identity (one row per known document, ingested or stub)
# ---------------------------------------------------------------------------


class Ref(Base):
    __tablename__ = "refs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    corpus_id: Mapped[str] = mapped_column(
        ForeignKey("corpora.id"), nullable=False, default="papers"
    )
    slug: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    doi: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    s2_id: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    arxiv_id: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    authors: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    published_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    keywords: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    meta: Mapped[str | None] = mapped_column(
        "metadata", Text, nullable=True
    )  # JSON object
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC).replace(tzinfo=None),
    )

    # 1:1 optional → ingested content
    paper: Mapped[Paper | None] = relationship(
        back_populates="ref", cascade="all, delete-orphan", uselist=False
    )
    # 1:N → blocks (including abstract, summaries)
    blocks: Mapped[list[Block]] = relationship(
        back_populates="ref", cascade="all, delete-orphan"
    )

    # Citations where this ref is the citer
    cites: Mapped[list[Citation]] = relationship(
        back_populates="citer",
        foreign_keys="Citation.citing_id",
        cascade="all, delete-orphan",
    )
    # Citations where this ref is the cited
    cited_by: Mapped[list[Citation]] = relationship(
        back_populates="cited",
        foreign_keys="Citation.cited_id",
    )

    __table_args__ = (
        Index("idx_refs_year", "year"),
        Index("idx_refs_corpus", "corpus_id"),
        Index("idx_refs_slug", "slug"),
    )

    __mapper_args__ = {
        "polymorphic_on": "corpus_id",
        "polymorphic_identity": "papers",
    }

    # ── Metadata helpers ──────────────────────────────────────────────

    @property
    def _meta(self) -> dict[str, Any]:
        """Parsed metadata JSON (cached per access)."""
        import json as _json

        if not self.meta:
            return {}
        try:
            return _json.loads(self.meta)
        except (ValueError, TypeError):
            return {}

    def _set_meta_field(self, key: str, value: Any) -> None:
        """Set a single field in the metadata JSON."""
        import json as _json

        m = self._meta
        m[key] = value
        self.meta = _json.dumps(m)

    # Paper-specific accessors (on base class for backward compat)
    @property
    def journal(self) -> str | None:
        return self._meta.get("journal")

    @property
    def entry_type(self) -> str:
        return self._meta.get("entry_type", "article")

    @property
    def source(self) -> str | None:
        return self._meta.get("source")

    @property
    def retracted(self) -> bool:
        return self._meta.get("retracted", False)

    @property
    def retraction_note(self) -> str | None:
        return self._meta.get("retraction_note")

    def to_dict(self) -> dict[str, Any]:
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        # Expose metadata fields at top level for backward compat
        d.update(self._meta)
        # Merge paper fields if ingested
        if self.paper:
            d.update(self.paper.to_dict())
        return d

    @property
    def is_ingested(self) -> bool:
        return self.paper is not None


# ---------------------------------------------------------------------------
# papers — ingestion receipt (1:1 extension of refs, any corpus)
# ---------------------------------------------------------------------------


class Paper(Base):
    __tablename__ = "papers"

    ref_id: Mapped[int] = mapped_column(
        ForeignKey("refs.id", ondelete="CASCADE"), primary_key=True
    )
    pdf_hash: Mapped[str] = mapped_column(String, nullable=False)
    bundle_path: Mapped[str] = mapped_column(Text, nullable=False)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    supplements: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC).replace(tzinfo=None),
    )

    ref: Mapped[Ref] = relationship(back_populates="paper")

    __table_args__ = (Index("idx_papers_pdf_hash", "pdf_hash"),)

    def to_dict(self) -> dict[str, Any]:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# ---------------------------------------------------------------------------
# blocks — text chunks + synthetic blocks (FK → refs)
# ---------------------------------------------------------------------------


class Block(Base):
    __tablename__ = "blocks"

    node_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile: Mapped[str] = mapped_column(String, primary_key=True, default="default")
    ref_id: Mapped[int] = mapped_column(
        ForeignKey("refs.id", ondelete="CASCADE"), nullable=False
    )
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    block_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    block_type: Mapped[str] = mapped_column(
        ForeignKey("block_types.name"), nullable=False, default="text"
    )
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    supplement: Mapped[str | None] = mapped_column(String, nullable=True)
    section_path: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    bbox_x0: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_y0: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_x1: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_y1: Mapped[float | None] = mapped_column(Float, nullable=True)

    # embedding column: added dynamically for pgvector backends
    # See add_pgvector_column() below.

    ref: Mapped[Ref] = relationship(back_populates="blocks")

    __table_args__ = (
        Index("idx_blocks_ref_page", "ref_id", "page"),
        Index("idx_blocks_ref_type", "ref_id", "block_type"),
        Index("idx_blocks_ref_supplement", "ref_id", "supplement"),
    )

    def to_dict(self) -> dict[str, Any]:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# ---------------------------------------------------------------------------
# link_types — relation registry (name PK, inverse label)
# ---------------------------------------------------------------------------

LINK_TYPE_SEEDS = [
    ("cites", "cited_by", "Formal citation or reference"),
    ("discusses", "discussed_in", "Substantive discussion of"),
    ("summarizes", "summarized_by", "Distilled summary of"),
    ("annotates", "annotated_by", "Note or annotation on"),
    ("contradicts", "contradicted_by", "Contradictory claim"),
    ("supports", "supported_by", "Supporting evidence"),
    ("supersedes", "superseded_by", "Newer version replaces older"),
    ("cross_references", "cross_referenced_by", "Legal/statutory cross-ref"),
    ("responds_to", "has_response", "Conversation response"),
    ("depends_on", "depended_on_by", "Prerequisite relationship"),
    ("contains", "contained_in", "Parent-child hierarchy"),
    ("references", "referenced_by", "Generic inline reference"),
]


class LinkType(Base):
    __tablename__ = "link_types"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    inverse: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


def seed_link_types(session: Session) -> None:
    """Insert seed rows into link_types if they don't exist."""
    for name, inverse, desc in LINK_TYPE_SEEDS:
        if not session.get(LinkType, name):
            session.add(LinkType(name=name, inverse=inverse, description=desc))
    session.commit()


# ---------------------------------------------------------------------------
# links — slug-based edges between refs or blocks
# ---------------------------------------------------------------------------


class Link(Base):
    __tablename__ = "links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    src_slug: Mapped[str] = mapped_column(String, nullable=False)
    src_node_id: Mapped[str | None] = mapped_column(String, nullable=True)
    dst_slug: Mapped[str] = mapped_column(String, nullable=False)
    dst_node_id: Mapped[str | None] = mapped_column(String, nullable=True)
    relation: Mapped[str] = mapped_column(
        ForeignKey("link_types.name"), nullable=False, default="cites"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC).replace(tzinfo=None),
    )

    link_type: Mapped[LinkType] = relationship()

    __table_args__ = (
        Index("idx_links_src", "src_slug"),
        Index("idx_links_dst", "dst_slug"),
        Index("idx_links_src_node", "src_slug", "src_node_id"),
        Index("idx_links_dst_node", "dst_slug", "dst_node_id"),
        Index("idx_links_relation", "relation"),
    )

    def to_dict(self) -> dict[str, Any]:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# ---------------------------------------------------------------------------
# DEPRECATED: citations — use links with relation='cites' instead
# ---------------------------------------------------------------------------


class Citation(Base):
    __tablename__ = "citations"

    citing_id: Mapped[int] = mapped_column(
        ForeignKey("refs.id", ondelete="CASCADE"), primary_key=True
    )
    cited_id: Mapped[int] = mapped_column(
        ForeignKey("refs.id", ondelete="CASCADE"), primary_key=True
    )

    citer: Mapped[Ref] = relationship(back_populates="cites", foreign_keys=[citing_id])
    cited: Mapped[Ref] = relationship(
        back_populates="cited_by", foreign_keys=[cited_id]
    )

    __table_args__ = (Index("idx_cited", "cited_id"),)


# ---------------------------------------------------------------------------
# DEPRECATED: notes — use refs in 'notes' corpus with 'annotates' links
# ---------------------------------------------------------------------------


class Note(Base):
    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ref_id: Mapped[int | None] = mapped_column(
        ForeignKey("refs.id", ondelete="CASCADE"), nullable=True
    )
    block_node_id: Mapped[str | None] = mapped_column(String, nullable=True)
    block_profile: Mapped[str | None] = mapped_column(String, nullable=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    origin: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # e.g. "reto", "claude", "bot"
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC).replace(tzinfo=None),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC).replace(tzinfo=None),
        onupdate=lambda: datetime.now(UTC).replace(tzinfo=None),
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["block_node_id", "block_profile"],
            ["blocks.node_id", "blocks.profile"],
            ondelete="CASCADE",
        ),
        Index("idx_notes_ref", "ref_id"),
        Index("idx_notes_block", "block_node_id", "block_profile"),
    )

    def to_dict(self) -> dict[str, Any]:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# ---------------------------------------------------------------------------
# blocks_v — convenience view joining provenance from block_types
# ---------------------------------------------------------------------------

BLOCKS_VIEW_SQL = """\
CREATE OR REPLACE VIEW blocks_v AS
SELECT b.*, bt.provenance, bt.description AS type_description
FROM blocks b
JOIN block_types bt ON b.block_type = bt.name"""


def create_blocks_view(engine) -> None:
    """Create the blocks_v view (call after create_all)."""
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text(BLOCKS_VIEW_SQL))


# ---------------------------------------------------------------------------
# pgvector helper
# ---------------------------------------------------------------------------


def add_pgvector_column(embed_dim: int = 384) -> None:
    """Add a pgvector ``embedding`` column to the Block model.

    Call this BEFORE ``create_all()`` when the backend is Postgres
    with pgvector. For SQLite/MySQL, this is a no-op (blocks store
    text only; vectors live in Chroma via LlamaIndex).
    """
    from pgvector.sqlalchemy import Vector
    from sqlalchemy.orm import column_property

    if "embedding" not in {c.name for c in Block.__table__.columns}:
        Block.__table__.append_column(
            mapped_column("embedding", Vector(embed_dim), nullable=True).column
        )
    # Also register as ORM attribute (append_column only touches Table)
    if not hasattr(Block, "embedding"):
        Block.embedding = column_property(Block.__table__.c.embedding)
