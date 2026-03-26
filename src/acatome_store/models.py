"""SQLAlchemy ORM models for acatome-store.

Schema overview:
  ``block_types`` — Lookup table for block kinds (string PK).  Carries a
                    ``provenance`` flag ("original" | "generated") so
                    consumers always know what is the paper vs. what is
                    LLM-generated.
  ``refs``        — Paper identity (auto int PK).  One row per known paper,
                    ingested or stub.  DOI/S2/arxiv are nullable UNIQUE
                    lookup handles.  The citation graph references this table.
  ``papers``      — Ingested content (1:1 FK to refs).  Slug, PDF hash,
                    bundle path, verified flag.
  ``blocks``      — Text chunks + synthetic blocks (FK → refs).  Abstract
                    and paper summary are blocks with special ``block_type``.
  ``citations``   — Directed graph edges.  Both FKs → refs.
  ``notes``       — User annotations on refs or blocks.

Portable across SQLite, Postgres, and MySQL via connection string.
On Postgres, ``blocks.embedding`` uses pgvector for native ANN search.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
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
    ("abstract", "original", "Paper abstract"),
    ("paper_summary", "generated", "LLM-generated paper summary"),
    ("block_summary", "generated", "LLM-generated block summary"),
    ("junk", "original", "Frontmatter boilerplate (copyright, reviewer info, etc.)"),
]


class BlockType(Base):
    __tablename__ = "block_types"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    provenance: Mapped[str] = mapped_column(
        String, nullable=False
    )  # original|generated
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


def seed_block_types(session: Session) -> None:
    """Insert seed rows into block_types if they don't exist."""
    for name, prov, desc in BLOCK_TYPE_SEEDS:
        if not session.get(BlockType, name):
            session.add(BlockType(name=name, provenance=prov, description=desc))
    session.commit()


# ---------------------------------------------------------------------------
# refs — paper identity (one row per known paper, ingested or stub)
# ---------------------------------------------------------------------------


class Ref(Base):
    __tablename__ = "refs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doi: Mapped[Optional[str]] = mapped_column(String, unique=True, nullable=True)
    s2_id: Mapped[Optional[str]] = mapped_column(String, unique=True, nullable=True)
    arxiv_id: Mapped[Optional[str]] = mapped_column(String, unique=True, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    authors: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    journal: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    entry_type: Mapped[str] = mapped_column(String, default="article")
    keywords: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    source: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    retracted: Mapped[bool] = mapped_column(Boolean, default=False)
    retraction_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )

    # 1:1 optional → ingested content
    paper: Mapped[Optional[Paper]] = relationship(
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

    __table_args__ = (Index("idx_refs_year", "year"),)

    def to_dict(self) -> dict[str, Any]:
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        # Merge paper fields if ingested
        if self.paper:
            d.update(self.paper.to_dict())
        return d

    @property
    def is_ingested(self) -> bool:
        return self.paper is not None


# ---------------------------------------------------------------------------
# papers — ingested content (1:1 extension of refs)
# ---------------------------------------------------------------------------


class Paper(Base):
    __tablename__ = "papers"

    ref_id: Mapped[int] = mapped_column(
        ForeignKey("refs.id", ondelete="CASCADE"), primary_key=True
    )
    slug: Mapped[Optional[str]] = mapped_column(String, unique=True, nullable=True)
    pdf_hash: Mapped[str] = mapped_column(String, nullable=False)
    bundle_path: Mapped[str] = mapped_column(Text, nullable=False)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    supplements: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON array
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
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
    page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    block_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    block_type: Mapped[str] = mapped_column(
        ForeignKey("block_types.name"), nullable=False, default="text"
    )
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    supplement: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    section_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    bbox_x0: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_y0: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_x1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_y1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

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
# citations — directed graph edges (both FKs → refs)
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
# notes — user annotations on refs or blocks
# ---------------------------------------------------------------------------


class Note(Base):
    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ref_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("refs.id", ondelete="CASCADE"), nullable=True
    )
    block_node_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    block_profile: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    origin: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # e.g. "reto", "claude", "bot"
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
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
