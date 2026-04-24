"""Shared fixtures for acatome-store tests.

v1.0.0+ is Postgres-only.  All tests run against a live Postgres with
pgvector via the ``store`` fixture — SQLite/Chroma fallback is gone.

Env vars for connecting to the test Postgres instance:
  PG_TEST_HOST      (default: localhost)
  PG_TEST_PORT      (default: 5432)
  PG_TEST_DB        (default: acatome_test)
  PG_TEST_USER      (default: acatome)
  PG_TEST_PASSWORD  (default: acatome)

If the test Postgres is unreachable, every test using ``store`` is
skipped with a clear reason — but CI should always have one running.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

from acatome_store.config import StoreConfig
from acatome_store.models import Base
from acatome_store.store import Store

# ---------------------------------------------------------------------------
# Postgres connection detection (cached)
# ---------------------------------------------------------------------------

_pg_available: bool | None = None
_pg_url: str = ""


def _check_postgres() -> bool:
    global _pg_available, _pg_url
    if _pg_available is not None:
        return _pg_available

    host = os.environ.get("PG_TEST_HOST", "localhost")
    port = os.environ.get("PG_TEST_PORT", "5432")
    db = os.environ.get("PG_TEST_DB", "acatome_test")
    user = os.environ.get("PG_TEST_USER", "acatome")
    pw = os.environ.get("PG_TEST_PASSWORD", "acatome")
    _pg_url = f"postgresql+psycopg://{user}:{pw}@{host}:{port}/{db}"

    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(_pg_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        _pg_available = True
    except Exception:
        _pg_available = False
    return _pg_available


# ---------------------------------------------------------------------------
# Store fixture (Postgres-only)
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    """Fresh store against a dedicated Postgres test DB.

    Each test gets a fresh schema: the fixture drops the ``blocks_v``
    view (if present) and ``Base.metadata.drop_all`` at teardown so
    the next test starts from a clean DDL + seed state.
    """
    if not _check_postgres():
        pytest.skip(
            "Postgres test DB unreachable — set PG_TEST_* env vars or "
            "run 'createdb acatome_test && psql -c \"CREATE EXTENSION vector\" acatome_test'"
        )

    cfg = StoreConfig(
        store_path=tmp_path / "store",
        _db_url=_pg_url,
    )
    s = Store(config=cfg)
    try:
        yield s
    finally:
        from sqlalchemy import text

        with s._engine.begin() as conn:
            conn.execute(text("DROP VIEW IF EXISTS blocks_v CASCADE"))
        Base.metadata.drop_all(s._engine)
        s._engine.dispose()


# ---------------------------------------------------------------------------
# Bundle fixtures (backend-independent)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_bundle(tmp_path) -> Path:
    """Create a sample .acatome bundle on disk."""
    data = {
        "header": {
            "paper_id": "doi:10.1038/s41567-024-1234-5",
            "slug": "smith2024quantum",
            "title": "Quantum Error Correction in Practice",
            "authors": [{"name": "Smith, John"}],
            "year": 2024,
            "doi": "10.1038/s41567-024-1234-5",
            "arxiv_id": None,
            "journal": "Nature Physics",
            "abstract": "We present a new approach...",
            "entry_type": "article",
            "s2_id": None,
            "keywords": [],
            "pdf_hash": "a" * 64,
            "page_count": 12,
            "source": "crossref",
            "verified": True,
            "verify_warnings": [],
            "extracted_at": "2024-01-15T12:00:00+00:00",
        },
        "blocks": [
            {
                "node_id": "doi:10.1038/s41567-024-1234-5-p00-000",
                "page": 0,
                "type": "text",
                "text": "Quantum error correction is essential...",
                "section_path": ["1", "Introduction"],
                "bbox": [72, 100, 540, 200],
                "embeddings": {},
                "summary": None,
            }
        ],
        "enrichment_meta": None,
    }
    path = tmp_path / "smith2024quantum.acatome"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def supplement_bundle(tmp_path) -> Path:
    """Create a supplement .acatome bundle (no header metadata needed for supplement)."""
    data = {
        "header": {
            "paper_id": "supplement",
            "slug": "smith2024quantum_s1",
            "title": "Supplementary Materials",
            "authors": [],
            "year": 2024,
            "doi": None,
            "pdf_hash": "d" * 64,
            "page_count": 3,
            "source": "local",
            "verified": False,
            "verify_warnings": [],
            "extracted_at": "2024-01-15T12:00:00+00:00",
        },
        "blocks": [
            {
                "node_id": "supp-s1-p00-000",
                "page": 0,
                "type": "text",
                "text": "Supplementary figure S1 shows the raw data...",
                "section_path": ["S1", "Raw Data"],
                "bbox": [72, 100, 540, 200],
                "embeddings": {},
                "summary": None,
            },
            {
                "node_id": "supp-s1-p00-001",
                "page": 0,
                "type": "figure",
                "text": "Figure S1: Raw data plot",
                "section_path": ["S1", "Raw Data"],
                "bbox": [72, 200, 540, 400],
                "embeddings": {},
                "summary": None,
            },
        ],
        "enrichment_meta": None,
    }
    path = tmp_path / "smith2024quantum_s1.acatome"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def second_bundle(tmp_path) -> Path:
    """Create a second sample bundle with different paper."""
    data = {
        "header": {
            "paper_id": "doi:10.1103/PhysRevLett.123.456",
            "slug": "jones2023surface",
            "title": "Surface Code Thresholds",
            "authors": [{"name": "Jones, Alice"}],
            "year": 2023,
            "doi": "10.1103/PhysRevLett.123.456",
            "arxiv_id": None,
            "journal": "PRL",
            "abstract": "Surface codes...",
            "entry_type": "article",
            "s2_id": None,
            "keywords": [],
            "pdf_hash": "b" * 64,
            "page_count": 8,
            "source": "crossref",
            "verified": True,
            "verify_warnings": [],
            "extracted_at": "2024-02-01T12:00:00+00:00",
        },
        "blocks": [
            {
                "node_id": "doi:10.1103/PhysRevLett.123.456-p00-000",
                "page": 0,
                "type": "text",
                "text": "Surface codes are a family...",
                "section_path": ["1", "Introduction"],
                "bbox": [72, 100, 540, 200],
                "embeddings": {},
                "summary": None,
            }
        ],
        "enrichment_meta": None,
    }
    path = tmp_path / "jones2023surface.acatome"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    return path
