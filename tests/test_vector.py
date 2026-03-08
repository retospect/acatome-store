"""Tests for VectorIndex (LlamaIndex-backed) vector search."""

from __future__ import annotations

import pytest

from acatome_store.config import StoreConfig
from acatome_store.vector import VectorIndex, create_index


@pytest.fixture
def chroma_index(tmp_path) -> VectorIndex:
    config = StoreConfig(store_path=tmp_path, vector_backend="chroma")
    return create_index(config, collection_name="test_blocks")


@pytest.fixture
def blocks_with_embeddings():
    """Sample blocks with dummy 384-dim embeddings."""
    import random

    random.seed(42)

    def _rand_emb():
        return [random.gauss(0, 1) for _ in range(384)]

    return [
        {
            "node_id": "doi:10.1038/test-p00-000",
            "page": 0,
            "type": "text",
            "text": "Quantum error correction is essential for fault-tolerant quantum computing.",
            "section_path": ["1", "Introduction"],
            "embeddings": {"default": _rand_emb()},
        },
        {
            "node_id": "doi:10.1038/test-p00-001",
            "page": 0,
            "type": "text",
            "text": "Surface codes provide high threshold error rates.",
            "section_path": ["2", "Surface Codes"],
            "embeddings": {"default": _rand_emb()},
        },
        {
            "node_id": "doi:10.1038/test-p01-000",
            "page": 1,
            "type": "text",
            "text": "We compare performance of different decoder algorithms.",
            "section_path": ["3", "Results"],
            "embeddings": {"default": _rand_emb()},
        },
        {
            "node_id": "doi:10.1038/test-p01-001",
            "page": 1,
            "type": "section_header",
            "text": "Results",
            "section_path": ["3"],
            "embeddings": {},
        },
    ]


class TestAddBlocks:
    def test_add(self, chroma_index, blocks_with_embeddings):
        n = chroma_index.add_blocks("doi:10.1038/test", blocks_with_embeddings)
        assert n == 3  # 3 blocks have embeddings, 1 (section_header) does not

    def test_count(self, chroma_index, blocks_with_embeddings):
        chroma_index.add_blocks("doi:10.1038/test", blocks_with_embeddings)
        assert chroma_index.count() == 3

    def test_empty_blocks(self, chroma_index):
        n = chroma_index.add_blocks("test", [])
        assert n == 0

    def test_no_embeddings(self, chroma_index):
        blocks = [{"node_id": "x-p00-000", "text": "hi", "embeddings": {}}]
        n = chroma_index.add_blocks("test", blocks)
        assert n == 0

    def test_upsert_idempotent(self, chroma_index, blocks_with_embeddings):
        chroma_index.add_blocks("doi:10.1038/test", blocks_with_embeddings)
        chroma_index.add_blocks("doi:10.1038/test", blocks_with_embeddings)
        assert chroma_index.count() == 3


class TestSearch:
    def test_search_text(self, chroma_index, blocks_with_embeddings):
        chroma_index.add_blocks("doi:10.1038/test", blocks_with_embeddings)
        hits = chroma_index.search_text("quantum error correction")
        assert len(hits) > 0
        assert hits[0]["text"]  # Has text content

    def test_search_top_k(self, chroma_index, blocks_with_embeddings):
        chroma_index.add_blocks("doi:10.1038/test", blocks_with_embeddings)
        hits = chroma_index.search_text("surface codes", top_k=2)
        assert len(hits) <= 2

    def test_search_returns_metadata(self, chroma_index, blocks_with_embeddings):
        chroma_index.add_blocks("doi:10.1038/test", blocks_with_embeddings)
        hits = chroma_index.search_text("decoder algorithms")
        assert len(hits) > 0
        meta = hits[0]["metadata"]
        assert "paper_id" in meta
        assert meta["paper_id"] == "doi:10.1038/test"

    def test_search_empty_index(self, chroma_index):
        hits = chroma_index.search_text("anything")
        assert hits == []


class TestDelete:
    def test_delete_paper(self, chroma_index, blocks_with_embeddings):
        chroma_index.add_blocks("doi:10.1038/test", blocks_with_embeddings)
        assert chroma_index.count() == 3
        chroma_index.delete_paper("doi:10.1038/test")
        assert chroma_index.count() == 0
