"""Tests for the core Store class."""

from __future__ import annotations

import sqlite3

import pytest

from acatome_store.config import StoreConfig
from acatome_store.models import Ref
from acatome_store.store import Store


class TestIngest:
    def test_ingest_single(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle)
        assert isinstance(ref_id, int)

    def test_ingest_dedup_same_hash(self, store, sample_bundle):
        ref_id1 = store.ingest(sample_bundle)
        ref_id2 = store.ingest(sample_bundle)
        assert ref_id1 == ref_id2
        assert store.stats()["total_papers"] == 1

    def test_ingest_atomic_replace(self, store, sample_bundle):
        store.ingest(sample_bundle)
        # Re-ingest same DOI → dedup, not replace
        store.ingest(sample_bundle)
        assert store.stats()["total_papers"] == 1

    def test_ingest_multiple(self, store, sample_bundle, second_bundle):
        store.ingest(sample_bundle)
        store.ingest(second_bundle)
        assert store.stats()["total_papers"] == 2

    def test_slug_collision_fails(self, store, sample_bundle, tmp_path):
        import gzip
        import json

        store.ingest(sample_bundle)

        # Create a bundle with same slug but different DOI
        data = {
            "header": {
                "paper_id": "doi:10.9999/different",
                "slug": "smith2024quantum",
                "title": "Different Paper",
                "authors": [{"name": "Smith"}],
                "year": 2024,
                "doi": "10.9999/different",
                "pdf_hash": "c" * 64,
                "page_count": 5,
                "source": "crossref",
                "verified": True,
                "verify_warnings": [],
                "extracted_at": "2024-01-01T00:00:00+00:00",
            },
            "blocks": [],
            "enrichment_meta": None,
        }
        collision_path = tmp_path / "collision.acatome"
        with gzip.open(collision_path, "wt") as f:
            json.dump(data, f)

        ref_id = store.ingest(collision_path)
        # Should auto-disambiguate to smith2024quantuma
        with store._Session() as session:
            ref = session.get(Ref, ref_id)
            assert ref.slug == "smith2024quantuma"


class TestRefreshOnReingest:
    """Re-ingest should repair stale garbage metadata when the new bundle is better."""

    @staticmethod
    def _write_bundle(tmp_path, name, header_overrides, blocks=None):
        import gzip
        import json

        base_header = {
            "paper_id": "",
            "slug": "",
            "title": "",
            "authors": [],
            "year": None,
            "doi": None,
            "arxiv_id": None,
            "s2_id": None,
            "journal": None,
            "abstract": "",
            "entry_type": "article",
            "keywords": [],
            "pdf_hash": "0" * 64,
            "page_count": 1,
            "source": "test",
            "verified": False,
            "verify_warnings": [],
            "extracted_at": "2024-01-01T00:00:00+00:00",
        }
        base_header.update(header_overrides)
        data = {
            "header": base_header,
            "blocks": blocks
            or [
                {
                    "node_id": f"{base_header['slug']}-p00-000",
                    "page": 0,
                    "type": "text",
                    "text": "body text",
                    "section_path": [],
                    "bbox": None,
                    "embeddings": {},
                    "summary": None,
                }
            ],
            "enrichment_meta": None,
        }
        path = tmp_path / f"{name}.acatome"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    def test_garbage_title_replaced_by_clean_title(self, store, tmp_path):
        """Bug: stale InDesign-filename title survives re-ingest of same PDF."""
        garbage = self._write_bundle(
            tmp_path,
            "stale",
            {
                "slug": "simpson2007nmat",
                "title": "nmat1849 Geim Progress Article.indd",
                "authors": [],
                "year": None,
                "doi": None,
                "pdf_hash": "d" * 64,
                "verified": False,
                "source": "embedded",
            },
        )
        clean = self._write_bundle(
            tmp_path,
            "clean",
            {
                "slug": "geim2007rise",
                "title": "The rise of graphene",
                "authors": [{"name": "Geim, A. K."}, {"name": "Novoselov, K. S."}],
                "year": 2007,
                "doi": "10.1038/nmat1849",
                "journal": "Nature Materials",
                "pdf_hash": "d" * 64,  # same PDF
                "verified": True,
                "source": "crossref_filename",
            },
        )

        ref_id_1 = store.ingest(garbage)
        ref_id_2 = store.ingest(clean)
        assert ref_id_1 == ref_id_2  # still deduped

        with store._Session() as session:
            ref = session.get(Ref, ref_id_2)
            assert ref.title == "The rise of graphene"
            assert ref.doi == "10.1038/nmat1849"
            assert ref.year == 2007
            assert ref.slug == "geim2007rise"
            assert ref.paper.verified is True

    def test_fill_blanks_on_verified_reingest(self, store, tmp_path):
        """Original ingest with missing DOI/year; re-ingest with verified data fills them."""
        sparse = self._write_bundle(
            tmp_path,
            "sparse",
            {
                "slug": "anon2024paper",
                "title": "Some Title",
                "pdf_hash": "e" * 64,
                "verified": False,
            },
        )
        full = self._write_bundle(
            tmp_path,
            "full",
            {
                "slug": "anon2024paper",
                "title": "Some Title",
                "authors": [{"name": "Doe, Jane"}],
                "year": 2024,
                "doi": "10.1234/fill-me-in",
                "pdf_hash": "e" * 64,
                "verified": True,
            },
        )

        store.ingest(sparse)
        ref_id = store.ingest(full)

        with store._Session() as session:
            ref = session.get(Ref, ref_id)
            assert ref.doi == "10.1234/fill-me-in"
            assert ref.year == 2024
            assert ref.authors  # populated

    def test_verified_metadata_not_overwritten_by_unverified(self, store, tmp_path):
        """Safety: a verified Ref must NOT be clobbered by a later unverified bundle."""
        verified = self._write_bundle(
            tmp_path,
            "verified",
            {
                "slug": "real2024paper",
                "title": "Real Paper Title",
                "authors": [{"name": "Real, Author"}],
                "year": 2024,
                "doi": "10.1234/real",
                "pdf_hash": "a1" * 32,
                "verified": True,
            },
        )
        bad = self._write_bundle(
            tmp_path,
            "bad",
            {
                "slug": "junk",
                "title": "some-pdf-filename.indd",
                "year": 1999,
                "doi": "10.9999/wrong",
                "pdf_hash": "a1" * 32,  # same PDF
                "verified": False,
            },
        )

        ref_id = store.ingest(verified)
        store.ingest(bad)

        with store._Session() as session:
            ref = session.get(Ref, ref_id)
            # Clean title + verified DOI/year preserved
            assert ref.title == "Real Paper Title"
            assert ref.doi == "10.1234/real"
            assert ref.year == 2024
            assert ref.slug == "real2024paper"

    def test_clean_title_not_downgraded_by_verified_garbage(self, store, tmp_path):
        """Safety: a verified bundle with a garbage title must not overwrite a clean title."""
        clean_unverified = self._write_bundle(
            tmp_path,
            "clean_unverified",
            {
                "slug": "geim2007rise",
                "title": "The rise of graphene",
                "year": 2007,
                "pdf_hash": "d4" * 32,
                "verified": False,
            },
        )
        verified_garbage = self._write_bundle(
            tmp_path,
            "verified_garbage",
            {
                "slug": "bad",
                "title": "nmat1849 Geim Progress Article.indd",
                "doi": "10.1038/nmat1849",
                "pdf_hash": "d4" * 32,  # same PDF
                "verified": True,
            },
        )

        ref_id = store.ingest(clean_unverified)
        store.ingest(verified_garbage)

        with store._Session() as session:
            ref = session.get(Ref, ref_id)
            # Clean title preserved…
            assert ref.title == "The rise of graphene"
            # …but the DOI from the verified bundle still filled in
            assert ref.doi == "10.1038/nmat1849"
            # …and Paper.verified upgraded to True
            assert ref.paper.verified is True

    def test_slug_upgrade_skipped_on_collision(self, store, tmp_path):
        """Upgrade must not overwrite slug if the new slug collides with another Ref."""
        # Seed another Ref with the target slug
        other = self._write_bundle(
            tmp_path,
            "other",
            {
                "slug": "geim2007rise",
                "title": "Something Else Entirely",
                "doi": "10.1111/other",
                "pdf_hash": "b2" * 32,
                "verified": True,
            },
        )
        garbage = self._write_bundle(
            tmp_path,
            "stale2",
            {
                "slug": "simpson2007nmat",
                "title": "nmat1849 Geim Progress Article.indd",
                "pdf_hash": "c3" * 32,
                "verified": False,
            },
        )
        clean = self._write_bundle(
            tmp_path,
            "clean2",
            {
                "slug": "geim2007rise",  # collides with 'other'
                "title": "The rise of graphene",
                "doi": "10.1038/nmat1849",
                "year": 2007,
                "pdf_hash": "c3" * 32,
                "verified": True,
            },
        )

        store.ingest(other)
        stale_id = store.ingest(garbage)
        store.ingest(clean)

        with store._Session() as session:
            ref = session.get(Ref, stale_id)
            # Title/DOI/year upgraded…
            assert ref.title == "The rise of graphene"
            assert ref.doi == "10.1038/nmat1849"
            assert ref.year == 2007
            # …but slug preserved (would collide with 'other')
            assert ref.slug == "simpson2007nmat"


class TestReembed:
    """Test embedding model mismatch detection and re-embed on ingest."""

    def test_matching_model_no_reembed(self, store, tmp_path):
        """When bundle model matches system config, embeddings pass through."""
        import gzip
        import json
        from unittest.mock import patch

        dim = store._config.embed_dim
        data = {
            "header": {
                "paper_id": "doi:10.1234/match",
                "slug": "match2024",
                "title": "Match",
                "authors": [],
                "year": 2024,
                "doi": "10.1234/match",
                "pdf_hash": "e" * 64,
                "page_count": 1,
                "source": "test",
                "verified": True,
                "verify_warnings": [],
                "extracted_at": "2024-01-01T00:00:00+00:00",
            },
            "blocks": [
                {
                    "node_id": "match-p00-000",
                    "page": 0,
                    "type": "text",
                    "text": "Some text about matching embeddings.",
                    "section_path": [],
                    "bbox": None,
                    "embeddings": {"default": [0.1] * dim},
                    "summary": None,
                }
            ],
            "enrichment_meta": {
                "embedding_models": {
                    "default": {"model": store._config.embed_model, "dim": dim}
                }
            },
        }
        path = tmp_path / "match.acatome"
        with gzip.open(path, "wt") as f:
            json.dump(data, f)

        from acatome_store import store as store_mod

        with patch.object(
            store_mod, "_reembed_blocks", wraps=store_mod._reembed_blocks
        ) as mock_reembed:
            store.ingest(path)
            mock_reembed.assert_not_called()

    def test_mismatched_model_triggers_reembed(self, store, tmp_path):
        """When bundle model differs from system config, blocks are re-embedded."""
        import gzip
        import json
        from unittest.mock import patch

        dim = store._config.embed_dim
        data = {
            "header": {
                "paper_id": "doi:10.1234/mismatch",
                "slug": "mismatch2024",
                "title": "Mismatch",
                "authors": [],
                "year": 2024,
                "doi": "10.1234/mismatch",
                "pdf_hash": "f" * 64,
                "page_count": 1,
                "source": "test",
                "verified": True,
                "verify_warnings": [],
                "extracted_at": "2024-01-01T00:00:00+00:00",
            },
            "blocks": [
                {
                    "node_id": "mismatch-p00-000",
                    "page": 0,
                    "type": "text",
                    "text": "Some text about mismatched embeddings.",
                    "section_path": [],
                    "bbox": None,
                    "embeddings": {"default": [0.1] * 768},
                    "summary": None,
                }
            ],
            "enrichment_meta": {
                "embedding_models": {
                    "default": {"model": "some-other-model", "dim": 768}
                }
            },
        }
        path = tmp_path / "mismatch.acatome"
        with gzip.open(path, "wt") as f:
            json.dump(data, f)

        from acatome_store import store as store_mod

        dummy_embedder = lambda texts: [[0.0] * dim for _ in texts]
        with (
            patch.object(
                store_mod, "_reembed_blocks", wraps=store_mod._reembed_blocks
            ) as mock_reembed,
            patch.object(store_mod, "_get_embedder", return_value=dummy_embedder),
        ):
            store.ingest(path)
            mock_reembed.assert_called_once()

    def test_no_enrichment_meta_triggers_reembed(self, store, tmp_path):
        """When bundle has no enrichment_meta, blocks are re-embedded."""
        import gzip
        import json
        from unittest.mock import patch

        dim = store._config.embed_dim
        data = {
            "header": {
                "paper_id": "doi:10.1234/noenrich",
                "slug": "noenrich2024",
                "title": "No Enrich",
                "authors": [],
                "year": 2024,
                "doi": "10.1234/noenrich",
                "pdf_hash": "1a" * 32,
                "page_count": 1,
                "source": "test",
                "verified": True,
                "verify_warnings": [],
                "extracted_at": "2024-01-01T00:00:00+00:00",
            },
            "blocks": [
                {
                    "node_id": "noenrich-p00-000",
                    "page": 0,
                    "type": "text",
                    "text": "Some text without any embeddings at all.",
                    "section_path": [],
                    "bbox": None,
                    "embeddings": {},
                    "summary": None,
                }
            ],
            "enrichment_meta": None,
        }
        path = tmp_path / "noenrich.acatome"
        with gzip.open(path, "wt") as f:
            json.dump(data, f)

        from acatome_store import store as store_mod

        dummy_embedder = lambda texts: [[0.0] * dim for _ in texts]
        with (
            patch.object(
                store_mod, "_reembed_blocks", wraps=store_mod._reembed_blocks
            ) as mock_reembed,
            patch.object(store_mod, "_get_embedder", return_value=dummy_embedder),
        ):
            store.ingest(path)
            mock_reembed.assert_called_once()


class TestGet:
    def test_get_by_ref_id(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle)
        paper = store.get(ref_id)
        assert paper is not None
        assert paper["title"] == "Quantum Error Correction in Practice"

    def test_get_by_slug(self, store, sample_bundle):
        store.ingest(sample_bundle)
        paper = store.get("smith2024quantum")
        assert paper is not None
        assert paper["title"] == "Quantum Error Correction in Practice"

    def test_get_by_doi(self, store, sample_bundle):
        store.ingest(sample_bundle)
        paper = store.get("10.1038/s41567-024-1234-5")
        assert paper is not None

    def test_get_not_found(self, store):
        assert store.get("nonexistent") is None

    def test_get_blocks(self, store, sample_bundle):
        store.ingest(sample_bundle)
        blocks = store.get_blocks("smith2024quantum")
        # 1 text block + 1 abstract block
        assert len(blocks) == 2
        types = {b["block_type"] for b in blocks}
        assert "text" in types
        assert "abstract" in types

    def test_abstract_is_block(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle)
        blocks = store.get_blocks(ref_id)
        abstract = [b for b in blocks if b["block_type"] == "abstract"]
        assert len(abstract) == 1
        assert abstract[0]["text"] == "We present a new approach..."
        assert abstract[0]["page"] is None
        assert abstract[0]["block_index"] is None


class TestDelete:
    def test_delete(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle)
        assert store.delete("smith2024quantum") is True
        # Slug is on Ref now, so stub is still findable by slug
        stub = store.get("smith2024quantum")
        assert stub is not None
        assert stub["slug"] == "smith2024quantum"
        # Paper (ingestion receipt) is gone — no pdf_hash etc.
        assert "pdf_hash" not in stub or stub.get("pdf_hash") is None
        # Also findable by DOI
        stub2 = store.get("10.1038/s41567-024-1234-5")
        assert stub2 is not None

    def test_delete_not_found(self, store):
        assert store.delete("nonexistent") is False


class TestList:
    def test_list_empty(self, store):
        assert store.list_papers() == []

    def test_list_papers(self, store, sample_bundle, second_bundle):
        store.ingest(sample_bundle)
        store.ingest(second_bundle)
        papers = store.list_papers()
        assert len(papers) == 2

    def test_list_limit(self, store, sample_bundle, second_bundle):
        store.ingest(sample_bundle)
        store.ingest(second_bundle)
        papers = store.list_papers(limit=1)
        assert len(papers) == 1

    def test_list_has_ref_id(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle)
        papers = store.list_papers()
        assert papers[0]["ref_id"] == ref_id


class TestStats:
    def test_stats_empty(self, store):
        s = store.stats()
        assert s["total_papers"] == 0
        assert s["total_refs"] == 0
        assert s["verified"] == 0

    def test_stats_after_ingest(self, store, sample_bundle):
        store.ingest(sample_bundle)
        s = store.stats()
        assert s["total_papers"] == 1
        assert s["total_refs"] == 1
        assert s["verified"] == 1

    def test_stats_ref_persists_after_delete(self, store, sample_bundle):
        store.ingest(sample_bundle)
        store.delete("smith2024quantum")
        s = store.stats()
        assert s["total_papers"] == 0
        assert s["total_refs"] == 1


class TestNotes:
    def test_add_note_on_paper(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle)
        note_id = store.add_note("Great paper", ref_id=ref_id, title="Review")
        assert isinstance(note_id, int)
        notes = store.get_notes(ref_id=ref_id)
        assert len(notes) == 1
        assert notes[0]["content"] == "Great paper"
        assert notes[0]["title"] == "Review"

    def test_add_note_on_block(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle)
        blocks = store.get_blocks(ref_id)
        text_blocks = [b for b in blocks if b["block_type"] == "text"]
        node_id = text_blocks[0]["node_id"]
        note_id = store.add_note("Key finding", block_node_id=node_id)
        notes = store.get_notes(block_node_id=node_id)
        assert len(notes) == 1
        assert notes[0]["content"] == "Key finding"

    def test_add_note_with_tags(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle)
        store.add_note("Tagged", ref_id=ref_id, tags=["important", "review"])
        notes = store.get_notes(ref_id=ref_id)
        import json

        assert json.loads(notes[0]["tags"]) == ["important", "review"]

    def test_multiple_notes(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle)
        store.add_note("Note 1", ref_id=ref_id)
        store.add_note("Note 2", ref_id=ref_id)
        notes = store.get_notes(ref_id=ref_id)
        assert len(notes) == 2

    def test_update_note(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle)
        note_id = store.add_note("Draft", ref_id=ref_id)
        assert store.update_note(note_id, content="Final version") is True
        notes = store.get_notes(ref_id=ref_id)
        assert notes[0]["content"] == "Final version"

    def test_update_nonexistent(self, store):
        assert store.update_note(9999, content="nope") is False

    def test_delete_note(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle)
        note_id = store.add_note("Temporary", ref_id=ref_id)
        assert store.delete_note(note_id) is True
        assert store.get_notes(ref_id=ref_id) == []

    def test_delete_nonexistent(self, store):
        assert store.delete_note(9999) is False

    def test_notes_cascade_on_paper_delete(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle)
        blocks = store.get_blocks(ref_id)
        text_blocks = [b for b in blocks if b["block_type"] == "text"]
        node_id = text_blocks[0]["node_id"]
        store.add_note("Paper note", ref_id=ref_id)
        store.add_note("Block note", block_node_id=node_id)
        store.delete("smith2024quantum")
        # Block note should be gone (block deleted via CASCADE)
        assert store.get_notes(block_node_id=node_id) == []

    def test_add_note_with_origin(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle)
        note_id = store.add_note("Human note", ref_id=ref_id, origin="reto")
        notes = store.get_notes(ref_id=ref_id)
        assert notes[0]["origin"] == "reto"

    def test_add_note_bot_origin(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle)
        store.add_note("Bot note", ref_id=ref_id, origin="bot")
        notes = store.get_notes(ref_id=ref_id)
        assert notes[0]["origin"] == "bot"

    def test_add_note_no_origin(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle)
        store.add_note("No origin", ref_id=ref_id)
        notes = store.get_notes(ref_id=ref_id)
        assert notes[0]["origin"] is None


class TestTags:
    def test_ingest_with_tags(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle, tags=["chlorine-evolution", "review"])
        tags = store.get_tags(ref_id)
        assert tags == ["chlorine-evolution", "review"]

    def test_ingest_dedup_merges_tags(self, store, sample_bundle):
        store.ingest(sample_bundle, tags=["batch-1"])
        store.ingest(sample_bundle, tags=["batch-2"])
        tags = store.get_tags("smith2024quantum")
        assert "batch-1" in tags
        assert "batch-2" in tags

    def test_add_tags(self, store, sample_bundle):
        ref_id = store.ingest(sample_bundle)
        assert store.get_tags(ref_id) == []
        store.add_tags(ref_id, ["new-tag"])
        assert store.get_tags(ref_id) == ["new-tag"]

    def test_remove_tags(self, store, sample_bundle):
        store.ingest(sample_bundle, tags=["a", "b", "c"])
        store.remove_tags("smith2024quantum", ["b"])
        assert store.get_tags("smith2024quantum") == ["a", "c"]

    def test_find_by_tag(self, store, sample_bundle, second_bundle):
        store.ingest(sample_bundle, tags=["shared"])
        store.ingest(second_bundle, tags=["shared", "extra"])
        results = store.find_by_tag("shared")
        assert len(results) == 2
        results = store.find_by_tag("extra")
        assert len(results) == 1

    def test_list_tags(self, store, sample_bundle, second_bundle):
        store.ingest(sample_bundle, tags=["a", "b"])
        store.ingest(second_bundle, tags=["b", "c"])
        tag_counts = store.list_tags()
        assert tag_counts["b"] == 2
        assert tag_counts["a"] == 1
        assert tag_counts["c"] == 1

    def test_tags_not_found(self, store):
        assert store.get_tags("nonexistent") == []
        assert not store.add_tags("nonexistent", ["x"])
        assert not store.remove_tags("nonexistent", ["x"])


class TestSupplements:
    def test_ingest_supplement(self, store, sample_bundle, supplement_bundle):
        ref_id = store.ingest(sample_bundle)
        result = store.ingest_supplement("smith2024quantum", supplement_bundle, "s1")
        assert result == ref_id

    def test_supplement_blocks_separate(self, store, sample_bundle, supplement_bundle):
        store.ingest(sample_bundle)
        store.ingest_supplement("smith2024quantum", supplement_bundle, "s1")
        # Main paper blocks (supplement=None by default)
        main = store.get_blocks("smith2024quantum")
        assert all(b.get("supplement") is None for b in main)
        # Supplement blocks
        supp = store.get_blocks("smith2024quantum", supplement="s1")
        assert len(supp) == 2
        assert all(b["supplement"] == "s1" for b in supp)

    def test_supplement_blocks_all(self, store, sample_bundle, supplement_bundle):
        store.ingest(sample_bundle)
        store.ingest_supplement("smith2024quantum", supplement_bundle, "s1")
        all_blocks = store.get_blocks("smith2024quantum", supplement="*")
        main = store.get_blocks("smith2024quantum")
        supp = store.get_blocks("smith2024quantum", supplement="s1")
        assert len(all_blocks) == len(main) + len(supp)

    def test_get_supplements(self, store, sample_bundle, supplement_bundle):
        store.ingest(sample_bundle)
        assert store.get_supplements("smith2024quantum") == []
        store.ingest_supplement("smith2024quantum", supplement_bundle, "s1")
        assert store.get_supplements("smith2024quantum") == ["s1"]

    def test_supplement_list_updates(self, store, sample_bundle, supplement_bundle):
        store.ingest(sample_bundle)
        store.ingest_supplement("smith2024quantum", supplement_bundle, "s1")
        store.ingest_supplement("smith2024quantum", supplement_bundle, "methods")
        supps = store.get_supplements("smith2024quantum")
        assert supps == ["methods", "s1"]  # sorted

    def test_supplement_reingest(self, store, sample_bundle, supplement_bundle):
        store.ingest(sample_bundle)
        store.ingest_supplement("smith2024quantum", supplement_bundle, "s1")
        # Re-ingest same supplement — should replace
        store.ingest_supplement("smith2024quantum", supplement_bundle, "s1")
        supp = store.get_blocks("smith2024quantum", supplement="s1")
        assert len(supp) == 2  # same count, not doubled

    def test_supplement_toc(self, store, sample_bundle, supplement_bundle):
        store.ingest(sample_bundle)
        store.ingest_supplement("smith2024quantum", supplement_bundle, "s1")
        toc_main = store.get_toc("smith2024quantum")
        toc_supp = store.get_toc("smith2024quantum", supplement="s1")
        assert len(toc_supp) == 2
        # Main TOC should NOT include supplement blocks
        for item in toc_main:
            assert item.get("supplement") is None or "supplement" not in item

    def test_ingest_supplement_parent_not_found(self, store, supplement_bundle):
        with pytest.raises(ValueError, match="not found"):
            store.ingest_supplement("nonexistent", supplement_bundle, "s1")


class TestRetractions:
    def test_retract(self, store, sample_bundle):
        store.ingest(sample_bundle)
        assert store.retract("smith2024quantum", note="Data issues") is True
        paper = store.get("smith2024quantum")
        assert paper["retracted"] is True
        assert paper["retraction_note"] == "Data issues"

    def test_retract_not_found(self, store):
        assert store.retract("nonexistent") is False

    def test_unretract(self, store, sample_bundle):
        store.ingest(sample_bundle)
        store.retract("smith2024quantum", note="Oops")
        assert store.unretract("smith2024quantum") is True
        paper = store.get("smith2024quantum")
        assert paper["retracted"] is False
        assert paper["retraction_note"] is None

    def test_unretract_not_found(self, store):
        assert store.unretract("nonexistent") is False


class TestSQLiteMigration:
    """Regression: Store must auto-migrate legacy SQLite without corpus_id."""

    def test_legacy_refs_gets_corpus_id(self, tmp_path):
        """Create a v2-era SQLite DB (no corpus_id on refs), then open
        Store which should add the missing column automatically."""
        db_path = tmp_path / "store" / "acatome.db"
        db_path.parent.mkdir(parents=True)

        # Create a minimal legacy schema — refs without corpus_id/slug/metadata
        con = sqlite3.connect(str(db_path))
        con.execute(
            "CREATE TABLE refs ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  doi TEXT UNIQUE,"
            "  s2_id TEXT UNIQUE,"
            "  arxiv_id TEXT UNIQUE,"
            "  title TEXT,"
            "  authors TEXT,"
            "  year INTEGER,"
            "  keywords TEXT,"
            "  first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        con.execute(
            "INSERT INTO refs (doi, title, year) "
            "VALUES ('10.1234/test', 'Legacy paper', 2023)"
        )
        con.commit()
        con.close()

        # Open Store — should auto-migrate without error
        cfg = StoreConfig(store_path=tmp_path / "store")
        s = Store(config=cfg)

        # Verify the column exists and has the default value
        with s._Session() as session:
            from sqlalchemy import text

            row = session.execute(
                text("SELECT corpus_id FROM refs WHERE doi = '10.1234/test'")
            ).fetchone()
            assert row is not None
            assert row[0] == "papers"

        # Verify ORM query doesn't crash on corpus_id
        result = s.get("10.1234/test")
        assert result is not None
        assert result["title"] == "Legacy paper"

        s.close()
