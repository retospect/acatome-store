"""Tests for the links system (LinkType, Link, store methods)."""

from __future__ import annotations

import pytest

from acatome_store.models import Link, LinkType, Corpus


class TestLinkTypes:
    """link_types table is seeded correctly."""

    def test_link_types_seeded(self, store):
        """All seed link types are present after store init."""
        from acatome_store.models import LINK_TYPE_SEEDS
        from sqlalchemy import select

        with store._Session() as session:
            rows = session.execute(select(LinkType)).scalars().all()
            names = {r.name for r in rows}
            for name, inverse, _desc in LINK_TYPE_SEEDS:
                assert name in names, f"Missing link type: {name}"

    def test_link_type_has_inverse(self, store):
        """Each link type has a non-empty inverse label."""
        from sqlalchemy import select

        with store._Session() as session:
            for lt in session.execute(select(LinkType)).scalars():
                assert lt.inverse, f"Link type '{lt.name}' has empty inverse"
                assert lt.inverse != lt.name, (
                    f"Link type '{lt.name}' inverse is same as name"
                )


class TestCorpusWritePolicy:
    """Corpus model includes write_policy."""

    def test_corpus_has_write_policy(self, store):
        """All corpora have a write_policy set."""
        from sqlalchemy import select

        with store._Session() as session:
            for c in session.execute(select(Corpus)).scalars():
                assert c.write_policy in ("ingestion", "direct", "system"), (
                    f"Corpus '{c.id}' has invalid write_policy: {c.write_policy}"
                )

    def test_papers_are_ingestion(self, store):
        from sqlalchemy import select

        with store._Session() as session:
            papers = session.get(Corpus, "papers")
            assert papers.write_policy == "ingestion"

    def test_notes_are_direct(self, store):
        from sqlalchemy import select

        with store._Session() as session:
            notes = session.get(Corpus, "notes")
            assert notes is not None, "notes corpus not seeded"
            assert notes.write_policy == "direct"

    def test_todos_are_direct(self, store):
        from sqlalchemy import select

        with store._Session() as session:
            todos = session.get(Corpus, "todos")
            assert todos is not None, "todos corpus not seeded"
            assert todos.write_policy == "direct"


class TestCreateLink:
    """store.create_link() validation and creation."""

    def test_create_link_between_papers(self, store, sample_bundle, second_bundle):
        store.ingest(sample_bundle)
        store.ingest(second_bundle)

        link = store.create_link("smith2024quantum", "jones2023surface", "cites")
        assert link.id is not None
        assert link.src_slug == "smith2024quantum"
        assert link.dst_slug == "jones2023surface"
        assert link.relation == "cites"
        assert link.src_node_id is None
        assert link.dst_node_id is None

    def test_create_link_with_block_anchors(self, store, sample_bundle, second_bundle):
        store.ingest(sample_bundle)
        store.ingest(second_bundle)

        link = store.create_link(
            "smith2024quantum", "jones2023surface", "discusses",
            src_node_id="doi:10.1038/s41567-024-1234-5-p00-000",
            dst_node_id="doi:10.1103/PhysRevLett.123.456-p00-000",
        )
        assert link.src_node_id == "doi:10.1038/s41567-024-1234-5-p00-000"
        assert link.dst_node_id == "doi:10.1103/PhysRevLett.123.456-p00-000"
        assert link.relation == "discusses"

    def test_create_link_invalid_relation_raises(self, store, sample_bundle, second_bundle):
        store.ingest(sample_bundle)
        store.ingest(second_bundle)

        with pytest.raises(ValueError, match="Unknown link relation 'bogus'"):
            store.create_link("smith2024quantum", "jones2023surface", "bogus")

    def test_create_link_missing_src_raises(self, store, sample_bundle):
        store.ingest(sample_bundle)

        with pytest.raises(ValueError, match="Source ref not found"):
            store.create_link("nonexistent", "smith2024quantum", "cites")

    def test_create_link_missing_dst_raises(self, store, sample_bundle):
        store.ingest(sample_bundle)

        with pytest.raises(ValueError, match="Target ref not found"):
            store.create_link("smith2024quantum", "nonexistent", "cites")

    def test_create_link_default_relation_is_cites(self, store, sample_bundle, second_bundle):
        store.ingest(sample_bundle)
        store.ingest(second_bundle)

        link = store.create_link("smith2024quantum", "jones2023surface")
        assert link.relation == "cites"

    def test_create_multiple_links_same_pair(self, store, sample_bundle, second_bundle):
        """Multiple links between the same refs are allowed (different relations)."""
        store.ingest(sample_bundle)
        store.ingest(second_bundle)

        l1 = store.create_link("smith2024quantum", "jones2023surface", "cites")
        l2 = store.create_link("smith2024quantum", "jones2023surface", "discusses")
        assert l1.id != l2.id


class TestGetLinks:
    """store.get_links() retrieval and bidirectional display."""

    @pytest.fixture(autouse=True)
    def _setup(self, store, sample_bundle, second_bundle):
        store.ingest(sample_bundle)
        store.ingest(second_bundle)
        store.create_link("smith2024quantum", "jones2023surface", "cites")
        store.create_link("jones2023surface", "smith2024quantum", "discusses")

    def test_get_links_both_directions(self, store):
        links = store.get_links("smith2024quantum")
        assert len(links) == 2
        # One outbound (cites), one inbound (discussed_in = inverse of discusses)
        directions = {l["direction"] for l in links}
        assert directions == {"outbound", "inbound"}

    def test_outbound_only(self, store):
        links = store.get_links("smith2024quantum", direction="outbound")
        assert len(links) == 1
        assert links[0]["relation"] == "cites"
        assert links[0]["display_relation"] == "cites"
        assert links[0]["direction"] == "outbound"

    def test_inbound_shows_inverse_label(self, store):
        links = store.get_links("smith2024quantum", direction="inbound")
        assert len(links) == 1
        assert links[0]["relation"] == "discusses"
        assert links[0]["display_relation"] == "discussed_in"
        assert links[0]["direction"] == "inbound"

    def test_filter_by_relation(self, store):
        links = store.get_links("smith2024quantum", relation="cites")
        assert len(links) == 1
        assert links[0]["relation"] == "cites"

    def test_invalid_direction_raises(self, store):
        with pytest.raises(ValueError, match="Invalid direction"):
            store.get_links("smith2024quantum", direction="sideways")

    def test_get_links_empty(self, store):
        """Unlinked ref returns empty list."""
        links = store.get_links("nonexistent_slug")
        assert links == []


class TestGetLinkCount:
    """store.get_link_count() summary."""

    def test_link_counts(self, store, sample_bundle, second_bundle):
        store.ingest(sample_bundle)
        store.ingest(second_bundle)
        store.create_link("smith2024quantum", "jones2023surface", "cites")
        store.create_link("smith2024quantum", "jones2023surface", "discusses")

        counts = store.get_link_count("smith2024quantum")
        assert counts["cites"] == 1
        assert counts["discusses"] == 1

    def test_link_counts_include_inbound(self, store, sample_bundle, second_bundle):
        store.ingest(sample_bundle)
        store.ingest(second_bundle)
        store.create_link("jones2023surface", "smith2024quantum", "cites")

        counts = store.get_link_count("smith2024quantum")
        assert counts.get("cited_by") == 1

    def test_empty_counts(self, store, sample_bundle):
        store.ingest(sample_bundle)
        counts = store.get_link_count("smith2024quantum")
        assert counts == {}


class TestDeleteLink:
    """store.delete_link() and store.delete_links_for_slug()."""

    def test_delete_link_by_id(self, store, sample_bundle, second_bundle):
        store.ingest(sample_bundle)
        store.ingest(second_bundle)
        link = store.create_link("smith2024quantum", "jones2023surface", "cites")

        assert store.delete_link(link.id) is True
        assert store.get_links("smith2024quantum") == []

    def test_delete_link_not_found(self, store):
        assert store.delete_link(99999) is False

    def test_delete_links_for_slug(self, store, sample_bundle, second_bundle):
        store.ingest(sample_bundle)
        store.ingest(second_bundle)
        store.create_link("smith2024quantum", "jones2023surface", "cites")
        store.create_link("jones2023surface", "smith2024quantum", "discusses")

        deleted = store.delete_links_for_slug("smith2024quantum")
        assert deleted == 2
        assert store.get_links("smith2024quantum") == []
        assert store.get_links("jones2023surface") == []

    def test_delete_links_for_slug_no_links(self, store, sample_bundle):
        store.ingest(sample_bundle)
        deleted = store.delete_links_for_slug("smith2024quantum")
        assert deleted == 0


class TestBlockLevelLinks:
    """Links with block-level node_id anchors."""

    def test_get_links_filtered_by_node_id(self, store, sample_bundle, second_bundle):
        store.ingest(sample_bundle)
        store.ingest(second_bundle)

        # Create doc-level and block-level links
        store.create_link("smith2024quantum", "jones2023surface", "cites")
        store.create_link(
            "smith2024quantum", "jones2023surface", "discusses",
            src_node_id="doi:10.1038/s41567-024-1234-5-p00-000",
        )

        # Doc-level: both links
        all_links = store.get_links("smith2024quantum")
        assert len(all_links) == 2

        # Block-level: only the block-anchored link
        block_links = store.get_links(
            "smith2024quantum",
            node_id="doi:10.1038/s41567-024-1234-5-p00-000",
        )
        assert len(block_links) == 1
        assert block_links[0]["relation"] == "discusses"
