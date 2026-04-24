# Changelog

All notable changes to **acatome-store** will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.9.0] — unreleased

### Added

- **Cross-corpus filter on `search_text`.**  The vector index's
  `search_text(where={...})` now accepts `corpus_id` as a filter
  key, scalar (`corpus_id='memories'`) or list
  (`{'$in': ['papers', 'websites']}`).  On pgvector the filter is
  a JOIN to `refs.corpus_id` at query time.  On Chroma the filter
  reads a `corpus_id` metadata field stamped at `add_blocks` time.
- **`Store.search_text(corpora=[...])` convenience kwarg.**  Passes
  a list of corpus ids through as the corpus filter so cross-corpus
  callers (e.g. the new `search(type='all')` in precis-mcp) don't
  have to craft the where dict themselves.
- **`corpus_id` always emitted in hit metadata.**  The pgvector
  `search_text` now JOINs `blocks → refs` on every call and
  surfaces `corpus_id`, `slug`, and `ref_title` in each hit's
  metadata.  Removes the N+1 `Store.get(pid)` lookup that existing
  callers used for enrichment.
- **`ChromaIndex.add_blocks(..., corpus_id=...)`.**  Stamps the
  corpus id onto each block's metadata dict so cross-corpus filters
  work on the Chroma backend too.  `PgVectorIndex.add_blocks`
  accepts the kwarg for API parity but ignores it (the corpus lives
  on `refs` already).
- **`corpus_id` threaded through every ingest path.**  Paper bundle
  ingest now passes `corpus_id='papers'` to `index.add_blocks`;
  direct-write corpora (todos, flashcards, memories, web, book,
  conversations) pass their own `corpus_id` automatically via
  `_index_direct_blocks`.

### Fixed

- **Stale `Ref` metadata on re-ingest.**  When a bundle's `pdf_hash`
  matched an existing `Paper`, `ingest()` previously returned the
  existing `ref_id` without touching the `Ref` row — so garbage
  metadata from an earlier unverified ingest (e.g. an InDesign
  filename like `"nmat1849 Geim Progress Article.indd"` as the
  title) was preserved forever.  The same fill-blanks-only policy
  also meant `_upsert_ref` never repaired stale non-null fields.
  A new `_should_upgrade_ref` / `_refresh_ref_metadata` pair now
  refreshes `doi`, `s2_id`, `arxiv_id`, `title`, `authors`, `year`,
  `journal`, `entry_type`, and `source` when (a) no `Paper` exists
  yet, (b) the existing `Paper` is unverified and the new bundle
  is verified, or (c) the current title is garbage and the new
  title is clean.  Slug is also upgraded on dedup, with a
  collision check.  Safety guards: verified refs cannot be
  clobbered by later unverified bundles; a clean title is never
  downgraded to garbage even when the new bundle is verified;
  user-curated `keywords` and `tags` are preserved.

### Changed

- **`Store.search_text` return shape.**  Every hit's `metadata`
  dict now includes `corpus_id`, `ref_id` (int, alongside the
  legacy `paper_id` string), `slug`, and `ref_title`.  Existing
  keys (`paper_id`, `node_id`, `block_index`, `page`,
  `block_type`, `profile`, `section_path`) are unchanged — this
  is additive.

### Migration

No schema change required.  Historical blocks with NULL embeddings
are still handled by `acatome-store backfill-embeddings`
(Postgres-only; works per-corpus via `--corpus=<id>`).  The new
`websites` and `books` corpora are auto-seeded at first startup.

## [0.7.5] — 2026-04-10

### Fixed

- SQLite auto-migration: `_ensure_missing_columns()` now runs on all backends,
  not just Postgres. Users upgrading from pre-0.6.0 SQLite databases no longer
  get `no such column: refs.corpus_id` errors.
- SQLite DDL compatibility: split `REFERENCES` + `DEFAULT` (forbidden together)
  and `UNIQUE` (forbidden in ALTER TABLE ADD COLUMN) into separate statements.

## [0.7.0] — 2026-04-02

### Added

- `reingest` CLI command: re-ingest all bundles from `~/.acatome/papers/`
  - `--drop` flag: drop schema and recreate before ingesting (with confirmation)
  - `--path` / `-p`: override bundle directory
- `Store.reset_schema()`: drop all tables and recreate from current model

### Fixed

- Schema drift: old `entry_type`, `journal`, `source`, `retracted`, `retraction_note`
  columns removed from Ref model in v0.6.0 but not from Postgres — caused
  `NotNullViolation` on every INSERT. Fix: drop stale columns via `reset_schema()`
  or `reingest --drop`.

## [0.6.0] — 2026-04-01

### Added

- Corpus model with write_policy (ingestion/direct/system) and seed corpora
- LinkType registry with 12 seed relations (cites/cited_by, annotates, etc.)
- Link model for slug-based edges between refs or blocks
- `create_ref()` for direct ref creation without .acatome bundle
- `update_ref_metadata()` and `update_block_text()` methods
- Link CRUD: `create_link`, `get_links`, `get_link_count`, `delete_link`

### Changed

- Slug moved from Paper to Ref (all refs can have slugs)
- Metadata JSON column on Ref replaces journal/entry_type/source/retracted columns
- Ref is polymorphic on corpus_id
- Slug collisions auto-disambiguated with a/b/c suffixes
- `~` stripped from slugs during ingest (reserved URI separator)

### Deprecated

- Citation model (use links with relation='cites')
- Note model (use refs in 'notes' corpus with 'annotates' links)

## [0.1.0] — 2026-03-11

### Added

- Initial release.
