# Changelog

All notable changes to **acatome-store** will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] — unreleased

### Removed — BREAKING

- **Chroma vector backend** is gone.  `ChromaIndex` is no longer
  exported; `chromadb` and `llama-index-vector-stores-chroma` are
  no longer dependencies.  The factory always returns a
  `PgVectorIndex`.
- **SQLite metadata backend** is gone.  `Store.__init__` raises
  `RuntimeError` if `db_url` isn't a `postgresql+psycopg://…` URL.
  The old SQLite-specific migration branch (`PRAGMA foreign_keys`,
  `PRAGMA table_info`, `CREATE UNIQUE INDEX` workaround) has been
  deleted.
- **`StoreConfig(metadata_backend=..., vector_backend=..., graph_backend=...)`**
  constructor args are gone (`TypeError` at construction).  They
  remain exposed as read-only properties always returning
  `"postgres"` / `"pgvector"` / `"none"` so downstream UI callers
  (e.g. `acatome-chat stats`) keep working unchanged.

### Migration (0.9 → 1.0)

Existing SQLite+Chroma deployments need to move to Postgres with
pgvector before upgrading.  One-liner to move your content:

```
# 1. Stand up Postgres with pgvector
createdb acatome && psql -c 'CREATE EXTENSION vector' acatome

# 2. Point acatome at it (in ~/.acatome/config.toml)
[store]
pg_host = "localhost"
pg_user = "acatome"
pg_password = "…"

# 3. Re-ingest your bundles
acatome-store reingest
```

The acatome-meta `[store] vector_backend = "chroma"` /
`metadata_backend = "sqlite"` config keys are silently ignored by
acatome-store 1.0 — acatome-meta still reads them for compatibility
with older packages but the store only consults the `pg_*` fields.

### Added

- **Cross-corpus filter on `search_text`.**  The vector index's
  `search_text(where={...})` accepts `corpus_id` as a filter key,
  scalar (`corpus_id='memories'`) or list
  (`{'$in': ['papers', 'websites']}`).  Implemented as a JOIN to
  `refs.corpus_id` at query time.
- **`Store.search_text(corpora=[...])` convenience kwarg.**  Passes
  a list of corpus ids through as the corpus filter so cross-corpus
  callers (e.g. the new `search(type='all')` in precis-mcp) don't
  have to craft the where dict themselves.
- **`corpus_id` always emitted in hit metadata.**  `search_text`
  JOINs `blocks → refs` on every call and surfaces `corpus_id`,
  `slug`, `ref_title`, and `ref_id` (int) in each hit's metadata.
  Removes the N+1 `Store.get(pid)` lookup that callers used for
  enrichment.
- **`corpus_id` threaded through every ingest path.**  Paper bundle
  ingest passes `corpus_id='papers'` to `index.add_blocks`;
  direct-write corpora (todos, flashcards, memories, web, book,
  conversations) pass their own `corpus_id` automatically via
  `_index_direct_blocks`.  The kwarg is accepted by `PgVectorIndex`
  for API parity with legacy callers but ignored (corpus already
  lives on `refs`).

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
