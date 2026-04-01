# Changelog

All notable changes to **acatome-store** will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

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
