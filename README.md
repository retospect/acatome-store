# acatome-store

Persistent storage, deduplication, metadata queries, and semantic search for scientific paper bundles.

## Features

- **SQLAlchemy ORM** — portable across SQLite, Postgres, MySQL
- **Refs + Papers split** — identity table (refs) separate from ingested content (papers)
- **Citation graph** — directed `citing → cited` edges, works for ingested + stub papers
- **Supplements** — ingest supplementary PDFs with scoped block retrieval
- **Retractions** — flag papers as retracted with notes
- **Vector search** — ChromaDB (default) or pgvector (zero text duplication)
- **CLI** — `acatome-store` command for ingest, reingest, query, retract, and stats
- **Schema management** — `reset_schema()` and `reingest --drop` for clean rebuilds

## Installation

```bash
uv pip install -e .
```

With Postgres support:

```bash
uv pip install -e ".[postgres]"
```

## Usage

```python
from acatome_store import Store

store = Store()
ref_id = store.ingest(bundle_path)
paper = store.get(ref_id)
results = store.search_text("transformer attention", top_k=5)
# hits include paper info, block summaries, and text
```

## CLI

```bash
acatome-store ingest /path/to/bundle.acatome    # single bundle
acatome-store ingest /path/to/dir/               # directory of bundles
acatome-store reingest                            # re-ingest all from ~/.acatome/papers/
acatome-store reingest --drop                     # drop schema + re-ingest (confirm prompt)
acatome-store reingest --path /other/dir          # custom bundle directory
acatome-store stats
acatome-store search "CO2 capture"
acatome-store list
acatome-store info doi:10.1234/example
acatome-store retract doi:10.1234/fake --note "Fabricated data"
```

### Schema Reset

If the database schema drifts from the model (e.g. after upgrading acatome-store),
use `reingest --drop` to drop all tables, recreate from the current SQLAlchemy model,
and re-ingest all `.acatome` bundles. No data is lost since bundles are the source of truth.

```bash
acatome-store reingest --drop
# prompts for confirmation, then:
# 1. Drops all tables (refs, blocks, papers, links, etc.)
# 2. Recreates schema from current model
# 3. Re-ingests all bundles from ~/.acatome/papers/
```

Programmatically:

```python
store = Store()
store.reset_schema()  # drop + recreate tables
```

## Testing

```bash
uv run python -m pytest tests/ -v
```

## License

LGPL-3.0-or-later — see [LICENSE](LICENSE).
