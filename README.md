# acatome-store

Persistent storage, deduplication, metadata queries, and semantic search for scientific paper bundles.

## Features

- **SQLAlchemy ORM** — portable across SQLite, Postgres, MySQL
- **Refs + Papers split** — identity table (refs) separate from ingested content (papers)
- **Citation graph** — directed `citing → cited` edges, works for ingested + stub papers
- **Supplements** — ingest supplementary PDFs with scoped block retrieval
- **Retractions** — flag papers as retracted with notes
- **Vector search** — ChromaDB (default) or pgvector (zero text duplication)
- **CLI** — `acatome-store` command for ingest, query, retract, and stats

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
acatome-store ingest /path/to/bundle.acatome
acatome-store stats
acatome-store retract doi:10.1234/fake --note "Fabricated data"
```

## Testing

```bash
uv run python -m pytest tests/ -v
```

## License

LGPL-3.0-or-later — see [LICENSE](LICENSE).
