"""CLI for acatome-store."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(name="acatome-store", help="Manage the acatome paper store.")


@app.command()
def ingest(
    path: Path = typer.Argument(..., help=".acatome bundle or directory of bundles"),
):
    """Ingest bundle(s) into the store."""
    from acatome_store.store import Store

    store = Store()
    if path.is_file():
        paper_id = store.ingest(path)
        typer.echo(f"✓ {path.name} → ref {paper_id}")
    elif path.is_dir():
        bundles = sorted(path.rglob("*.acatome"))
        if not bundles:
            typer.echo(f"No .acatome bundles found in {path}")
            raise typer.Exit(1)
        typer.echo(f"Found {len(bundles)} bundles in {path}")
        succeeded, failed = 0, 0
        for i, b in enumerate(bundles, 1):
            try:
                paper_id = store.ingest(b)
                succeeded += 1
                typer.echo(f"  [{i}/{len(bundles)}] ✓ {b.name} → ref {paper_id}")
            except Exception as e:
                failed += 1
                typer.echo(f"  [{i}/{len(bundles)}] ✗ {b.name}: {e}")
        typer.echo(f"\nDone: {succeeded} ingested, {failed} failed")
    else:
        typer.echo(f"Error: {path} not found", err=True)
        raise typer.Exit(1)
    store.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    author: str | None = typer.Option(None, "--author"),
    year: int | None = typer.Option(None, "--year"),
):
    """Semantic search over stored papers."""
    from acatome_store.store import Store

    store = Store()
    hits = store.search_text(query, top_k=top_k)
    if not hits:
        typer.echo("No results.")
    for i, hit in enumerate(hits, 1):
        paper = hit.get("paper", {})
        slug = paper.get("slug", "?")
        title = paper.get("title", "?")[:50]
        dist = hit.get("distance", 0)
        text = hit.get("text", "")[:80]
        typer.echo(f"  {i}. [{slug}] {title}  (dist={dist:.3f})")
        typer.echo(f"     {text}")
    store.close()


@app.command(name="list")
def list_papers(
    limit: int = typer.Option(20, "--limit", "-n"),
):
    """List papers in the store."""
    from acatome_store.store import Store

    store = Store()
    papers = store.list_papers(limit=limit)
    for p in papers:
        year = p.get("year") or "????"
        typer.echo(f"  {p['slug']:30s}  {year}  {p['title'][:60]}")
    store.close()


@app.command()
def info(
    identifier: str = typer.Argument(..., help="paper_id, slug, or DOI"),
):
    """Show metadata for a paper."""
    from acatome_store.store import Store

    store = Store()
    paper = store.get(identifier)
    if not paper:
        typer.echo(f"Not found: {identifier}", err=True)
        raise typer.Exit(1)
    for k, v in paper.items():
        typer.echo(f"  {k}: {v}")
    store.close()


@app.command()
def reingest(
    path: Path = typer.Option(
        None,
        "--path",
        "-p",
        help="Bundle directory (default: ~/.acatome/papers)",
    ),
    drop: bool = typer.Option(False, "--drop", help="Drop and recreate schema first"),
):
    """Re-ingest all .acatome bundles from the papers directory."""
    import time

    from acatome_store.store import Store

    store = Store()

    if path is None:
        path = store._config.store_path.parent / "papers"

    if not path.is_dir():
        typer.echo(f"Error: {path} is not a directory", err=True)
        raise typer.Exit(1)

    bundles = sorted(path.rglob("*.acatome"))
    if not bundles:
        typer.echo(f"No .acatome bundles found in {path}")
        raise typer.Exit(1)

    if drop:
        typer.confirm(
            f"This will DROP all tables and re-ingest {len(bundles)} bundles. Continue?",
            abort=True,
        )
        store.reset_schema()
        typer.echo("Schema reset.")

    typer.echo(f"Ingesting {len(bundles)} bundles from {path}")
    t0 = time.time()
    succeeded, failed = 0, 0
    for i, b in enumerate(bundles, 1):
        try:
            store.ingest(b)
            succeeded += 1
        except Exception as e:
            failed += 1
            typer.echo(f"  ✗ {b.name}: {e}")
        if i % 100 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(bundles) - i) / rate
            typer.echo(
                f"  [{i}/{len(bundles)}] {succeeded} ok, {failed} fail — {rate:.0f}/s, ETA {eta:.0f}s"
            )

    elapsed = time.time() - t0
    typer.echo(f"\nDone: {succeeded} ingested, {failed} failed, {elapsed:.1f}s")
    store.close()


@app.command()
def stats():
    """Show store statistics."""
    from acatome_store.store import Store

    store = Store()
    s = store.stats()
    for k, v in s.items():
        typer.echo(f"  {k}: {v}")
    store.close()


@app.command()
def delete(
    identifier: str = typer.Argument(..., help="paper_id, slug, or DOI"),
):
    """Delete a paper from the store."""
    from acatome_store.store import Store

    store = Store()
    if store.delete(identifier):
        typer.echo(f"✓ Deleted {identifier}")
    else:
        typer.echo(f"Not found: {identifier}", err=True)
    store.close()


@app.command()
def retract(
    identifier: str = typer.Argument(..., help="paper_id, slug, or DOI"),
    note: str = typer.Option("", "--note", "-n", help="Retraction note"),
):
    """Mark a paper as retracted."""
    from acatome_store.store import Store

    store = Store()
    if store.retract(identifier, note=note):
        typer.echo(f"✓ {identifier} marked as retracted")
    else:
        typer.echo(f"Not found: {identifier}", err=True)
    store.close()


@app.command()
def unretract(
    identifier: str = typer.Argument(..., help="paper_id, slug, or DOI"),
):
    """Remove retraction flag from a paper."""
    from acatome_store.store import Store

    store = Store()
    if store.unretract(identifier):
        typer.echo(f"✓ {identifier} retraction removed")
    else:
        typer.echo(f"Not found: {identifier}", err=True)
    store.close()


@app.command()
def catalog(
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file (default: stdout). Use .tsv or .csv extension.",
    ),
    fmt: str = typer.Option(
        "tsv", "--format", "-f", help="Output format: tsv (default) or csv"
    ),
):
    """Export paper metadata catalog (slug, title, authors, year, DOI, blocks)."""
    import csv
    import io
    import json as _json

    from acatome_store.store import Store

    store = Store()
    papers = store.list_papers(limit=100000)
    store.close()

    if not papers:
        typer.echo("No papers in store.", err=True)
        raise typer.Exit(1)

    delimiter = "\t" if fmt == "tsv" else ","
    fields = ["slug", "year", "first_author", "title", "doi", "block_count"]

    if output is None:
        buf = io.StringIO()
    else:
        buf = open(output, "w", newline="")  # noqa: SIM115
    try:
        writer = csv.DictWriter(buf, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        for p in papers:
            # Extract first author surname
            first_author = ""
            raw = p.get("authors", "")
            if raw:
                try:
                    authors = _json.loads(raw) if isinstance(raw, str) else raw
                    if authors and isinstance(authors, list):
                        name = (
                            authors[0].get("name", "")
                            if isinstance(authors[0], dict)
                            else str(authors[0])
                        )
                        first_author = (
                            name.split(",")[0].strip()
                            if "," in name
                            else name.split()[-1]
                            if name.split()
                            else ""
                        )
                except (ValueError, TypeError, IndexError):
                    pass
            writer.writerow(
                {
                    "slug": p.get("slug", ""),
                    "year": p.get("year", ""),
                    "first_author": first_author,
                    "title": p.get("title", ""),
                    "doi": p.get("doi", ""),
                    "block_count": p.get("block_count", 0),
                }
            )
        if output is None:
            typer.echo(buf.getvalue(), nl=False)
        else:
            typer.echo(f"✓ {len(papers)} papers → {output}")
    finally:
        buf.close()


if __name__ == "__main__":
    app()
