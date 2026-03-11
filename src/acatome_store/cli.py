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


if __name__ == "__main__":
    app()
