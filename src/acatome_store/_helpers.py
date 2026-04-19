"""Free-function helpers used by :mod:`acatome_store.store`.

These are pure functions that do not touch the :class:`Store` state —
they are separated out so ``store.py`` stays focused on the ``Store``
class itself.  All are re-exported from ``acatome_store.store`` for
backward compatibility.

Contents
--------
Safe SQL identifiers
    :data:`SAFE_IDENT_RE`, :func:`assert_safe_ident` — defense-in-depth
    validator for the handful of DDL statements where bound parameters
    are not supported.
Embedders
    :func:`get_embedder` — thin wrapper around
    :func:`acatome_meta.literature.build_embedder` with the right
    parameter names for :class:`StoreConfig`.
    :func:`reembed_blocks` — recompute embeddings for a block list.
Bundle IO
    :func:`read_bundle` — read a gzipped ``.acatome`` bundle.
    :func:`update_bundle_embeddings` — write embeddings back into the
    bundle so future re-ingests skip the re-embed step.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from acatome_meta.literature import SKIP_EMBED_TYPES, build_embedder

from acatome_store.config import StoreConfig

log = logging.getLogger(__name__)

# ─── SQL-identifier validation ────────────────────────────────────

#: Strict validator for SQL identifiers that must be string-interpolated.
#:
#: SQLite's ``PRAGMA`` and most backends' DDL do not accept bound
#: parameters for identifiers, so some call sites must interpolate.
#: This pattern whitelists ``[A-Za-z_][A-Za-z0-9_]{0,62}`` — 1–63 ASCII
#: characters, which covers the PostgreSQL identifier length limit and
#: rules out every metacharacter an attacker would need.
SAFE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def assert_safe_ident(name: str, *, kind: str = "identifier") -> None:
    """Raise :class:`ValueError` if ``name`` is not a safe SQL identifier.

    Defends against metacharacter injection into DDL statements that
    cannot use bound parameters.
    """
    if not isinstance(name, str) or not SAFE_IDENT_RE.match(name):
        raise ValueError(
            f"Unsafe SQL {kind} {name!r}: "
            "must match [A-Za-z_][A-Za-z0-9_]{0,62}"
        )


# ─── Embedders ────────────────────────────────────────────────────


def get_embedder(
    config: StoreConfig,
) -> Callable[[list[str]], list[list[float]]]:
    """Build an embedding function from the store's configured profile.

    Raises:
        EmbedderUnavailableError: the configured backend is not installed
            (error message includes the ``pip install`` incantation).
        ValueError: ``config.embed_provider`` is not recognised.
    """
    return build_embedder(
        provider=config.embed_provider,
        model=config.embed_model,
        dim=config.embed_dim,
        index_dim=config.embed_index_dim,
    )


def reembed_blocks(
    blocks: list[dict[str, Any]],
    embedder: Callable[[list[str]], list[list[float]]],
    profile: str = "default",
) -> list[dict[str, Any]]:
    """Recompute embeddings for ``blocks``, replacing existing ones.

    Skips block types that are not meant to be embedded (section
    headers, equations, junk) and blocks with empty text.
    """
    texts = []
    indices = []
    for i, b in enumerate(blocks):
        if b.get("type") in SKIP_EMBED_TYPES:
            continue
        text = b.get("text", "").strip()
        if not text:
            continue
        texts.append(text)
        indices.append(i)

    if not texts:
        return blocks

    embeddings = embedder(texts)
    for idx, emb in zip(indices, embeddings):
        if "embeddings" not in blocks[idx]:
            blocks[idx]["embeddings"] = {}
        blocks[idx]["embeddings"][profile] = emb

    return blocks


# ─── Bundle IO ────────────────────────────────────────────────────


def read_bundle(path: str | Path) -> dict[str, Any]:
    """Read a ``.acatome`` bundle (gzipped JSON)."""
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def update_bundle_embeddings(
    bundle_path: Path,
    data: dict[str, Any],
    blocks: list[dict[str, Any]],
    embed_model: str,
    embed_dim: int,
) -> None:
    """Persist re-computed embeddings + metadata back into the bundle.

    Called after :func:`reembed_blocks` during ingest when the bundle's
    embedding model doesn't match the store's configured model.  Writing
    the new vectors back means the next re-ingest of the same bundle
    will not need to re-embed again.

    ``blocks`` is expected to be the same list :func:`reembed_blocks`
    returned (in-place updated); it replaces ``data["blocks"]`` verbatim.

    On read-only filesystems or permission-denied we log a WARNING and
    continue — the DB already has the blocks; the cost is repeating
    the embedding work on the next ingest.
    """
    data["blocks"] = blocks
    em = data.get("enrichment_meta") or {}
    models = em.get("embedding_models") or {}
    models["default"] = {"model": embed_model, "dim": embed_dim}
    em["embedding_models"] = models
    em.setdefault("profiles", [])
    if "default" not in em["profiles"]:
        em["profiles"].append("default")
    data["enrichment_meta"] = em
    try:
        with gzip.open(bundle_path, "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    except OSError as exc:
        # Bundle on a read-only filesystem or permission-denied is non-fatal
        # for ingest (DB already has the blocks), but the next re-ingest will
        # waste work re-embedding again — surface the path so the user can
        # fix permissions.
        log.warning(
            "Failed to write embeddings back to %s: %s. "
            "Check file permissions; next re-ingest will repeat the embedding work.",
            bundle_path, exc,
        )
