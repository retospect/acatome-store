"""Store configuration.

Postgres-only as of v1.0.0.  The previous SQLite+Chroma fallback was
removed to cut the backend matrix from 2×2 to 1×1: a single real
database (``metadata_backend = "postgres"``, ``vector_backend =
"pgvector"``) for both relational CRUD and ANN search.

See ``CHANGELOG.md`` and the 0.9→1.0 migration note for the rationale
and the one-liner to migrate an existing SQLite deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus


@dataclass
class StoreConfig:
    """Configuration for the acatome store — Postgres + pgvector only.

    The ``metadata_backend`` / ``vector_backend`` knobs from earlier
    versions are gone; the store always uses Postgres for refs/blocks
    and pgvector for embeddings.  This field layout retains the pg_*
    connection fields plus the embedding profile so ``from_global``
    can still hydrate a config from ``~/.acatome/config.toml``.
    """

    store_path: Path = field(default_factory=lambda: Path.home() / ".acatome" / "store")

    # Explicit db_url overrides the pg_* auto-build.  Useful for tests
    # and for deployments that already manage a secret URL elsewhere.
    _db_url: str = ""

    # Postgres connection settings.  All required at runtime, with
    # sensible defaults that match the dev-cluster setup.
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "acatome"
    pg_schema: str = "acatome"
    pg_user: str = "acatome"
    pg_password: str = ""

    # Embedding profile ("default" profile from extract config)
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    embed_provider: str = "sentence-transformers"
    embed_index_dim: int | None = None

    @property
    def db_url(self) -> str:
        """SQLAlchemy connection string (contains plaintext password).

        Built from ``pg_*`` fields, or returns ``_db_url`` when set.
        The password is percent-encoded so passwords containing ``@``,
        ``:``, ``/``, ``#``, ``?`` or ``%`` still produce a valid URL.

        **Security**: never log ``db_url`` directly.  Use
        :attr:`masked_db_url` for log output and error messages.
        """
        if self._db_url:
            return self._db_url

        pw = f":{quote_plus(self.pg_password)}" if self.pg_password else ""
        user = quote_plus(self.pg_user)
        return (
            f"postgresql+psycopg://{user}{pw}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @property
    def masked_db_url(self) -> str:
        """Log-safe version of :attr:`db_url` with the password replaced by ``***``.

        Use this in ``log.info(...)``, ``print(...)``, and any error
        message that might reach logs or LLM-facing output.
        """
        if self._db_url:
            # Explicit override — try to mask a password if one is present.
            return _mask_url_password(self._db_url)

        pw = ":***" if self.pg_password else ""
        user = quote_plus(self.pg_user)
        return (
            f"postgresql+psycopg://{user}{pw}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    # ------------------------------------------------------------------
    # Back-compat shims for consumers that still read these attributes
    # ------------------------------------------------------------------
    #
    # Downstream packages (acatome-chat, tools that render ``stats()``)
    # read ``vector_backend`` and ``metadata_backend`` to label the
    # runtime in UI strings.  Since v1.0.0 both values are always
    # ``"postgres"`` / ``"pgvector"`` — expose them as read-only
    # properties so those callers keep working without a synchronised
    # rev bump.  Writes at construction are silently ignored.

    @property
    def metadata_backend(self) -> str:
        """Always ``'postgres'`` since v1.0.0."""
        return "postgres"

    @property
    def vector_backend(self) -> str:
        """Always ``'pgvector'`` since v1.0.0."""
        return "pgvector"

    @property
    def graph_backend(self) -> str:
        """Always ``'none'`` since v1.0.0 (graph backend is a future
        extension; not implemented in the core store)."""
        return "none"

    def __repr__(self) -> str:  # pragma: no cover — trivial
        """Masked repr so a stray ``print(cfg)`` cannot leak the password."""
        cls = type(self).__name__
        return (
            f"{cls}(metadata_backend='postgres', vector_backend='pgvector', "
            f"store_path={str(self.store_path)!r}, "
            f"db_url={self.masked_db_url!r})"
        )

    @classmethod
    def from_global(cls) -> StoreConfig:
        """Load from the shared acatome config (~/.acatome/config.toml + env).

        Reads only the Postgres connection fields and the embedding
        profile from acatome-meta.  The old ``vector_backend`` /
        ``metadata_backend`` fields in the shared config are ignored —
        the store is pinned to Postgres + pgvector.
        """
        from acatome_meta.config import load_config

        cfg = load_config()
        profile = cfg.extract.profiles.get("default")
        return cls(
            store_path=cfg.store_path,
            pg_host=cfg.store.pg_host,
            pg_port=cfg.store.pg_port,
            pg_database=cfg.store.pg_database,
            pg_schema=cfg.store.pg_schema,
            pg_user=cfg.store.pg_user,
            pg_password=cfg.store.pg_password,
            embed_model=profile.model if profile else "BAAI/bge-m3",
            embed_dim=profile.dim if profile else 1024,
            embed_provider=profile.provider if profile else "sentence-transformers",
            embed_index_dim=profile.index_dim if profile else None,
        )


def _mask_url_password(url: str) -> str:
    """Return ``url`` with any ``user:password@`` segment reduced to ``user:***@``."""
    from urllib.parse import urlparse, urlunparse

    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    if not parsed.password:
        return url
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    user = parsed.username or ""
    netloc = f"{user}:***@{host}" if user else f":***@{host}"
    return urlunparse(parsed._replace(netloc=netloc))
