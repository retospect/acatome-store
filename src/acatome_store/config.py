"""Store configuration.

Wraps the shared AcatomeConfig from acatome-meta, exposing only
store-relevant fields. Accepts overrides for testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus


@dataclass
class StoreConfig:
    """Configuration for the acatome store."""

    store_path: Path = field(default_factory=lambda: Path.home() / ".acatome" / "store")
    vector_backend: str = "chroma"
    graph_backend: str = "none"
    metadata_backend: str = "sqlite"

    # Explicit db_url overrides auto-generated one
    _db_url: str = ""

    # Postgres settings (used when metadata_backend = "postgres")
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

        Auto-generated from metadata_backend + settings, or use
        explicit ``_db_url`` override.  The password is percent-encoded
        so passwords containing ``@``, ``:``, ``/``, ``#``, ``?`` or ``%``
        still produce a valid URL.

        **Security**: never log ``db_url`` directly.  Use
        :attr:`masked_db_url` for log output and error messages.
        """
        if self._db_url:
            return self._db_url

        if self.metadata_backend == "postgres":
            pw = f":{quote_plus(self.pg_password)}" if self.pg_password else ""
            user = quote_plus(self.pg_user)
            return (
                f"postgresql+psycopg://{user}{pw}"
                f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
            )

        # Default: SQLite file in store_path
        db_file = self.store_path / "acatome.db"
        return f"sqlite:///{db_file}"

    @property
    def masked_db_url(self) -> str:
        """Log-safe version of :attr:`db_url` with the password replaced by ``***``.

        Use this in ``log.info(...)``, ``print(...)``, and any error
        message that might reach logs or LLM-facing output.
        """
        if self._db_url:
            # Explicit override — try to mask a password if one is present.
            return _mask_url_password(self._db_url)

        if self.metadata_backend == "postgres":
            pw = ":***" if self.pg_password else ""
            user = quote_plus(self.pg_user)
            return (
                f"postgresql+psycopg://{user}{pw}"
                f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
            )

        db_file = self.store_path / "acatome.db"
        return f"sqlite:///{db_file}"

    def __repr__(self) -> str:  # pragma: no cover — trivial
        """Masked repr so a stray ``print(cfg)`` cannot leak the password."""
        cls = type(self).__name__
        return (
            f"{cls}(metadata_backend={self.metadata_backend!r}, "
            f"vector_backend={self.vector_backend!r}, "
            f"store_path={str(self.store_path)!r}, "
            f"db_url={self.masked_db_url!r})"
        )

    @classmethod
    def from_global(cls) -> StoreConfig:
        """Load from the shared acatome config (~/.acatome/config.toml + env)."""
        from acatome_meta.config import load_config

        cfg = load_config()
        profile = cfg.extract.profiles.get("default")
        return cls(
            store_path=cfg.store_path,
            vector_backend=cfg.store.vector_backend,
            graph_backend=cfg.store.graph_backend,
            metadata_backend=cfg.store.metadata_backend,
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
