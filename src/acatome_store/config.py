"""Store configuration.

Wraps the shared AcatomeConfig from acatome-meta, exposing only
store-relevant fields. Accepts overrides for testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
    embed_model: str = "all-MiniLM-L6-v2"
    embed_dim: int = 384
    embed_provider: str = "chroma"
    embed_index_dim: int | None = None

    @property
    def db_url(self) -> str:
        """SQLAlchemy connection string.

        Auto-generated from metadata_backend + settings, or use
        explicit ``_db_url`` override.
        """
        if self._db_url:
            return self._db_url

        if self.metadata_backend == "postgres":
            pw = f":{self.pg_password}" if self.pg_password else ""
            return (
                f"postgresql+psycopg://{self.pg_user}{pw}"
                f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
            )

        # Default: SQLite file in store_path
        db_file = self.store_path / "acatome.db"
        return f"sqlite:///{db_file}"

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
            embed_model=profile.model if profile else "all-MiniLM-L6-v2",
            embed_dim=profile.dim if profile else 384,
            embed_provider=profile.provider if profile else "chroma",
            embed_index_dim=profile.index_dim if profile else None,
        )
