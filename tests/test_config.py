"""Tests for StoreConfig, particularly the password-safe url helpers.

v1.0.0+ is Postgres-only.  The old SQLite default-url tests and the
``metadata_backend`` / ``vector_backend`` construction args are gone
— both are now read-only properties pinned to ``'postgres'`` /
``'pgvector'``.
"""

from __future__ import annotations

from acatome_store.config import StoreConfig, _mask_url_password


class TestDbUrl:
    def test_default_no_password(self):
        """Default config builds a local Postgres DSN with no password."""
        cfg = StoreConfig()
        assert cfg.db_url == "postgresql+psycopg://acatome@localhost:5432/acatome"

    def test_postgres_with_password(self):
        cfg = StoreConfig(
            pg_host="db.example.com",
            pg_port=5433,
            pg_database="acatome",
            pg_user="bob",
            pg_password="hunter2",
        )
        assert cfg.db_url == "postgresql+psycopg://bob:hunter2@db.example.com:5433/acatome"

    def test_password_with_special_chars_url_encoded(self):
        """Passwords with @/:/#/%/space must be percent-encoded in the URL."""
        cfg = StoreConfig(
            pg_user="bob",
            pg_password="p@ss:w/ord#1%",
        )
        url = cfg.db_url
        # Literal special chars must not appear — they'd confuse the parser.
        assert "p@ss:w/ord#1%" not in url
        # Encoded form is present.
        assert "p%40ss%3Aw%2Ford%23" in url

    def test_username_with_special_chars_url_encoded(self):
        cfg = StoreConfig(
            pg_user="user@prod",
            pg_password="pw",
        )
        assert "user%40prod:" in cfg.db_url

    def test_explicit_override_used_verbatim(self):
        cfg = StoreConfig(_db_url="postgresql+psycopg://custom:pw@host:5432/db")
        assert cfg.db_url == "postgresql+psycopg://custom:pw@host:5432/db"


class TestMaskedDbUrl:
    def test_postgres_password_masked(self):
        cfg = StoreConfig(pg_user="bob", pg_password="hunter2")
        masked = cfg.masked_db_url
        assert "hunter2" not in masked
        assert "***" in masked
        assert "bob:***@" in masked

    def test_postgres_no_password_no_mask_artifact(self):
        cfg = StoreConfig(pg_user="bob")
        assert "***" not in cfg.masked_db_url
        assert cfg.masked_db_url == "postgresql+psycopg://bob@localhost:5432/acatome"

    def test_explicit_override_has_password_masked(self):
        cfg = StoreConfig(_db_url="postgresql://alice:secret@host:5432/db")
        masked = cfg.masked_db_url
        assert "secret" not in masked
        assert "alice:***@host" in masked

    def test_explicit_override_without_password_passes_through(self):
        cfg = StoreConfig(_db_url="postgresql://alice@host:5432/db")
        assert cfg.masked_db_url == "postgresql://alice@host:5432/db"


class TestRepr:
    def test_repr_masks_password(self):
        cfg = StoreConfig(pg_user="bob", pg_password="hunter2")
        r = repr(cfg)
        assert "hunter2" not in r
        assert "***" in r

    def test_repr_shows_postgres_backend_labels(self):
        """The read-only backend properties are always 'postgres' /
        'pgvector' since v1.0.0 — __repr__ surfaces them so a stray
        ``print(cfg)`` still carries that context."""
        cfg = StoreConfig()
        r = repr(cfg)
        assert "metadata_backend='postgres'" in r
        assert "vector_backend='pgvector'" in r


class TestBackendProperties:
    """The backend fields became read-only properties in v1.0.0.  They
    always return the Postgres+pgvector values — consumers that used to
    branch on these strings (acatome-chat's stats display) keep
    working without a synchronised rev bump."""

    def test_metadata_backend_is_postgres(self):
        assert StoreConfig().metadata_backend == "postgres"

    def test_vector_backend_is_pgvector(self):
        assert StoreConfig().vector_backend == "pgvector"

    def test_graph_backend_is_none(self):
        assert StoreConfig().graph_backend == "none"

    def test_backend_fields_cannot_be_set_via_constructor(self):
        """Attempting to pass ``metadata_backend=``/``vector_backend=``
        to the constructor is a TypeError — the fields are gone.  This
        is a breaking change from pre-1.0 callers and is called out
        in the CHANGELOG migration note."""
        import pytest

        with pytest.raises(TypeError):
            StoreConfig(metadata_backend="sqlite")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            StoreConfig(vector_backend="chroma")  # type: ignore[call-arg]


class TestMaskUrlPasswordHelper:
    def test_handles_url_without_password(self):
        assert (
            _mask_url_password("postgresql://user@host/db")
            == "postgresql://user@host/db"
        )

    def test_masks_password(self):
        out = _mask_url_password("postgresql://alice:secret@host:5432/db")
        assert "secret" not in out
        assert "alice:***@host:5432" in out

    def test_handles_username_only(self):
        out = _mask_url_password("postgresql://alice@host:5432/db")
        assert out == "postgresql://alice@host:5432/db"

    def test_handles_password_only(self):
        """Edge case: no username but password present (rare but valid)."""
        out = _mask_url_password("postgresql://:secret@host:5432/db")
        assert "secret" not in out
        assert ":***@host" in out
