"""Tests for StoreConfig, particularly the password-safe url helpers."""

from __future__ import annotations

from acatome_store.config import StoreConfig, _mask_url_password


class TestDbUrl:
    def test_sqlite_default(self, tmp_path):
        cfg = StoreConfig(store_path=tmp_path)
        assert cfg.db_url == f"sqlite:///{tmp_path / 'acatome.db'}"

    def test_postgres_no_password(self):
        cfg = StoreConfig(
            metadata_backend="postgres",
            pg_host="db.example.com",
            pg_port=5432,
            pg_database="acatome",
            pg_user="bob",
        )
        assert cfg.db_url == "postgresql+psycopg://bob@db.example.com:5432/acatome"

    def test_postgres_simple_password(self):
        cfg = StoreConfig(
            metadata_backend="postgres",
            pg_user="bob",
            pg_password="hunter2",
        )
        assert "bob:hunter2@" in cfg.db_url

    def test_postgres_password_with_special_chars_url_encoded(self):
        """Passwords with @/:/#/%/space must be percent-encoded in the URL."""
        cfg = StoreConfig(
            metadata_backend="postgres",
            pg_user="bob",
            pg_password="p@ss:w/ord#1%",
        )
        url = cfg.db_url
        # Password characters must not appear literally in the URL —
        # they'd confuse the parser.
        assert "p@ss:w/ord#1%" not in url
        # Encoded form is present.
        assert "p%40ss%3Aw%2Ford%23" in url

    def test_postgres_username_with_special_chars_url_encoded(self):
        cfg = StoreConfig(
            metadata_backend="postgres",
            pg_user="user@prod",
            pg_password="pw",
        )
        assert "user%40prod:" in cfg.db_url

    def test_explicit_override_used_verbatim(self):
        cfg = StoreConfig(_db_url="sqlite:///:memory:")
        assert cfg.db_url == "sqlite:///:memory:"


class TestMaskedDbUrl:
    def test_sqlite_unchanged(self, tmp_path):
        cfg = StoreConfig(store_path=tmp_path)
        assert cfg.masked_db_url == cfg.db_url  # no password to mask

    def test_postgres_password_masked(self):
        cfg = StoreConfig(
            metadata_backend="postgres",
            pg_user="bob",
            pg_password="hunter2",
        )
        masked = cfg.masked_db_url
        assert "hunter2" not in masked
        assert "***" in masked
        assert "bob:***@" in masked

    def test_postgres_no_password_no_mask_artifact(self):
        cfg = StoreConfig(
            metadata_backend="postgres",
            pg_user="bob",
        )
        assert "***" not in cfg.masked_db_url
        assert cfg.masked_db_url == "postgresql+psycopg://bob@localhost:5432/acatome"

    def test_explicit_override_has_password_masked(self):
        cfg = StoreConfig(_db_url="postgresql://alice:secret@host:5432/db")
        masked = cfg.masked_db_url
        assert "secret" not in masked
        assert "alice:***@host" in masked

    def test_explicit_override_without_password_passes_through(self):
        cfg = StoreConfig(_db_url="sqlite:///:memory:")
        assert cfg.masked_db_url == "sqlite:///:memory:"


class TestRepr:
    def test_repr_masks_password(self):
        cfg = StoreConfig(
            metadata_backend="postgres",
            pg_user="bob",
            pg_password="hunter2",
        )
        r = repr(cfg)
        assert "hunter2" not in r
        assert "***" in r

    def test_repr_includes_backend_hints(self, tmp_path):
        cfg = StoreConfig(store_path=tmp_path, vector_backend="chroma")
        r = repr(cfg)
        assert "metadata_backend='sqlite'" in r
        assert "vector_backend='chroma'" in r


class TestMaskUrlPasswordHelper:
    def test_handles_url_without_password(self):
        assert _mask_url_password("sqlite:///foo.db") == "sqlite:///foo.db"

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
