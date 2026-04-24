"""acatome-store: Persistent storage for acatome bundles (Postgres + pgvector)."""

from importlib.metadata import version

from acatome_store.models import Ref
from acatome_store.store import Store
from acatome_store.vector import PgVectorIndex, VectorIndex, create_index

__all__ = ["PgVectorIndex", "Ref", "Store", "VectorIndex", "create_index"]
__version__ = version("acatome-store")
