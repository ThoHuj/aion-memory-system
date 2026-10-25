"""What a host sees when it imports the package."""

import aion_memory


def test_the_package_exports_only_the_public_api() -> None:
    """Hosts get the factory, MemorySystem, its result types and the setup SQL, nothing else."""
    assert sorted(aion_memory.__all__) == [
        "FactData",
        "MemoryAddResult",
        "MemorySystem",
        "RetrievalResult",
        "create_memory_system",
        "schema_sql",
    ]


def test_the_stores_are_not_exported() -> None:
    """The storage interface and the Postgres store are internal."""
    assert not hasattr(aion_memory, "FactsStore")
    assert not hasattr(aion_memory, "PostgresFactsStore")
