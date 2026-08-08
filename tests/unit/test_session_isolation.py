"""Session isolation — the hardest invariant in the system.

No user's documents may ever appear in another user's retrieval results, and no
code path may search without a session namespace.
"""

from __future__ import annotations

import pytest

from omnirag.core.enums import FileType
from omnirag.core.exceptions import SessionIsolationError
from omnirag.core.models import Chunk, DocumentSummary
from omnirag.rag.vector_store import InMemoryVectorStore, SearchFilter
from omnirag.storage.files import MemoryFileStore
from omnirag.storage.sessions import (
    DocumentRegistry,
    is_valid_session_id,
    new_session_id,
    require_session_id,
)


def chunk(session_id, *, document_id="d1", text="Revenue reached 8.4 million.", page=1):
    return Chunk(
        document_id=document_id,
        session_id=session_id,
        filename="report.pdf",
        file_type=FileType.PDF,
        page_number=page,
        page_label=f"Page {page}",
        block_ids=[f"b{page}"],
        text=text,
    )


class TestSessionIds:
    def test_ids_are_unique_and_unguessable(self):
        ids = {new_session_id() for _ in range(200)}
        assert len(ids) == 200
        assert all(len(i) > 20 for i in ids)

    @pytest.mark.parametrize("value", [None, "", "short", 123, []])
    def test_invalid_ids_are_rejected(self, value):
        assert is_valid_session_id(value) is False
        with pytest.raises(SessionIsolationError):
            require_session_id(value)


class TestVectorStoreIsolation:
    def test_search_only_returns_the_caller_session(self):
        store = InMemoryVectorStore()
        alice, bob = new_session_id(), new_session_id()

        store.upsert(alice, [chunk(alice, text="Alice confidential revenue.")], [[1.0, 0.0]])
        store.upsert(bob, [chunk(bob, text="Bob confidential revenue.")], [[1.0, 0.0]])

        results = store.search(alice, [1.0, 0.0], top_k=10)

        assert len(results) == 1
        assert results[0].chunk.session_id == alice
        assert "Bob" not in results[0].chunk.text

    def test_identical_vectors_still_do_not_leak(self):
        store = InMemoryVectorStore()
        alice, bob = new_session_id(), new_session_id()
        vector = [0.5, 0.5, 0.5]

        for _ in range(5):
            store.upsert(bob, [chunk(bob, document_id="bob-doc")], [vector])
        store.upsert(alice, [chunk(alice, document_id="alice-doc")], [vector])

        results = store.search(alice, vector, top_k=50)
        assert {r.chunk.session_id for r in results} == {alice}

    def test_listing_chunks_is_session_scoped(self):
        store = InMemoryVectorStore()
        alice, bob = new_session_id(), new_session_id()
        store.upsert(alice, [chunk(alice)], [[1.0]])
        store.upsert(bob, [chunk(bob)], [[1.0]])

        assert len(store.list_chunks(alice)) == 1
        assert len(store.list_chunks(bob)) == 1

    def test_search_without_a_session_is_blocked(self):
        store = InMemoryVectorStore()
        with pytest.raises(SessionIsolationError):
            store.search("", [1.0], top_k=5)
        with pytest.raises(SessionIsolationError):
            store.search(None, [1.0], top_k=5)

    def test_upsert_without_a_session_is_blocked(self):
        store = InMemoryVectorStore()
        session = new_session_id()
        with pytest.raises(SessionIsolationError):
            store.upsert("", [chunk(session)], [[1.0]])

    def test_writing_a_foreign_chunk_is_blocked(self):
        # Defence in depth: even if a caller passes the wrong chunk, the store
        # refuses rather than mixing namespaces.
        store = InMemoryVectorStore()
        alice, bob = new_session_id(), new_session_id()

        with pytest.raises(SessionIsolationError):
            store.upsert(alice, [chunk(bob)], [[1.0]])

    def test_deleting_a_session_leaves_others_intact(self):
        store = InMemoryVectorStore()
        alice, bob = new_session_id(), new_session_id()
        store.upsert(alice, [chunk(alice)], [[1.0]])
        store.upsert(bob, [chunk(bob)], [[1.0]])

        store.delete_session(alice)

        assert store.count(alice) == 0
        assert store.count(bob) == 1

    def test_deleting_a_document_is_scoped_to_the_session(self):
        store = InMemoryVectorStore()
        alice, bob = new_session_id(), new_session_id()
        store.upsert(alice, [chunk(alice, document_id="shared-id")], [[1.0]])
        store.upsert(bob, [chunk(bob, document_id="shared-id")], [[1.0]])

        store.delete_document(alice, "shared-id")

        assert store.count(alice) == 0
        assert store.count(bob) == 1


class TestMetadataFiltering:
    def test_document_filter(self):
        store = InMemoryVectorStore()
        session = new_session_id()
        store.upsert(
            session,
            [chunk(session, document_id="d1"), chunk(session, document_id="d2", page=2)],
            [[1.0, 0.0], [1.0, 0.0]],
        )

        results = store.search(
            session, [1.0, 0.0], top_k=10, filters=SearchFilter(document_ids=["d1"])
        )
        assert [r.chunk.document_id for r in results] == ["d1"]

    def test_page_filter(self):
        store = InMemoryVectorStore()
        session = new_session_id()
        store.upsert(
            session,
            [chunk(session, page=1), chunk(session, page=17)],
            [[1.0], [1.0]],
        )

        results = store.search(
            session, [1.0], top_k=10, filters=SearchFilter(page_numbers=[17])
        )
        assert [r.chunk.page_number for r in results] == [17]

    def test_block_type_filter(self):
        from omnirag.core.enums import BlockType

        store = InMemoryVectorStore()
        session = new_session_id()
        text_chunk = chunk(session)
        chart_chunk = chunk(session, page=2)
        chart_chunk.block_type = BlockType.CHART
        store.upsert(session, [text_chunk, chart_chunk], [[1.0], [1.0]])

        results = store.search(
            session, [1.0], top_k=10, filters=SearchFilter(block_types=["chart"])
        )
        assert [r.chunk.block_type for r in results] == [BlockType.CHART]


class TestRegistryIsolation:
    def test_documents_are_listed_per_session(self):
        registry = DocumentRegistry()
        alice, bob = new_session_id(), new_session_id()

        registry.add(_summary(alice, "alice.pdf"))
        registry.add(_summary(bob, "bob.pdf"))

        assert [d.filename for d in registry.list(alice)] == ["alice.pdf"]
        assert [d.filename for d in registry.list(bob)] == ["bob.pdf"]

    def test_dedup_lookup_is_session_scoped(self):
        registry = DocumentRegistry()
        alice, bob = new_session_id(), new_session_id()
        registry.add(_summary(alice, "shared.pdf", content_hash="abc123"))

        assert registry.find_by_hash(alice, "abc123") is not None
        # The same file uploaded by another visitor must be processed for them.
        assert registry.find_by_hash(bob, "abc123") is None

    def test_clearing_one_session_leaves_the_other(self):
        registry = DocumentRegistry()
        alice, bob = new_session_id(), new_session_id()
        registry.add(_summary(alice, "a.pdf"))
        registry.add(_summary(bob, "b.pdf"))

        registry.clear(alice)

        assert registry.list(alice) == []
        assert len(registry.list(bob)) == 1

    def test_registry_requires_a_valid_session(self):
        registry = DocumentRegistry()
        with pytest.raises(SessionIsolationError):
            registry.list("")


class TestFileStoreIsolation:
    def test_clearing_a_session_removes_only_its_assets(self):
        store = MemoryFileStore()
        alice, bob = new_session_id(), new_session_id()

        alice_asset = store.put(alice, b"alice-bytes", media_type="image/png")
        bob_asset = store.put(bob, b"bob-bytes", media_type="image/png")

        store.delete_session(alice)

        assert store.get(alice_asset.asset_id) is None
        assert store.get(bob_asset.asset_id) == b"bob-bytes"

    def test_identical_bytes_in_different_sessions_get_different_ids(self):
        store = MemoryFileStore()
        alice, bob = new_session_id(), new_session_id()

        a = store.put(alice, b"same-bytes", media_type="image/png")
        b = store.put(bob, b"same-bytes", media_type="image/png")

        assert a.asset_id != b.asset_id


def _summary(session_id, filename, content_hash=""):
    return DocumentSummary(
        document_id=f"doc-{filename}-{session_id[:6]}",
        session_id=session_id,
        filename=filename,
        file_type=FileType.PDF,
        content_hash=content_hash,
    )
