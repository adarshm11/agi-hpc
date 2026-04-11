# AGI-HPC WikiPage Tests
from __future__ import annotations

import pytest

from agi.memory.knowledge.wiki import WikiCompiler


@pytest.fixture
def compiler():
    return WikiCompiler(db_path=":memory:")


ENTITIES = [
    {
        "name": "PCA",
        "type": "method",
        "description": "Dim reduction",
        "source": "doc.md",
    },
    {
        "name": "BGE-M3",
        "type": "tool",
        "description": "Embedding model",
        "source": "doc.md",
    },
]

RELS = [
    {"subject": "PCA", "predicate": "used_by", "object": "BGE-M3"},
]


class TestCompile:
    def test_creates_page(self, compiler):
        page = compiler.compile_page("PCA", ENTITIES, RELS, "PCA is great.")
        assert page.topic == "PCA"
        assert "# PCA" in page.content
        assert page.entity_count == 2
        assert page.relationship_count == 1
        assert page.version == 1

    def test_version_increments(self, compiler):
        compiler.compile_page("PCA", ENTITIES, RELS)
        page2 = compiler.compile_page("PCA", ENTITIES, RELS)
        assert page2.version == 2

    def test_empty_entities(self, compiler):
        page = compiler.compile_page("Empty", [], [])
        assert page.entity_count == 0
        assert "# Empty" in page.content


class TestGetPage:
    def test_exists(self, compiler):
        compiler.compile_page("PCA", ENTITIES, RELS)
        page = compiler.get_page("PCA")
        assert page is not None
        assert page.topic == "PCA"

    def test_not_found(self, compiler):
        assert compiler.get_page("nonexistent") is None


class TestListPages:
    def test_empty(self, compiler):
        assert compiler.list_pages() == []

    def test_after_compile(self, compiler):
        compiler.compile_page("PCA", ENTITIES, RELS)
        compiler.compile_page("BGE", ENTITIES, RELS)
        pages = compiler.list_pages()
        assert len(pages) == 2
        assert "PCA" in pages


class TestUpdate:
    def test_update_nonexistent(self, compiler):
        page = compiler.update_page("New", ENTITIES, RELS)
        assert page.topic == "New"
        assert page.entity_count == 2

    def test_update_existing(self, compiler):
        compiler.compile_page("PCA", ENTITIES, RELS)
        new_ent = [{"name": "SVD", "type": "method", "description": "Singular value"}]
        page = compiler.update_page("PCA", new_ent, [])
        assert page.version == 2


class TestStats:
    def test_empty(self, compiler):
        stats = compiler.get_stats()
        assert stats["pages"] == 0

    def test_after_compile(self, compiler):
        compiler.compile_page("PCA", ENTITIES, RELS)
        stats = compiler.get_stats()
        assert stats["pages"] == 1
        assert stats["total_entities"] == 2
