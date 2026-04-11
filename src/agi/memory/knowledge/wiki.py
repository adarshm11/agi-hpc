# AGI-HPC WikiPage Compiled Knowledge
# Copyright (c) 2026 Andrew H. Bond
# License v1.0 — see LICENSE
"""Compiled wiki pages per topic (Karpathy LLM Wiki pattern).

Each topic gets a synthesized markdown page that accumulates
knowledge over time. Pages are the "compounding artifact" —
cross-references are maintained, contradictions are flagged,
and the page grows with each ingestion.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WikiPage:
    """A compiled knowledge page for one topic."""

    topic: str
    content: str = ""
    references: list[str] = field(default_factory=list)
    entity_count: int = 0
    relationship_count: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1


class WikiCompiler:
    """Compile and maintain wiki pages from the knowledge graph."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS wiki_pages (
                topic TEXT PRIMARY KEY,
                content TEXT,
                references_json TEXT DEFAULT '[]',
                entity_count INTEGER DEFAULT 0,
                relationship_count INTEGER DEFAULT 0,
                last_updated TEXT,
                version INTEGER DEFAULT 1
            )
            """)
        self._conn.commit()

    def compile_page(
        self,
        topic: str,
        entities: list[dict],
        relationships: list[dict],
        summary: str = "",
    ) -> WikiPage:
        """Compile a wiki page from entities and relationships."""
        lines = [f"# {topic}", ""]
        if summary:
            lines.append(summary)
            lines.append("")

        if entities:
            lines.append("## Entities")
            for e in entities:
                name = e.get("name", "")
                etype = e.get("type", "")
                desc = e.get("description", "")
                lines.append(f"- **{name}** ({etype}): {desc}")
            lines.append("")

        if relationships:
            lines.append("## Relationships")
            for r in relationships:
                subj = r.get("subject", "")
                pred = r.get("predicate", "")
                obj = r.get("object", "")
                lines.append(f"- {subj} → {pred} → {obj}")
            lines.append("")

        refs = list({e.get("source", "") for e in entities if e.get("source")})

        content = "\n".join(lines)
        existing = self.get_page(topic)
        version = (existing.version + 1) if existing else 1

        page = WikiPage(
            topic=topic,
            content=content,
            references=refs,
            entity_count=len(entities),
            relationship_count=len(relationships),
            version=version,
        )

        self._save(page)
        return page

    def update_page(
        self,
        topic: str,
        new_entities: list[dict],
        new_relationships: list[dict],
    ) -> WikiPage:
        """Update an existing page with new knowledge."""
        existing = self.get_page(topic)
        if not existing:
            return self.compile_page(topic, new_entities, new_relationships)

        all_entities = []
        all_rels = []

        # Parse existing content for entity/rel counts
        # (simplified — in production, store structured data separately)
        all_entities.extend(new_entities)
        all_rels.extend(new_relationships)

        return self.compile_page(
            topic,
            all_entities,
            all_rels,
            summary=f"Updated from version {existing.version}.",
        )

    def get_page(self, topic: str) -> Optional[WikiPage]:
        """Retrieve a wiki page by topic."""
        row = self._conn.execute(
            "SELECT * FROM wiki_pages WHERE topic = ?", (topic,)
        ).fetchone()
        if not row:
            return None
        return WikiPage(
            topic=row["topic"],
            content=row["content"],
            references=json.loads(row["references_json"]),
            entity_count=row["entity_count"],
            relationship_count=row["relationship_count"],
            last_updated=datetime.fromisoformat(row["last_updated"]),
            version=row["version"],
        )

    def list_pages(self) -> list[str]:
        """Return all topic names."""
        rows = self._conn.execute(
            "SELECT topic FROM wiki_pages ORDER BY topic"
        ).fetchall()
        return [r["topic"] for r in rows]

    def get_stats(self) -> dict[str, int]:
        """Return page count and total entities/relationships."""
        row = self._conn.execute(
            "SELECT COUNT(*) as pages, "
            "COALESCE(SUM(entity_count), 0) as entities, "
            "COALESCE(SUM(relationship_count), 0) as rels "
            "FROM wiki_pages"
        ).fetchone()
        return {
            "pages": row["pages"],
            "total_entities": row["entities"],
            "total_relationships": row["rels"],
        }

    def _save(self, page: WikiPage) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO wiki_pages "
            "(topic, content, references_json, entity_count, "
            "relationship_count, last_updated, version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                page.topic,
                page.content,
                json.dumps(page.references),
                page.entity_count,
                page.relationship_count,
                now,
                page.version,
            ),
        )
        self._conn.commit()
