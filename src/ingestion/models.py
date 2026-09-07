"""Typed representations of an acquired and parsed Conditions of Carriage document.

Keeping these as explicit models (rather than passing dicts around) is what lets
the parser, indexer and any future retrieval/generation stage agree on a shape
without re-deriving it from the docx structure each time.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ServiceScope = Literal["domestic", "international"]
FareEra = Literal["current", "legacy", "unspecified"]


class TableRow(BaseModel):
    """One row of a source table, keyed by its column headers.

    Keeping cells keyed (rather than flattening to a string) preserves the
    row/column association -- e.g. that "30 kg" belongs to Asia + Economy
    Saver -- which is exactly the detail a naive text dump loses.
    """

    cells: dict[str, str]


class Table(BaseModel):
    headers: list[str]
    rows: list[TableRow]

    def to_text(self) -> str:
        lines = []
        for row in self.rows:
            lines.append(", ".join(f"{col}: {val}" for col, val in row.cells.items()))
        return "\n".join(lines)


class Section(BaseModel):
    """A single numbered clause of the Conditions of Carriage."""

    section_number: str | None
    title: str
    paragraphs: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)

    # Applicability metadata, inferred heuristically from the section's own
    # title/text. Empty `service_scope` means the section applies regardless
    # of domestic/international routing (e.g. carry-on rules).
    service_scope: list[ServiceScope] = Field(default_factory=list)
    fare_era: FareEra = "unspecified"

    def to_text(self) -> str:
        heading = f"Section {self.section_number}: {self.title}" if self.section_number else self.title
        parts = [heading, *self.paragraphs, *(f"- {b}" for b in self.bullets)]
        parts.extend(t.to_text() for t in self.tables)
        return "\n".join(p for p in parts if p)


class DocumentMetadata(BaseModel):
    title: str
    version: str | None = None
    effective_date: str | None = None
    source_url: str
    retrieved_at: str
    sha256: str
    size_bytes: int


class ParsedDocument(BaseModel):
    metadata: DocumentMetadata
    sections: list[Section]


class Chunk(BaseModel):
    """A retrieval unit produced from a Section.

    Short sections become exactly one chunk. Long sections (the scalability
    concern the assessment calls out -- lengthy/complex documents) are split
    into overlapping word-window chunks, each still carrying the *full*
    section-level metadata so a retriever never has to guess which fare era
    or destination region a fragment belongs to.
    """

    chunk_id: str
    section_number: str | None
    section_title: str
    service_scope: list[ServiceScope]
    fare_era: FareEra
    text: str
