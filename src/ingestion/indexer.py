"""Chunking and indexing of parsed sections.

Scalability design
-------------------
This sample document is short enough that one chunk per section would work
fine. To also support "lengthy and complex documents" (per the brief), each
section is chunked independently on word-count windows with overlap, so a
future 200-page Conditions of Carriage doesn't produce single chunks too
large to embed usefully. Every chunk still carries its parent section's full
metadata (number, title, service scope, fare era) -- splitting a section
never detaches a fragment from the applicability rules that govern it.

The vector store itself sits behind a narrow interface (`upsert_chunks`,
`query`) backed here by a local Chroma collection. Swapping in a hosted
store (pgvector, Pinecone, etc.) for larger corpora is a change to this one
module, not to the parsing or retrieval logic that calls it.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from .models import Chunk, ParsedDocument, Section

DEFAULT_INDEX_DIR = Path("data/index")
COLLECTION_NAME = "conditions_of_carriage"

_CHUNK_WORDS = 180
_CHUNK_OVERLAP_WORDS = 30


def _chunk_section(section: Section) -> list[Chunk]:
    words = section.to_text().split()
    if len(words) <= _CHUNK_WORDS:
        windows = [words]
    else:
        windows = []
        start = 0
        step = _CHUNK_WORDS - _CHUNK_OVERLAP_WORDS
        while start < len(words):
            windows.append(words[start : start + _CHUNK_WORDS])
            start += step

    chunks = []
    for i, window in enumerate(windows):
        chunks.append(
            Chunk(
                chunk_id=f"section-{section.section_number or 'na'}-{i}",
                section_number=section.section_number,
                section_title=section.title,
                service_scope=section.service_scope,
                fare_era=section.fare_era,
                text=" ".join(window),
            )
        )
    return chunks


def build_chunks(parsed: ParsedDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in parsed.sections:
        chunks.extend(_chunk_section(section))
    return chunks


def index_chunks(chunks: list[Chunk], index_dir: Path = DEFAULT_INDEX_DIR) -> int:
    """Embed and persist chunks into the local vector store. Returns count indexed."""
    index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # Rebuild fresh each run: acquisition always re-fetches the live document,
    # so stale chunk ids from a previous version should not linger.
    existing = collection.get()["ids"]
    if existing:
        collection.delete(ids=existing)

    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "section_number": c.section_number or "",
                "section_title": c.section_title,
                "service_scope": ",".join(c.service_scope),
                "fare_era": c.fare_era,
                # Derived booleans, not the raw tags above, are what the
                # retrieval tool actually filters on: Chroma's `where` only
                # does exact/`$in` matching, so a comma-joined scope string
                # can't be queried as "contains domestic". A section with an
                # empty service_scope applies to *both* scopes, and one with
                # fare_era="unspecified" applies under *either* era -- both
                # collapse to True/True here so an empty/unspecified section
                # is never wrongly excluded by a scope- or era-filtered query.
                "applies_domestic": (not c.service_scope) or "domestic" in c.service_scope,
                "applies_international": (not c.service_scope) or "international" in c.service_scope,
                "applies_current": c.fare_era in ("current", "unspecified"),
                "applies_legacy": c.fare_era in ("legacy", "unspecified"),
            }
            for c in chunks
        ],
    )
    return len(chunks)


def query_index(
    query_text: str,
    n_results: int = 5,
    index_dir: Path = DEFAULT_INDEX_DIR,
    *,
    service_scope: str | None = None,
    fare_era: str | None = None,
):
    """Search the index, optionally filtered by service_scope / fare_era metadata.

    Caveat verified during ingestion testing: for a query like "baggage
    allowance for Japan on Economy Flex", *unfiltered* semantic search ranks
    Section 5 (Domestic) above Section 6 (International, which actually
    covers Japan under "Asia") -- the embedding doesn't reliably map a place
    name to domestic/international scope. This is exactly the confusion the
    brief warns about. Passing `service_scope`/`fare_era` once the caller has
    resolved them (e.g. after looking up which region a country belongs to)
    restricts results to sections that actually apply, instead of trusting
    semantic similarity alone to keep domestic and international -- or
    current and legacy -- apart.
    """
    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    conditions = []
    if service_scope:
        conditions.append({f"applies_{service_scope}": True})
    if fare_era:
        conditions.append({f"applies_{fare_era}": True})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    return collection.query(query_texts=[query_text], n_results=n_results, where=where)
