"""Chunking and indexing of parsed sections.

Scalability design
-------------------
This sample document is short enough that one chunk per section would work
fine. To also support "lengthy and complex documents" (per the brief), each
section is chunked independently with LangChain's RecursiveCharacterTextSplitter,
sized in *tokens of the actual embedding model* rather than words or raw
characters -- see `_token_length` below. Every chunk still carries its parent
section's full metadata (number, title, service scope, fare era) -- splitting
a section never detaches a fragment from the applicability rules that govern
it.

The vector store itself sits behind a narrow interface (`index_chunks`,
`query_index`) backed here by a local Chroma collection. Swapping in a hosted
store (pgvector, Pinecone, etc.) for larger corpora is a change to this one
module, not to the parsing or retrieval logic that calls it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import Chunk, ParsedDocument, Section

DEFAULT_INDEX_DIR = Path("data/index")
COLLECTION_NAME = "conditions_of_carriage"

_CHUNK_OVERLAP_TOKENS = 40

# Everything below touches onnxruntime/tokenizers, which have a known
# native-level teardown crash on macOS ("recursive_mutex lock failed") when
# more than one embedding-function instance gets constructed in a process
# lifetime. Query-only entry points (the agent's search tool) never chunk a
# document, so building the splitter eagerly at import time would force this
# extra instantiation on every run for no benefit -- keep it fully lazy, and
# only pay for it in the ingestion pipeline that actually chunks sections.


@lru_cache(maxsize=1)
def _raw_tokenizer():
    """A copy of the embedding model's own tokenizer, without truncation/padding.

    Chroma's cached tokenizer (`ONNXMiniLM_L6_V2().tokenizer`) has truncation
    and padding to 256 baked in for embedding calls -- measuring length with
    it would make every text report back capped at 256, which is useless for
    deciding *where* to split. This loads the identical tokenizer.json the
    embedder uses, so counts agree with what will actually be embedded.
    """
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
    from tokenizers import Tokenizer

    embedding_fn = ONNXMiniLM_L6_V2()
    embedding_fn._download_model_if_not_exists()  # same one-time fetch Chroma itself does
    tokenizer_path = Path(embedding_fn.DOWNLOAD_PATH) / embedding_fn.EXTRACTED_FOLDER_NAME / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def _token_length(text: str) -> int:
    return len(_raw_tokenizer().encode(text).ids)


@lru_cache(maxsize=1)
def _splitter() -> RecursiveCharacterTextSplitter:
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    # Chroma's default embedding function -- what actually embeds our chunks
    # -- is all-MiniLM-L6-v2 with a hard 256-token limit: it *raises*, rather
    # than silently truncating, if a document exceeds it. So "token-aware"
    # here means literally the token count this model's own tokenizer
    # produces, not an estimate from words or characters.
    max_embed_tokens = ONNXMiniLM_L6_V2().max_tokens()  # 256
    chunk_tokens = max_embed_tokens - 32  # safety margin below the hard limit
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=_CHUNK_OVERLAP_TOKENS,
        length_function=_token_length,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _chunk_section(section: Section) -> list[Chunk]:
    text = section.to_text()
    pieces = _splitter().split_text(text) if text else []
    return [
        Chunk(
            chunk_id=f"section-{section.section_number or 'na'}-{i}",
            section_number=section.section_number,
            section_title=section.title,
            service_scope=section.service_scope,
            fare_era=section.fare_era,
            text=piece,
        )
        for i, piece in enumerate(pieces)
    ]


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
