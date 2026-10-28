"""
ingest.py  –  LexPal Ingestion Pipeline
========================================
Reads every PDF in data/, extracts text, chunks it (section-aware),
embeds each chunk, and upserts into ChromaDB.

Run:
    python ingest.py
    python ingest.py --data-dir path/to/pdfs --act-name "My Act"
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("lexpal.ingest")


def ingest_file(pdf_path: Path, act_name: str, collection, embed_fn) -> int:
    """Ingest a single PDF file into ChromaDB. Returns number of chunks added."""
    from utils.extractor import extract_pdf
    from utils.chunker import chunk_document
    from config import CHUNK_SIZE, CHUNK_OVERLAP

    logger.info(f"Processing: {pdf_path.name}")
    text = extract_pdf(pdf_path)
    chunks = chunk_document(text, act_name, CHUNK_SIZE, CHUNK_OVERLAP)

    if not chunks:
        logger.warning(f"No chunks produced from {pdf_path.name}")
        return 0

    texts      = [c.text for c in chunks]
    metadatas  = [c.to_metadata() for c in chunks]
    ids        = [c.chunk_id for c in chunks]
    embeddings = embed_fn(texts)

    # Upsert in batches of 100 to avoid memory spikes
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=texts[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
    logger.info(f"  ✔ {len(chunks)} chunks ingested from '{act_name}'")
    return len(chunks)


def build_embed_fn():
    """Return a function that embeds a list of strings."""
    from sentence_transformers import SentenceTransformer
    from config import EMBED_MODEL
    model = SentenceTransformer(EMBED_MODEL)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, show_progress_bar=False).tolist()

    return embed


def get_collection():
    """Return (or create) the ChromaDB collection."""
    import chromadb
    from config import CHROMA_DIR, CHROMA_COLLECTION

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ── Act name inference ────────────────────────────────────────────────────────
# Maps lowercase filename stems to human-readable act names
FILENAME_ACT_MAP = {
    "consumer_protection":      "Consumer Protection Act 2019",
    "consumer":                 "Consumer Protection Act 2019",
    "negotiable_instruments":   "Negotiable Instruments Act 1881",
    "negotiable":               "Negotiable Instruments Act 1881",
    "ni_act":                   "Negotiable Instruments Act 1881",
    "information_technology":   "Information Technology Act 2000",
    "it_act":                   "Information Technology Act 2000",
    "maharashtra_rent":         "Maharashtra Rent Control Act 1999",
    "rent_control":             "Maharashtra Rent Control Act 1999",
    "rti":                      "Right to Information Act 2005",
    "right_to_information":     "Right to Information Act 2005",
}


def infer_act_name(path: Path) -> str:
    stem = path.stem.lower().replace(" ", "_").replace("-", "_")
    for key, name in FILENAME_ACT_MAP.items():
        if key in stem:
            return name
    # Fallback: title-case the filename stem
    return path.stem.replace("_", " ").replace("-", " ").title()


def main():
    parser = argparse.ArgumentParser(description="LexPal ingestion pipeline")
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory containing law PDFs (default: data/)"
    )
    parser.add_argument(
        "--act-name", default=None,
        help="Override act name (used when ingesting a single file)"
    )
    args = parser.parse_args()

    from config import DATA_DIR
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    pdfs = list(data_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"No PDF files found in {data_dir}. Place law PDFs there first.")
        sys.exit(0)

    logger.info(f"Found {len(pdfs)} PDF(s) to ingest.")

    embed_fn   = build_embed_fn()
    collection = get_collection()

    total_chunks = 0
    for pdf in pdfs:
        act_name = args.act_name if args.act_name else infer_act_name(pdf)
        total_chunks += ingest_file(pdf, act_name, collection, embed_fn)

    logger.info(f"\n✅ Ingestion complete. Total chunks in DB: {total_chunks}")


if __name__ == "__main__":
    main()
