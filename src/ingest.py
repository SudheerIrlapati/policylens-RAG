"""Day 1-3: load policy PDFs, split them into chunks, embed them, store them in ChromaDB.

Run:  python -m src.ingest
"""
import argparse
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


def load_pdfs(data_dir: str = config.DATA_DIR) -> list[Document]:
    """Load every PDF in data_dir. One Document per page, tagged with policy name and page number."""
    from langchain_community.document_loaders import PyPDFLoader

    pdfs = sorted(Path(data_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in '{data_dir}'. Add policy PDFs first.")

    pages: list[Document] = []
    for pdf in pdfs:
        for page in PyPDFLoader(str(pdf)).load():
            page.metadata["policy"] = pdf.stem
            # PyPDFLoader counts pages from 0; people count from 1.
            page.metadata["page_number"] = int(page.metadata.get("page", 0)) + 1
            pages.append(page)
        print(f"Loaded {pdf.name}")
    return pages


def split_docs(
    docs: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """Split pages into overlapping chunks. Metadata (policy, page_number) is copied to every chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return [c for c in chunks if c.page_content.strip()]


def build_index(chunks: list[Document]) -> None:
    """Embed chunks and save them to a fresh ChromaDB folder."""
    from langchain_chroma import Chroma

    # Rebuild from scratch so re-running ingest never creates duplicates.
    shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model=config.EMBED_MODEL),
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_DIR,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Index policy PDFs into ChromaDB")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--preview", action="store_true", help="Print 3 chunks and stop (no embedding cost)")
    args = parser.parse_args()

    pages = load_pdfs(args.data_dir)
    chunks = split_docs(pages)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")

    if args.preview:
        for c in chunks[:3]:
            print("-" * 60)
            print(c.metadata["policy"], "page", c.metadata["page_number"])
            print(c.page_content[:500])
        return

    build_index(chunks)
    print(f"Saved index to '{config.CHROMA_DIR}'")


if __name__ == "__main__":
    main()
