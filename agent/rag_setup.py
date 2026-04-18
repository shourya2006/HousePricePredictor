import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


ROOT_DIR = Path(__file__).parent.parent
KB_PATH = ROOT_DIR / "knowledge_base" / "real_estate_insights.txt"
CHROMA_PATH = ROOT_DIR / "chroma_db"

COLLECTION_NAME = "real_estate_knowledge"


def build_vector_store(force_rebuild: bool = False) -> Chroma:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    if CHROMA_PATH.exists() and not force_rebuild:
        print(f"[RAG] Loading existing ChromaDB from: {CHROMA_PATH}")
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_PATH),
        )
        count = vector_store._collection.count()
        print(f"[RAG] Loaded {count} document chunks from ChromaDB.")
        return vector_store

    print(f"[RAG] Building ChromaDB from: {KB_PATH}")

    loader = TextLoader(str(KB_PATH), encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n---\n", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"[RAG] Split into {len(chunks)} chunks.")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PATH),
    )
    print(f"[RAG] Embedded and persisted {len(chunks)} chunks to ChromaDB.")
    return vector_store


def get_retriever(k: int = 4):
    vector_store = build_vector_store()
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


if __name__ == "__main__":
    force = "--rebuild" in sys.argv
    build_vector_store(force_rebuild=force)
    print("[RAG] Setup complete. ChromaDB is ready.")
