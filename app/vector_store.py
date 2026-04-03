
import uuid
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger

from app.chunker import Chunk


class VectorStore:

    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "documents"):
        logger.info("جاري تحميل نموذج الـ embeddings...")
        self.model = SentenceTransformer(self.EMBEDDING_MODEL)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # طريقة حساب التشابه
        )

        logger.success(f"ChromaDB جاهز | المجموعة: {collection_name} | "
                       f"عدد الـ chunks المحفوظة: {self.collection.count()}")

    def add_chunks(self, chunks: List[Chunk], document_id: int, file_name: str = "") -> List[str]:
        if not chunks:
            return []

        texts    = [chunk.text for chunk in chunks]
        ids      = [str(uuid.uuid4()) for _ in chunks]
        metadata = [
            {
                "document_id":  document_id,
                "chunk_index":  chunk.index,
                "strategy":     chunk.strategy,
                "word_count":   chunk.word_count(),
                "file_name":    file_name,
                "section":      chunk.metadata.get("section", ""),
            }
            for chunk in chunks
        ]

        logger.info(f"جاري تحويل {len(texts)} chunk إلى vectors...")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,  # تطبيع لتحسين الدقة
        ).tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadata,
        )

        logger.success(f"تم حفظ {len(chunks)} chunk في ChromaDB")
        return ids

    def search(
        self,
        query: str,
        n_results: int = 5,
        document_id: Optional[int] = None,
        language_filter: Optional[str] = None,
    ) -> List[dict]:
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()

        where_filter = None
        if document_id is not None:
            where_filter = {"document_id": {"$eq": document_id}}

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, self.collection.count() or 1),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        formatted = []
        if results and results["documents"]:
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                similarity = 1 - dist

                formatted.append({
                    "rank":        i + 1,
                    "text":        doc,
                    "similarity":  round(similarity, 4),
                    "document_id": meta.get("document_id"),
                    "chunk_index": meta.get("chunk_index"),
                    "file_name":   meta.get("file_name", ""),
                    "section":     meta.get("section", ""),
                    "strategy":    meta.get("strategy", ""),
                })

        return formatted

    def delete_document(self, document_id: int):
        self.collection.delete(where={"document_id": {"$eq": document_id}})
        logger.info(f"تم حذف chunks الوثيقة {document_id} من ChromaDB")

    def get_stats(self) -> dict:
        return {
            "total_vectors": self.collection.count(),
            "model_used":    self.EMBEDDING_MODEL,
        }
