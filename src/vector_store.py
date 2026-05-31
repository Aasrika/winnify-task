import numpy as np
import faiss
from typing import List
from models import EMBED_MODEL


class VectorStore:
    def __init__(self, chunks: List[dict]):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        embeddings = EMBED_MODEL.encode(texts, show_progress_bar=False).astype("float32")
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.embeddings = embeddings

    def retrieve(self, query: str, k: int = 4) -> List[dict]:
        enriched_query = f"Python Fundamentals: {query} educational concepts examples"
        q_emb = EMBED_MODEL.encode([enriched_query]).astype("float32")
        faiss.normalize_L2(q_emb)
        _, indices = self.index.search(q_emb, k)
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        id_set = set(chunk_ids)
        return [c for c in self.chunks if c["chunk_id"] in id_set]