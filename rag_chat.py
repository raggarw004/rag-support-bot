import os, json
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama

from config import (
    EMBEDDING_MODEL, INDEX_DIR, TOP_K, DEFAULT_OLLAMA_MODEL,
    SYSTEM_PROMPT, CONTEXT_CHUNK_TEMPLATE
)

class RAGChatbot:
    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL):
        # Load FAISS & metadata
        idx_path = os.path.join(INDEX_DIR, "index.faiss")
        meta_path = os.path.join(INDEX_DIR, "meta.json")
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            raise RuntimeError("Vector index not found. Run `python ingest.py` first.")
        self.index = faiss.read_index(idx_path)
        with open(meta_path, "r") as f:
            store = json.load(f)
        self.texts = store["texts"]
        self.metas = store["metas"]

        # Embedding model
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # LLM (Ollama local)
        self.model_name = model_name

    def embed_query(self, q: str) -> np.ndarray:
        qv = self.embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        return qv.astype(np.float32)

    def retrieve(self, q: str, k: int = TOP_K) -> List[dict]:
        qv = self.embed_query(q)
        D, I = self.index.search(qv, k)
        hits = []
        for idx, score in zip(I[0], D[0]):
            rec = self.metas[idx].copy()
            rec["content"] = self.texts[idx]
            rec["score"] = float(score)
            hits.append(rec)
        return hits

    def build_context(self, hits: List[dict]) -> str:
        parts = []
        for h in hits:
            src = os.path.basename(h["source"])
            parts.append(CONTEXT_CHUNK_TEMPLATE.format(source=src, content=h["content"]))
        return "\n\n".join(parts)

    def generate(self, question: str, k: int = TOP_K) -> Tuple[str, List[dict]]:
        hits = self.retrieve(question, k=k)
        context = self.build_context(hits)

        user_prompt = f"""CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
Follow the System Prompt. Answer only with the provided CONTEXT. If insufficient, say "I don't know based on the current documentation." Include brief [source: filename] citations.
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.2}  # more deterministic for support
        )
        answer = response["message"]["content"]
        return answer, hits

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    args = parser.parse_args()

    bot = RAGChatbot(model_name=args.model)
    print("RAG Support Bot ready. Type your question (Ctrl+C to exit).")
    try:
        while True:
            q = input("\nYou: ").strip()
            if not q:
                continue
            ans, hits = bot.generate(q)
            print("\nAssistant:\n", ans)
            print("\nTop sources:")
            for h in hits:
                print(f"- {os.path.basename(h['source'])} (score={h['score']:.3f})")
    except KeyboardInterrupt:
        print("\nBye!")
