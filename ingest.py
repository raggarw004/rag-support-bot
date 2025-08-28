import os, json, re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from bs4 import BeautifulSoup
import html2text
from pypdf import PdfReader

from config import (
    EMBEDDING_MODEL, INDEX_DIR, DOCS_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP
)

SUPPORTED = {".txt", ".md", ".pdf", ".html", ".htm"}

def read_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif ext in {".html", ".htm"}:
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        # fallback with html2text for safer conversion
        h = html2text.HTML2Text()
        h.ignore_images = True
        h.ignore_links = False
        return h.handle(str(soup))
    elif ext == ".pdf":
        text = []
        with open(path, "rb") as f:
            pdf = PdfReader(f)
            for page in pdf.pages:
                try:
                    text.append(page.extract_text() or "")
                except Exception:
                    pass
        return "\n".join(text)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def clean_text(s: str) -> str:
    s = re.sub(r"\u00a0", " ", s)   # nbsp
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def build_index(docs_dir: str, index_dir: str):
    os.makedirs(index_dir, exist_ok=True)
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = []
    metas = []

    files = []
    for p in Path(docs_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            files.append(p)
    if not files:
        raise SystemExit(f"No supported files found in {docs_dir}. Add PDFs/TXT/MD/HTML and retry.")

    print(f"Found {len(files)} files. Reading & chunking...")
    for path in tqdm(files):
        raw = read_text(path)
        txt = clean_text(raw)
        if not txt:
            continue
        chunks = chunk_text(txt, CHUNK_SIZE, CHUNK_OVERLAP)
        for ci, c in enumerate(chunks):
            texts.append(c)
            metas.append({
                "source": str(path),
                "chunk": ci
            })

    print(f"Total chunks: {len(texts)}. Embedding...")
    embs = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    dim = embs.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine with normalized vectors
    index.add(embs)

    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    np.save(os.path.join(index_dir, "embeddings.npy"), embs)  # optional (debug)
    with open(os.path.join(index_dir, "meta.json"), "w") as f:
        json.dump({"texts": texts, "metas": metas}, f)

    print(f"Saved index to {index_dir}. Done.")

if __name__ == "__main__":
    build_index(DOCS_DIR, INDEX_DIR)
