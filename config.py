# Global config knobs you may tweak

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast & free
INDEX_DIR = "vectorstore"
DOCS_DIR = "data/faqs"
CHUNK_SIZE = 800        # words per chunk
CHUNK_OVERLAP = 150     # words overlap
TOP_K = 5               # retrieved chunks per query
DEFAULT_OLLAMA_MODEL = "llama3:8b"   # or "llama2", "mistral", etc.

SYSTEM_PROMPT = """You are a helpful, accurate Customer Support assistant.
Answer ONLY using the provided CONTEXT. If the answer is not present, say "I don't know based on the current documentation."
Prefer bullet points for lists. Quote key snippets where useful. Include brief source attributions like [source: filename]."""

# Optional: format for each retrieved chunk when building the prompt
CONTEXT_CHUNK_TEMPLATE = """[Source: {source}]
{content}
"""
