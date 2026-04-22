# core/embeddings.py
# ──────────────────────────────────────────────────────────────
#  Semantic Embedding Engine
#  Model  : sentence-transformers/all-MiniLM-L6-v2 (free, local)
#  Storage: FAISS vector index
#  Flow   : KB text → chunks → embed → FAISS index
#           Query → embed → FAISS similarity search → top chunks
# ──────────────────────────────────────────────────────────────

import os, json, pickle, logging
import numpy as np

# ── Sentence Transformers (free embeddings) ───────────────────
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Embeddings disabled.")

# ── FAISS (vector similarity search) ─────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not installed. Using cosine similarity fallback.")


class EmbeddingEngine:
    """
    Converts knowledge base text into vector embeddings,
    stores them in FAISS, and performs fast semantic search.
    """

    def __init__(self, kb_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.kb_path     = kb_path
        self.model_name  = model_name
        self.model       = None
        self.index       = None      # FAISS index
        self.chunks      = []        # list of {"content", "section", "source"}
        self.embeddings  = []        # numpy array of embeddings
        self.dim         = 384       # all-MiniLM-L6-v2 dimension

        self._load_model()
        self._load_or_build_index()

    # ── Model Loading ─────────────────────────────────────────
    def _load_model(self):
        if not ST_AVAILABLE:
            logging.warning("SentenceTransformer not available")
            return
        try:
            print(f"  📦 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"  ✅ Embedding model loaded (dim={self.dim})")
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            self.model = None

    # ── Index Build / Load ────────────────────────────────────
    def _load_or_build_index(self):
        """Load existing FAISS index or build a new one from KB."""
        chunks_path = self.kb_path.replace("knowledge_base.json", "chunks.json")
        index_path  = self.kb_path.replace("knowledge_base.json", "faiss_index.bin")

        # Try loading existing index
        if os.path.exists(chunks_path) and os.path.exists(index_path):
            try:
                with open(chunks_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                if FAISS_AVAILABLE:
                    self.index = faiss.read_index(index_path)
                else:
                    emb_path = index_path.replace(".bin", "_embs.npy")
                    if os.path.exists(emb_path):
                        self.embeddings = np.load(emb_path)
                print(f"  ✅ Loaded FAISS index ({len(self.chunks)} chunks)")
                return
            except Exception as e:
                logging.warning(f"Could not load existing index: {e}")

        # Build fresh index from KB
        self.rebuild_index()

    def rebuild_index(self):
        """
        (Re)build FAISS index from the current knowledge base.
        Call this after updating the KB.
        """
        print("  🔨 Building embeddings index from knowledge base...")
        try:
            kb = self._load_kb()
            self.chunks = self._chunk_knowledge_base(kb)

            if not self.chunks:
                logging.warning("No chunks generated from KB")
                return

            if self.model is None:
                print("  ⚠️  No embedding model — semantic search disabled")
                return

            # Embed all chunks
            texts      = [c["content"] for c in self.chunks]
            embeddings = self.model.encode(
                texts, show_progress_bar=False,
                batch_size=32, normalize_embeddings=True
            )

            # Build FAISS index
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatIP(self.dim)   # Inner Product (cosine after normalize)
                self.index.add(np.array(embeddings, dtype="float32"))

                # Save index
                idx_path  = self.kb_path.replace("knowledge_base.json", "faiss_index.bin")
                faiss.write_index(self.index, idx_path)
            else:
                # Fallback: store as numpy array
                self.embeddings = embeddings
                np.save(self.kb_path.replace("knowledge_base.json","faiss_index_embs.npy"), embeddings)

            # Save chunks
            chunks_path = self.kb_path.replace("knowledge_base.json", "chunks.json")
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)

            print(f"  ✅ Index built: {len(self.chunks)} chunks, dim={self.dim}")
        except Exception as e:
            logging.error(f"rebuild_index error: {e}")

    # ── Semantic Search ───────────────────────────────────────
    def search(self, query: str, top_k: int = 4) -> list:
        """
        Convert query to embedding, search FAISS, return top matches.
        Returns: [{"content", "section", "source", "score"}]
        """
        if not self.chunks or self.model is None:
            return self._keyword_fallback(query, top_k)

        try:
            # Embed query
            q_emb = self.model.encode(
                [query], normalize_embeddings=True
            ).astype("float32")

            # FAISS search
            if FAISS_AVAILABLE and self.index is not None:
                scores, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and idx < len(self.chunks):
                        chunk = self.chunks[idx].copy()
                        chunk["score"] = float(score)
                        results.append(chunk)
            else:
                # Numpy cosine fallback
                if len(self.embeddings) == 0:
                    return self._keyword_fallback(query, top_k)
                sims    = np.dot(self.embeddings, q_emb[0])
                top_idx = np.argsort(sims)[::-1][:top_k]
                results = []
                for idx in top_idx:
                    chunk = self.chunks[idx].copy()
                    chunk["score"] = float(sims[idx])
                    results.append(chunk)

            # Sort by score descending
            results.sort(key=lambda x: x["score"], reverse=True)
            return results

        except Exception as e:
            logging.error(f"Search error: {e}")
            return self._keyword_fallback(query, top_k)

    # ── KB Loading ────────────────────────────────────────────
    def _load_kb(self) -> dict:
        """Load knowledge base JSON file."""
        if not os.path.exists(self.kb_path):
            logging.error(f"KB file not found: {self.kb_path}")
            return {}
        with open(self.kb_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── Text Chunking ─────────────────────────────────────────
    def _chunk_knowledge_base(self, kb: dict) -> list:
        """
        Convert knowledge base dict into text chunks for embedding.
        Each section → multiple 300-word chunks with 50-word overlap.
        """
        chunks = []

        def add_chunk(text: str, section: str, subsection: str = ""):
            text = text.strip()
            if len(text) < 20:   # skip very short texts
                return
            chunks.append({
                "content"    : text,
                "section"    : section,
                "subsection" : subsection,
                "source"     : f"{section} → {subsection}".strip(" →")
            })

        def flatten_and_chunk(value, section: str, subsection: str = "", max_words: int = 300):
            """Recursively flatten KB structure and chunk long texts."""
            if isinstance(value, str):
                words = value.split()
                if len(words) <= max_words:
                    add_chunk(value, section, subsection)
                else:
                    # Sliding window chunking with overlap
                    step = max_words - 50   # 50-word overlap
                    for i in range(0, len(words), step):
                        chunk_text = " ".join(words[i:i + max_words])
                        add_chunk(chunk_text, section, subsection)
                        if i + max_words >= len(words):
                            break

            elif isinstance(value, dict):
                for k, v in value.items():
                    sub = f"{subsection} {k}".strip() if subsection else k
                    # Add key as context prefix
                    if isinstance(v, str):
                        prefixed = f"{k}: {v}"
                        flatten_and_chunk(prefixed, section, sub, max_words)
                    else:
                        flatten_and_chunk(v, section, sub, max_words)

            elif isinstance(value, list):
                for i, item in enumerate(value):
                    sub = f"{subsection}[{i}]" if subsection else f"item_{i}"
                    if isinstance(item, str):
                        flatten_and_chunk(item, section, subsection, max_words)
                    elif isinstance(item, dict):
                        # Combine dict items into one chunk for context
                        combined = " | ".join(f"{k}: {v}" for k, v in item.items() if isinstance(v, str))
                        flatten_and_chunk(combined, section, subsection, max_words)

        for section, content in kb.items():
            flatten_and_chunk(content, section)

        return chunks

    # ── Keyword Fallback ─────────────────────────────────────
    def _keyword_fallback(self, query: str, top_k: int) -> list:
        """Simple keyword matching when embeddings are unavailable."""
        if not self.chunks:
            return []
        query_words = set(query.lower().split())
        results = []
        for chunk in self.chunks:
            chunk_words = set(chunk["content"].lower().split())
            overlap     = len(query_words & chunk_words)
            if overlap > 0:
                c = chunk.copy()
                c["score"] = overlap / max(len(query_words), 1)
                results.append(c)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ── Index Info ────────────────────────────────────────────
    def get_index_info(self) -> dict:
        return {
            "total_chunks"   : len(self.chunks),
            "model"          : self.model_name if self.model else "none",
            "faiss_available": FAISS_AVAILABLE,
            "st_available"   : ST_AVAILABLE,
            "index_ready"    : self.index is not None or len(self.embeddings) > 0
        }
