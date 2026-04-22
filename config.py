# config.py — Centralized configuration for SPCOET Smart AI Chatbot

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent


class Config:
    # ── Paths ──────────────────────────────────────────────────
    KNOWLEDGE_BASE_PATH = str(BASE_DIR / "data" / "knowledge_base.json")
    FAISS_INDEX_PATH    = str(BASE_DIR / "data" / "faiss_index.bin")
    CHUNKS_PATH         = str(BASE_DIR / "data" / "chunks.json")
    LOG_DIR             = str(BASE_DIR / "data" / "logs")

    # ── AI API ─────────────────────────────────────────────────
    ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")

    # Which AI provider to use: "anthropic" or "openai"
    AI_PROVIDER         = os.getenv("AI_PROVIDER", "anthropic")
    AI_MODEL_ANTHROPIC  = "claude-haiku-4-5-20251001"
    AI_MODEL_OPENAI     = "gpt-4o-mini"

    # ── Embeddings ─────────────────────────────────────────────
    # Free local model — no API key needed!
    EMBEDDING_MODEL     = "all-MiniLM-L6-v2"
    EMBEDDING_DIM       = 384          # dimension for all-MiniLM-L6-v2
    CHUNK_SIZE          = 300          # words per chunk
    CHUNK_OVERLAP       = 50           # overlap between chunks

    # ── Search ─────────────────────────────────────────────────
    SIMILARITY_THRESHOLD = 0.35        # min score to use as context
    TOP_K_RESULTS        = 4           # top results to fetch

    # ── Memory ─────────────────────────────────────────────────
    MAX_HISTORY          = 5           # last N message pairs to remember

    # ── App ────────────────────────────────────────────────────
    SECRET_KEY           = os.getenv("SECRET_KEY", "spcoet-2024-secret")
    ADMIN_KEY            = os.getenv("ADMIN_KEY",  "spcoet-admin-2024")
    MAX_MESSAGE_LENGTH   = 500         # chars
    FLASK_DEBUG          = os.getenv("FLASK_DEBUG", "false").lower() == "true"
