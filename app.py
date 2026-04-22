# ============================================================
#  SPCOET Smart AI Chatbot — Main Flask Application
#  Author  : Final Year Project Team
#  Stack   : Flask + spaCy + FAISS + Sentence Transformers + AI API
# ============================================================

import os, json, time, uuid, logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our custom modules
from config import Config
from core.nlp_processor  import NLPProcessor
from core.embeddings     import EmbeddingEngine
from core.ai_response    import AIResponseEngine
from core.memory         import ConversationMemory
from core.intent_detector import IntentDetector
from core.logger         import ChatLogger
from utils.validators    import validate_message, sanitize_input

# ── App Setup ────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "spcoet-secret-2024")
CORS(app)

# ── Rate Limiting ─────────────────────────────────────────────
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "60 per hour"],
    storage_uri="memory://"
)

# ── Initialize Core Engines ───────────────────────────────────
print("\n🚀 Initializing SPCOET Smart AI Chatbot...")
config       = Config()
nlp          = NLPProcessor()
embedder     = EmbeddingEngine(config.KNOWLEDGE_BASE_PATH)
ai_engine    = AIResponseEngine(config)
memory       = ConversationMemory(max_history=5)
intent_det   = IntentDetector()
chat_logger  = ChatLogger(config.LOG_DIR)
print("✅ All engines ready!\n")


# ════════════════════════════════════════════════════════════
#   MAIN CHAT ENDPOINT
# ════════════════════════════════════════════════════════════
@app.route("/chat", methods=["POST"])
@limiter.limit("30 per minute")
def chat():
    """
    Main chat endpoint.
    Receives: { message, session_id, lang }
    Returns:  { reply, intent, confidence, suggestions, sources }
    """
    start_time = time.time()
    data = request.get_json(silent=True) or {}

    # ── 1. Input validation & sanitization ────────────────────
    raw_message = data.get("message", "").strip()
    if not validate_message(raw_message):
        return jsonify({"error": "Invalid or empty message"}), 400

    message    = sanitize_input(raw_message)
    session_id = data.get("session_id") or str(uuid.uuid4())
    lang       = data.get("lang", "en")

    # ── 2. NLP Preprocessing ──────────────────────────────────
    nlp_result = nlp.process(message)
    # nlp_result = { cleaned_text, tokens, keywords, entities, normalized }

    # ── 3. Intent Detection ────────────────────────────────────
    intent_result = intent_det.detect(nlp_result["normalized"])
    # intent_result = { intent, confidence, sub_intent }

    # ── 4. Semantic Search in Knowledge Base ──────────────────
    search_results = embedder.search(
        query=nlp_result["normalized"],
        top_k=4
    )
    # search_results = [{ content, score, section, source }]

    # ── 5. Build Context from Search Results ──────────────────
    context_chunks = []
    sources        = []
    for r in search_results:
        if r["score"] > config.SIMILARITY_THRESHOLD:
            context_chunks.append(r["content"])
            sources.append({"section": r["section"], "score": round(r["score"], 3)})

    context_text = "\n\n".join(context_chunks) if context_chunks else ""

    # ── 6. Conversation History ────────────────────────────────
    chat_history = memory.get_history(session_id)

    # ── 7. Generate AI Response ────────────────────────────────
    ai_result = ai_engine.generate(
        user_message = message,
        context      = context_text,
        history      = chat_history,
        intent       = intent_result["intent"],
        lang         = lang
    )
    # ai_result = { reply, used_context }

    # ── 8. Update Memory ──────────────────────────────────────
    memory.add_message(session_id, "user",      message)
    memory.add_message(session_id, "assistant", ai_result["reply"])

    # ── 9. Generate Suggestions ───────────────────────────────
    suggestions = intent_det.get_suggestions(intent_result["intent"])

    # ── 10. Calculate Response Time ───────────────────────────
    response_time_ms = round((time.time() - start_time) * 1000)

    # ── 11. Log the Conversation ──────────────────────────────
    chat_logger.log(
        session_id    = session_id,
        user_message  = message,
        bot_reply     = ai_result["reply"],
        intent        = intent_result["intent"],
        confidence    = intent_result["confidence"],
        response_time = response_time_ms,
        sources_found = len(sources)
    )

    # ── 12. Return Response ───────────────────────────────────
    return jsonify({
        "reply"         : ai_result["reply"],
        "session_id"    : session_id,
        "intent"        : intent_result["intent"],
        "sub_intent"    : intent_result.get("sub_intent", ""),
        "confidence"    : round(intent_result["confidence"], 2),
        "sources"       : sources[:2],                # top 2 sources
        "suggestions"   : suggestions,
        "used_context"  : ai_result["used_context"],
        "response_time" : response_time_ms,
        "timestamp"     : datetime.now().isoformat()
    })


# ════════════════════════════════════════════════════════════
#   ADMIN — UPDATE KNOWLEDGE BASE
# ════════════════════════════════════════════════════════════
@app.route("/admin", methods=["POST"])
def admin_update():
    """
    Admin endpoint to update or add knowledge base entries.
    Expects: { section, content, admin_key }
    """
    data = request.get_json(silent=True) or {}

    # Simple admin key check (use a strong key in .env for production)
    admin_key = data.get("admin_key", "")
    if admin_key != os.getenv("ADMIN_KEY", "spcoet-admin-2024"):
        return jsonify({"error": "Unauthorized"}), 401

    section = data.get("section", "").strip()
    content = data.get("content", "").strip()
    action  = data.get("action", "update")   # update | delete | add

    if not section:
        return jsonify({"error": "section is required"}), 400

    try:
        # Load current KB
        with open(config.KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)

        if action == "delete":
            kb.pop(section, None)
            msg = f"Section '{section}' deleted"
        elif action == "add" or action == "update":
            if not content:
                return jsonify({"error": "content required for add/update"}), 400
            # Append or update
            if section in kb and isinstance(kb[section], list):
                kb[section].append({"text": content, "added": datetime.now().isoformat()})
            else:
                kb[section] = {"text": content, "updated": datetime.now().isoformat()}
            msg = f"Section '{section}' {action}d successfully"
        else:
            return jsonify({"error": "Invalid action"}), 400

        # Save KB
        with open(config.KNOWLEDGE_BASE_PATH, "w", encoding="utf-8") as f:
            json.dump(kb, f, indent=2, ensure_ascii=False)

        # Re-build embeddings
        embedder.rebuild_index()

        chat_logger.log_admin(action=action, section=section)
        return jsonify({"success": True, "message": msg})

    except Exception as e:
        logging.error(f"Admin update error: {e}")
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════
#   DATA — RETRIEVE KNOWLEDGE BASE
# ════════════════════════════════════════════════════════════
@app.route("/data", methods=["GET"])
def get_data():
    """Return the knowledge base (public read access)."""
    try:
        section = request.args.get("section")   # ?section=courses
        with open(config.KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)
        if section:
            return jsonify({section: kb.get(section, {})})
        # Return section names only (no full text) for privacy
        summary = {k: (type(v).__name__) for k, v in kb.items()}
        return jsonify({"sections": list(kb.keys()), "summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════
#   ADMIN DASHBOARD STATS
# ════════════════════════════════════════════════════════════
@app.route("/admin/stats", methods=["GET"])
def admin_stats():
    """Return chatbot statistics for admin dashboard."""
    admin_key = request.headers.get("X-Admin-Key", "")
    if admin_key != os.getenv("ADMIN_KEY", "spcoet-admin-2024"):
        return jsonify({"error": "Unauthorized"}), 401

    try:
        stats = chat_logger.get_stats()
        kb_info = embedder.get_index_info()
        return jsonify({
            "stats"   : stats,
            "kb_info" : kb_info,
            "uptime"  : "running"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════
#   CHAT HISTORY FOR SESSION
# ════════════════════════════════════════════════════════════
@app.route("/history/<session_id>", methods=["GET"])
def get_history(session_id):
    history = memory.get_history(session_id)
    return jsonify({"history": history, "count": len(history)})


@app.route("/history/<session_id>", methods=["DELETE"])
def clear_history(session_id):
    memory.clear(session_id)
    return jsonify({"success": True})


# ════════════════════════════════════════════════════════════
#   PAGE ROUTES
# ════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin-panel")
def admin_panel():
    return render_template("admin.html")

@app.route("/health")
def health():
    return jsonify({
        "status"     : "healthy",
        "version"    : "1.0.0",
        "project"    : "SPCOET Smart AI Chatbot",
        "ai_ready"   : bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")),
        "kb_entries" : embedder.get_index_info().get("total_chunks", 0),
        "timestamp"  : datetime.now().isoformat()
    })


# ── Error Handlers ────────────────────────────────────────────
@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({"error": "Too many requests. Please wait a moment."}), 429

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"🌐 Server starting at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
