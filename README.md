# 🤖 SPCOET Smart AI Chatbot
### Final Year Project — Advanced AI-Powered College Enquiry System

**Tech Stack:** Python · Flask · spaCy · NLTK · FAISS · Sentence Transformers · Claude AI · HTML/CSS/JS

---

## 📁 Project Structure

```
spcoet-smart-chatbot/
├── app.py                        ← Main Flask server (all REST endpoints)
├── config.py                     ← Centralized configuration
├── requirements.txt              ← All Python dependencies
├── .env.example                  ← API key template (rename to .env)
├── README.md                     ← This file
│
├── core/                         ← Core AI/NLP engines
│   ├── __init__.py
│   ├── nlp_processor.py          ← spaCy + NLTK preprocessing pipeline
│   ├── embeddings.py             ← Sentence Transformers + FAISS vector search
│   ├── ai_response.py            ← Claude / OpenAI response generation
│   ├── memory.py                 ← Conversation memory (last 5 turns)
│   ├── intent_detector.py        ← Intent detection + confidence scoring
│   └── logger.py                 ← Chat logging + statistics
│
├── data/                         ← Data files (auto-created)
│   ├── knowledge_base.json       ← SPCOET college knowledge base
│   ├── chunks.json               ← Auto-generated text chunks
│   ├── faiss_index.bin           ← Auto-generated FAISS vector index
│   └── logs/                     ← Chat logs and statistics
│
├── static/
│   ├── css/
│   │   ├── style.css             ← Main stylesheet (maroon + gold theme)
│   │   └── admin.css             ← Admin dashboard styles
│   └── js/
│       └── chat.js               ← Frontend chat engine (voice, TTS, i18n)
│
└── templates/
    ├── index.html                ← Main chat interface
    └── admin.html                ← Admin knowledge base dashboard
```

---

## ⚡ Quick Start — Run on Your Laptop

### Prerequisites
- **Python 3.9 or higher** — https://www.python.org/downloads/
- **pip** (comes with Python)
- **Git** (optional)

---

### Step 1 — Create Virtual Environment

**Windows:**
```cmd
cd spcoet-smart-chatbot
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
cd spcoet-smart-chatbot
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear in your terminal.

---

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: Flask, spaCy, NLTK, FAISS, Sentence Transformers, Anthropic SDK, and more.

> ⏳ First install takes 5–10 minutes (downloading ML models ~500MB)

---

### Step 3 — Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

---

### Step 4 — Configure API Key

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env        # Mac/Linux
   copy .env.example .env      # Windows
   ```

2. Open `.env` and add your Anthropic API key:
   ```env
   ANTHROPIC_API_KEY=sk-ant-api03-YOUR-ACTUAL-KEY-HERE
   ```

   **Get a free API key at:** https://console.anthropic.com
   (Sign up → API Keys → Create Key)

---

### Step 5 — Start the Server

```bash
python app.py
```

You should see:
```
🚀 Initializing SPCOET Smart AI Chatbot...
  📦 Loading embedding model: all-MiniLM-L6-v2
  ✅ Embedding model loaded
  ✅ AI Provider: Anthropic Claude
  🔨 Building embeddings index from knowledge base...
  ✅ Index built: 85 chunks, dim=384
✅ All engines ready!

🌐 Server starting at http://localhost:5000
```

---

### Step 6 — Open in Browser

| Page | URL |
|------|-----|
| 💬 Chatbot | http://localhost:5000 |
| ⚙️ Admin Panel | http://localhost:5000/admin-panel |
| 🟢 Health Check | http://localhost:5000/health |

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send message, get AI reply |
| `POST` | `/admin` | Update knowledge base |
| `GET` | `/data` | View knowledge base sections |
| `GET` | `/admin/stats` | Dashboard statistics |
| `GET` | `/history/<id>` | Get session chat history |
| `DELETE` | `/history/<id>` | Clear session history |
| `GET` | `/health` | Server health + AI status |

### Chat API Example (cURL):
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What courses does SPCOET offer?", "lang": "en"}'
```

### Chat API Example (JavaScript):
```javascript
const response = await fetch('/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "Tell me about admission process",
    session_id: "my-session-123",
    lang: "en"
  })
});
const data = await response.json();
console.log(data.reply);       // AI response
console.log(data.intent);      // detected intent
console.log(data.confidence);  // confidence score (0-1)
console.log(data.suggestions); // follow-up suggestions
```

### Admin Update Example:
```bash
curl -X POST http://localhost:5000/admin \
  -H "Content-Type: application/json" \
  -d '{
    "admin_key": "spcoet-admin-2024",
    "section": "courses",
    "content": "New course: Data Science B.E. with 60 seats",
    "action": "add"
  }'
```

---

## ✨ Features

| Feature | Technology |
|---------|-----------|
| 🤖 Full AI responses | Claude claude-haiku-4-5 via Anthropic API |
| 🔍 Semantic search | FAISS + Sentence Transformers (all-MiniLM-L6-v2) |
| 🧠 NLP preprocessing | spaCy (entities, tokens, lemmas) + NLTK |
| 🎯 Intent detection | Custom weighted keyword scoring (10 intents) |
| 💾 Context memory | In-memory conversation history (last 5 turns) |
| 📊 Confidence scores | Displayed live with color-coded bar |
| 💡 Smart suggestions | Intent-based follow-up question chips |
| 🎙️ Voice input | Web Speech API (Chrome/Edge) |
| 🔊 Voice output | SpeechSynthesis TTS API |
| 🌐 3 Languages | English, Hindi (हिंदी), Marathi (मराठी) |
| ⚙️ Admin dashboard | Live KB editor + stats + API tester |
| 📝 Chat logging | JSONL daily logs + statistics JSON |
| 🔒 Security | Input validation, rate limiting, .env keys |

---

## 🏗️ Architecture Flow

```
User Input
    │
    ▼
[1] Input Validation & Sanitization (utils/validators.py)
    │
    ▼
[2] NLP Processing (core/nlp_processor.py)
    │  spaCy: tokenize, lemmatize, extract entities
    │  NLTK: remove stopwords, normalize
    │
    ▼
[3] Intent Detection (core/intent_detector.py)
    │  Weighted keyword scoring → intent + confidence
    │
    ▼
[4] Semantic Search (core/embeddings.py)
    │  Query → Sentence Transformer embedding
    │  FAISS similarity search → top 4 KB chunks
    │
    ▼
[5] Context + History Assembly
    │  Relevant KB chunks + last 5 conversation turns
    │
    ▼
[6] AI Response Generation (core/ai_response.py)
    │  Prompt engineering → Claude API call
    │  Returns natural language answer
    │
    ▼
[7] Memory Update (core/memory.py)
    │  Store user+bot message in session history
    │
    ▼
[8] Logging (core/logger.py)
    │  Log to JSONL file + update stats
    │
    ▼
JSON Response → Frontend → User
```

---

## 🌐 Deploy on Render (Free Hosting)

1. Push code to GitHub (don't push `.env` or `data/logs/`)
2. Go to https://render.com → New Web Service
3. Connect your GitHub repo
4. Settings:
   - **Build Command:** `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
   - **Start Command:** `python app.py`
   - **Environment:** Python 3
5. Add Environment Variables:
   - `ANTHROPIC_API_KEY` = your key
   - `ADMIN_KEY` = your admin password
   - `SECRET_KEY` = random string
6. Deploy!

---

## 🚀 Deploy on Railway (Alternative)

1. Install Railway CLI: `npm install -g @railway/cli`
2. `railway login`
3. `railway init`
4. `railway up`
5. Set env vars in Railway dashboard

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `spaCy model not found` | Run `python -m spacy download en_core_web_sm` |
| `FAISS not installing` | Try `pip install faiss-cpu --no-cache-dir` |
| AI not responding | Check `.env` file has correct `ANTHROPIC_API_KEY` |
| Port already in use | Change `PORT=5001` in `.env` |
| `sentence-transformers` slow | Normal on first run — downloads ~80MB model |

---

## 👨‍💻 Tech Stack Summary

```
Backend   : Python 3.9+, Flask 3.0, Flask-CORS, Flask-Limiter
NLP       : spaCy 3.7 (en_core_web_sm), NLTK 3.9
Embeddings: sentence-transformers (all-MiniLM-L6-v2), numpy
Vector DB : FAISS (faiss-cpu)
AI LLM    : Anthropic Claude claude-haiku-4-5 (or OpenAI GPT-4o-mini)
Frontend  : HTML5, CSS3, Vanilla JavaScript
Voice     : Web Speech API (STT) + SpeechSynthesis API (TTS)
Storage   : JSON files (no database needed!)
```

---

## 📝 Project Info

**Project Name:** SPCOET Smart AI Chatbot  
**Type:** Final Year Engineering Project  
**College:** Sharadchandra Pawar College of Engineering and Technology  
**Location:** Someshwarnagar, Baramati, Pune, Maharashtra  
**Contact:** 9823141287 | secsomeshwar.ac.in
