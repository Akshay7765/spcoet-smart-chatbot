# core/ai_response.py
# ──────────────────────────────────────────────────────────────
#  AI Response Generation Engine
#  Supports: Anthropic Claude (primary) | OpenAI GPT (alternative)
#  Uses:     Retrieved KB context + conversation history + intent
#  Applies:  Prompt engineering for college-specific responses
# ──────────────────────────────────────────────────────────────

import os, logging
from config import Config

# ── Try importing Anthropic SDK ───────────────────────────────
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("anthropic SDK not installed.")

# ── Try importing OpenAI SDK ──────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("openai SDK not installed.")


class AIResponseEngine:
    """
    Generates natural language responses using AI (Claude or GPT).
    Uses retrieved context (from FAISS) + conversation history.
    Falls back to rule-based responses if no API key available.
    """

    # System prompt — sets the AI's persona and rules
    SYSTEM_PROMPT = """You are SPCOET Assistant, the official AI-powered enquiry chatbot for Sharadchandra Pawar College of Engineering and Technology (SPCOET), Someshwarnagar, Baramati, Pune, Maharashtra.

Your role is to help students, parents, and visitors with accurate information about the college.

RULES:
1. Answer ONLY based on the provided college knowledge base context (if available)
2. If context is provided, base your answer strictly on it
3. If no context is available, say "I don't have specific information about this. Please contact 9823141287 or visit secsomeshwar.ac.in"
4. Be warm, helpful, and student-friendly
5. Use simple, clear language
6. When answering about fees, always add: "Please verify the exact amount at the college office"
7. Respond in the SAME language the user writes in (English, Hindi, or Marathi)
8. Keep answers concise but complete
9. Never make up information
10. For urgent queries, always provide the contact number: 9823141287"""

    def __init__(self, config: Config):
        self.config   = config
        self.provider = config.AI_PROVIDER

        # Initialize chosen provider
        self.client = None
        if self.provider == "anthropic" and ANTHROPIC_AVAILABLE and config.ANTHROPIC_API_KEY:
            self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            self.provider = "anthropic"
            print("  ✅ AI Provider: Anthropic Claude")
        elif OPENAI_AVAILABLE and config.OPENAI_API_KEY:
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.provider = "openai"
            print("  ✅ AI Provider: OpenAI GPT")
        else:
            print("  ⚠️  No AI API key — using rule-based fallback")
            self.provider = "fallback"

    def generate(self,
                 user_message: str,
                 context:      str,
                 history:      list,
                 intent:       str,
                 lang:         str = "en") -> dict:
        """
        Generate AI response.
        Returns: { reply: str, used_context: bool }
        """
        used_context = bool(context and context.strip())

        try:
            if self.provider == "anthropic":
                reply = self._call_anthropic(user_message, context, history, intent, lang)
            elif self.provider == "openai":
                reply = self._call_openai(user_message, context, history, intent, lang)
            else:
                reply = self._rule_based_fallback(user_message, context, intent)

            return {"reply": reply, "used_context": used_context}

        except Exception as e:
            logging.error(f"AI generation error: {e}")
            # Graceful fallback
            fallback = self._rule_based_fallback(user_message, context, intent)
            return {"reply": fallback, "used_context": used_context}

    # ── Anthropic Claude ──────────────────────────────────────
    def _call_anthropic(self, message, context, history, intent, lang) -> str:
        # Build full prompt
        user_content = self._build_user_prompt(message, context, intent, lang)

        # Convert history to Anthropic message format
        messages = []
        for h in history:
            role = "user" if h["role"] == "user" else "assistant"
            messages.append({"role": role, "content": h["content"]})

        # Add current message
        messages.append({"role": "user", "content": user_content})

        response = self.client.messages.create(
            model      = self.config.AI_MODEL_ANTHROPIC,
            max_tokens = 1024,
            system     = self.SYSTEM_PROMPT,
            messages   = messages
        )
        return response.content[0].text

    # ── OpenAI GPT ────────────────────────────────────────────
    def _call_openai(self, message, context, history, intent, lang) -> str:
        user_content = self._build_user_prompt(message, context, intent, lang)

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model      = self.config.AI_MODEL_OPENAI,
            max_tokens = 1024,
            messages   = messages
        )
        return response.choices[0].message.content

    # ── Prompt Building ───────────────────────────────────────
    def _build_user_prompt(self, message, context, intent, lang) -> str:
        """
        Build the final user prompt with context and intent injected.
        This is the core of prompt engineering.
        """
        parts = []

        # Language instruction
        lang_map = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
        lang_name = lang_map.get(lang, "English")
        if lang != "en":
            parts.append(f"[IMPORTANT: Please respond in {lang_name}]")

        # Intent hint for better responses
        if intent and intent != "general":
            parts.append(f"[User intent: {intent}]")

        # College KB context (most important!)
        if context and context.strip():
            parts.append(f"""COLLEGE KNOWLEDGE BASE (use this to answer):
---
{context.strip()}
---
Answer the question STRICTLY based on the above college information.""")
        else:
            parts.append("(No specific college data found for this query. Provide general guidance and direct to contact.)")

        # The actual user question
        parts.append(f"\nStudent's question: {message}")

        return "\n\n".join(parts)

    # ── Rule-Based Fallback ───────────────────────────────────
    def _rule_based_fallback(self, message, context, intent) -> str:
        """
        When no AI API is available, return a helpful rule-based answer.
        Uses the retrieved context directly.
        """
        if context and context.strip():
            return (
                f"Based on our college information:\n\n"
                f"{context[:800]}\n\n"
                f"For more details, contact us at **9823141287** or visit **secsomeshwar.ac.in**"
            )

        # Intent-based fallback
        fallbacks = {
            "admission" : "For admission enquiries, please call **9823141287** or visit secsomeshwar.ac.in/courses-offered. Admissions are through MHT-CET/JEE (CAP process) for BE and MAH-MBA CET for MBA.",
            "fees"      : "For exact fee details, please visit secsomeshwar.ac.in/fee-structure or call **9823141287**. Fees vary by program and category.",
            "courses"   : "SPCOET offers BE in Computer, Mechanical, Civil, Electrical Engineering and MBA. Visit secsomeshwar.ac.in/courses-offered for full details.",
            "hostel"    : "Hostel facilities are available for both boys and girls. For availability and fee details, call **9823141287**.",
            "placement" : "SPCOET has an active T&P Cell with good placement records. For details visit secsomeshwar.ac.in/placements or call **9823141287**.",
            "contact"   : "📞 Phone: **9823141287**\n🌐 Website: secsomeshwar.ac.in\n📍 Someshwarnagar, Baramati, Pune, Maharashtra",
        }

        for key, response in fallbacks.items():
            if key in message.lower() or key == intent:
                return response

        return (
            "I'm here to help with SPCOET enquiries! Please ask me about:\n"
            "• Courses & Admission\n• Fees & Scholarships\n• Hostel & Facilities\n"
            "• Placements\n• Contact & Location\n\n"
            "📞 For direct help: **9823141287**\n🌐 **secsomeshwar.ac.in**"
        )
