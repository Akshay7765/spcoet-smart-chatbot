# core/nlp_processor.py
# ──────────────────────────────────────────────────────────────
#  NLP Preprocessing Pipeline
#  Uses: spaCy for entity/token extraction + NLTK for stopwords
#  Steps: clean → tokenize → remove stopwords → lemmatize → keyword extract
# ──────────────────────────────────────────────────────────────

import re
import logging

# ── spaCy ─────────────────────────────────────────────────────
try:
    import spacy
    # Try loading medium English model, fall back to small
    try:
        nlp_model = spacy.load("en_core_web_md")
    except OSError:
        try:
            nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            nlp_model = None
            logging.warning("spaCy model not found. Using fallback NLP.")
except ImportError:
    nlp_model = None
    logging.warning("spaCy not installed.")

# ── NLTK ──────────────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus   import stopwords
    from nltk.stem     import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    # Download required NLTK data (only first time)
    for pkg in ["stopwords", "punkt", "wordnet", "averaged_perceptron_tagger", "punkt_tab"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    STOP_WORDS  = set(stopwords.words("english"))
    lemmatizer  = WordNetLemmatizer()
    NLTK_READY  = True
except ImportError:
    NLTK_READY = False
    STOP_WORDS = {"the","a","an","is","are","was","were","be","been","has","have","had","do","does","did"}
    logging.warning("NLTK not installed. Using basic stopwords.")


class NLPProcessor:
    """
    Full NLP preprocessing pipeline.
    Input  : raw user text
    Output : dict with cleaned, tokens, keywords, entities, normalized
    """

    # Domain-specific important words (never remove these)
    DOMAIN_KEYWORDS = {
        "spcoet","college","admission","fees","fee","course","courses","department",
        "hostel","placement","placements","faculty","engineering","computer","mechanical",
        "civil","electrical","mba","eligibility","scholarship","scholarship","contact",
        "phone","address","website","exam","cet","jee","mht","naac","sppu",
        "campus","library","lab","transport","bus","canteen","sports","nss",
        "timetable","semester","syllabus","result","exam","marks","certificate",
        "cutoff","merit","document","form","apply","apply","deadline","date"
    }

    def process(self, text: str) -> dict:
        """
        Main processing method. Returns enriched NLP result dict.
        """
        if not text or not text.strip():
            return self._empty_result()

        # Step 1: Basic cleaning
        cleaned = self._clean(text)

        # Step 2: Use spaCy if available (better)
        if nlp_model:
            return self._process_spacy(cleaned, text)
        elif NLTK_READY:
            return self._process_nltk(cleaned, text)
        else:
            return self._process_basic(cleaned, text)

    # ── spaCy Pipeline ────────────────────────────────────────
    def _process_spacy(self, cleaned: str, original: str) -> dict:
        doc = nlp_model(cleaned)

        # Tokens (no punctuation)
        tokens = [t.text.lower() for t in doc if not t.is_punct and not t.is_space]

        # Keywords = nouns, proper nouns, verbs (not stop words)
        keywords = list(set(
            t.lemma_.lower()
            for t in doc
            if not t.is_stop
            and not t.is_punct
            and not t.is_space
            and len(t.text) > 2
            and (t.pos_ in ("NOUN","PROPN","VERB","ADJ") or t.lemma_.lower() in self.DOMAIN_KEYWORDS)
        ))

        # Named entities
        entities = [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
        ]

        # Normalized: lemmatized, no stopwords
        normalized_tokens = [
            t.lemma_.lower()
            for t in doc
            if not t.is_stop and not t.is_punct and not t.is_space and len(t.text) > 1
        ]
        normalized = " ".join(normalized_tokens) if normalized_tokens else cleaned

        return {
            "original"   : original,
            "cleaned"    : cleaned,
            "tokens"     : tokens,
            "keywords"   : keywords,
            "entities"   : entities,
            "normalized" : normalized,
            "method"     : "spacy"
        }

    # ── NLTK Pipeline ─────────────────────────────────────────
    def _process_nltk(self, cleaned: str, original: str) -> dict:
        tokens = word_tokenize(cleaned.lower())
        tokens = [t for t in tokens if re.match(r'^[a-z]+$', t)]

        # Keywords: remove stopwords, lemmatize
        keywords = []
        for token in tokens:
            if token not in STOP_WORDS and len(token) > 2:
                lemma = lemmatizer.lemmatize(token)
                keywords.append(lemma)
        keywords = list(set(keywords))

        # Normalized text
        normalized = " ".join(keywords) if keywords else cleaned

        return {
            "original"   : original,
            "cleaned"    : cleaned,
            "tokens"     : tokens,
            "keywords"   : keywords,
            "entities"   : [],
            "normalized" : normalized,
            "method"     : "nltk"
        }

    # ── Basic Pipeline (fallback) ─────────────────────────────
    def _process_basic(self, cleaned: str, original: str) -> dict:
        tokens = cleaned.lower().split()
        keywords = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
        return {
            "original"   : original,
            "cleaned"    : cleaned,
            "tokens"     : tokens,
            "keywords"   : keywords,
            "entities"   : [],
            "normalized" : " ".join(keywords) or cleaned,
            "method"     : "basic"
        }

    # ── Helpers ───────────────────────────────────────────────
    def _clean(self, text: str) -> str:
        """Remove special characters, extra spaces, normalize unicode."""
        text = text.strip()
        # Keep letters, numbers, spaces, punctuation that matters
        text = re.sub(r"[^\w\s\?\!\.\,\'\-]", " ", text)
        text = re.sub(r"\s+", " ", text)          # collapse whitespace
        text = re.sub(r"\.{2,}", ".", text)        # multiple dots → one
        return text.strip()

    def _empty_result(self) -> dict:
        return {
            "original":"","cleaned":"","tokens":[],
            "keywords":[],"entities":[],"normalized":"","method":"none"
        }

    def extract_language(self, text: str) -> str:
        """
        Detect if message is primarily Hindi/Marathi or English.
        Simple heuristic: check for Devanagari unicode range.
        """
        devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        if devanagari > len(text) * 0.3:
            return "hi"   # Hindi/Marathi
        return "en"
