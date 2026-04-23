# core/nlp_processor.py — NLTK only (no spaCy, works on all servers)
import re, logging

try:
    import nltk
    from nltk.corpus   import stopwords
    from nltk.stem     import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    for pkg in ["stopwords","punkt","wordnet","punkt_tab","averaged_perceptron_tagger"]:
        try: nltk.download(pkg, quiet=True)
        except: pass
    STOP_WORDS = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    NLTK_READY = True
    print("  ✅ NLP: NLTK ready")
except ImportError:
    NLTK_READY = False
    STOP_WORDS = {"the","a","an","is","are","was","were","be","been","has","have","had",
                  "do","does","did","will","would","could","should","i","me","my","you",
                  "your","we","our","they","their","it","its","this","that","and","or",
                  "but","in","on","at","to","for","of","with","about","what","how",
                  "when","where","who","which","can","please","tell","know","want"}
    logging.warning("NLTK not installed — using basic stopwords")


class NLPProcessor:
    """Pure-Python NLP pipeline using NLTK. No C compilation required."""

    DOMAIN_KEYWORDS = {
        "spcoet","college","admission","fee","fees","course","courses","department",
        "hostel","placement","engineering","computer","mechanical","civil","electrical",
        "mba","eligibility","scholarship","contact","phone","address","website","cet",
        "jee","mht","naac","sppu","campus","library","lab","transport","bus","canteen",
        "sports","nss","semester","result","exam","cutoff","merit","document","form",
        "apply","deadline","date","faculty","principal","hod","professor","technothon",
        "somotsav","fest","event","notice","announcement","timetable","syllabus","grade"
    }

    def process(self, text: str) -> dict:
        if not text or not text.strip():
            return self._empty()
        cleaned = self._clean(text)
        return self._process_nltk(cleaned, text) if NLTK_READY else self._process_basic(cleaned, text)

    def _process_nltk(self, cleaned: str, original: str) -> dict:
        try:
            tokens = word_tokenize(cleaned.lower())
        except Exception:
            tokens = cleaned.lower().split()
        tokens = [t for t in tokens if re.match(r'^[a-z]+$', t) and len(t) > 1]
        keywords = []
        for tok in tokens:
            if tok in self.DOMAIN_KEYWORDS:
                keywords.append(tok)
            elif tok not in STOP_WORDS and len(tok) > 2:
                try: keywords.append(lemmatizer.lemmatize(tok))
                except: keywords.append(tok)
        keywords = list(set(keywords))
        entities = [{"text": w.strip(".,!?"), "label": "ENTITY"}
                    for i, w in enumerate(original.split())
                    if i > 0 and len(w) > 2 and w[0].isupper() and w.lower() not in STOP_WORDS][:8]
        normalized = " ".join(keywords) if keywords else cleaned
        return {"original": original, "cleaned": cleaned, "tokens": tokens[:50],
                "keywords": keywords, "entities": entities, "normalized": normalized, "method": "nltk"}

    def _process_basic(self, cleaned: str, original: str) -> dict:
        tokens   = cleaned.lower().split()
        keywords = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
        return {"original": original, "cleaned": cleaned, "tokens": tokens,
                "keywords": keywords, "entities": [],
                "normalized": " ".join(keywords) or cleaned, "method": "basic"}

    def _clean(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"[^\w\s\?\!\.\,\'\-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _empty(self) -> dict:
        return {"original":"","cleaned":"","tokens":[],"keywords":[],
                "entities":[],"normalized":"","method":"none"}

    def extract_language(self, text: str) -> str:
        devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        return "hi" if devanagari > len(text) * 0.3 else "en"
