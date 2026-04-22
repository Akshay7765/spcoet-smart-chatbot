# utils/validators.py — Input validation and sanitization
import re, html

MAX_LEN = 500

def validate_message(text: str) -> bool:
    if not text or not isinstance(text, str): return False
    text = text.strip()
    if len(text) < 2 or len(text) > MAX_LEN: return False
    # Must have at least one alphanumeric char
    return bool(re.search(r'[a-zA-Z0-9\u0900-\u097F]', text))

def sanitize_input(text: str) -> str:
    text = html.escape(text.strip())
    text = re.sub(r'<[^>]+>', '', text)     # strip HTML tags
    text = re.sub(r'\s+', ' ', text)
    return text[:MAX_LEN]
