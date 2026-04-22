# core/memory.py
# ──────────────────────────────────────────────────────────────
#  Conversation Memory — Context-Aware Chat History
#  Stores last N message pairs per session in memory
#  Thread-safe with per-session locking
# ──────────────────────────────────────────────────────────────

import threading
from datetime import datetime, timedelta
from collections import defaultdict


class ConversationMemory:
    """
    In-memory conversation history manager.
    Stores last `max_history` user+assistant message pairs per session.
    Sessions auto-expire after 30 minutes of inactivity.
    """

    SESSION_TTL_MINUTES = 30   # Session expires after 30 mins inactivity

    def __init__(self, max_history: int = 5):
        self.max_history = max_history              # max message pairs (user+assistant)
        self._sessions   = defaultdict(list)        # { session_id: [messages] }
        self._last_seen  = {}                       # { session_id: datetime }
        self._lock       = threading.RLock()        # thread safety

    def add_message(self, session_id: str, role: str, content: str):
        """
        Add a message to the session history.
        role: "user" or "assistant"
        Trims to max_history pairs automatically.
        """
        with self._lock:
            self._sessions[session_id].append({
                "role"      : role,
                "content"   : content,
                "timestamp" : datetime.now().isoformat()
            })
            self._last_seen[session_id] = datetime.now()

            # Keep only last (max_history * 2) messages = max_history pairs
            max_msgs = self.max_history * 2
            if len(self._sessions[session_id]) > max_msgs:
                self._sessions[session_id] = self._sessions[session_id][-max_msgs:]

    def get_history(self, session_id: str) -> list:
        """
        Return conversation history for a session.
        Returns: [{ role, content, timestamp }, ...]
        """
        with self._lock:
            self._last_seen[session_id] = datetime.now()
            # Return without timestamp for AI context (cleaner)
            return [
                {"role": m["role"], "content": m["content"]}
                for m in self._sessions.get(session_id, [])
            ]

    def get_full_history(self, session_id: str) -> list:
        """Return full history including timestamps (for admin view)."""
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str):
        """Clear history for a session."""
        with self._lock:
            self._sessions.pop(session_id, None)
            self._last_seen.pop(session_id, None)

    def get_session_count(self) -> int:
        """Return number of active sessions."""
        with self._lock:
            self._cleanup_expired()
            return len(self._sessions)

    def get_stats(self) -> dict:
        with self._lock:
            self._cleanup_expired()
            total_messages = sum(len(v) for v in self._sessions.values())
            return {
                "active_sessions" : len(self._sessions),
                "total_messages"  : total_messages,
                "max_history"     : self.max_history
            }

    def _cleanup_expired(self):
        """Remove sessions that haven't been active for SESSION_TTL_MINUTES."""
        cutoff = datetime.now() - timedelta(minutes=self.SESSION_TTL_MINUTES)
        expired = [
            sid for sid, last in self._last_seen.items()
            if last < cutoff
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_seen.pop(sid, None)
