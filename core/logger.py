# core/logger.py — Chat logging + statistics

import os, json, logging
from datetime import datetime, date
from collections import defaultdict
from pathlib import Path


class ChatLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        # Configure standard Python logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(log_dir, "app.log")),
                logging.StreamHandler()
            ]
        )
        self.stats_file = os.path.join(log_dir, "stats.json")
        self._stats = self._load_stats()

    def log(self, session_id, user_message, bot_reply, intent, confidence, response_time, sources_found):
        today = date.today().isoformat()
        entry = {
            "ts": datetime.now().isoformat(), "session": session_id[:8],
            "user": user_message[:100], "intent": intent,
            "confidence": confidence, "response_ms": response_time,
            "sources": sources_found
        }
        # Append to daily log
        day_log = os.path.join(self.log_dir, f"chat_{today}.jsonl")
        with open(day_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Update stats
        self._stats.setdefault("total_messages", 0)
        self._stats["total_messages"] += 1
        self._stats.setdefault("intents", {})
        self._stats["intents"][intent] = self._stats["intents"].get(intent, 0) + 1
        self._stats.setdefault("avg_response_ms", [])
        self._stats["avg_response_ms"].append(response_time)
        if len(self._stats["avg_response_ms"]) > 1000:
            self._stats["avg_response_ms"] = self._stats["avg_response_ms"][-1000:]
        self._save_stats()
        logging.info(f"CHAT | intent={intent}({confidence:.0%}) | {response_time}ms | '{user_message[:50]}'")

    def log_admin(self, action, section):
        logging.info(f"ADMIN | action={action} | section={section}")

    def get_stats(self) -> dict:
        rt = self._stats.get("avg_response_ms", [])
        return {
            "total_messages": self._stats.get("total_messages", 0),
            "top_intents"   : sorted(self._stats.get("intents",{}).items(), key=lambda x:-x[1])[:5],
            "avg_response_ms": round(sum(rt)/len(rt)) if rt else 0,
        }

    def _load_stats(self) -> dict:
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file) as f: return json.load(f)
            except: pass
        return {}

    def _save_stats(self):
        with open(self.stats_file, "w") as f:
            json.dump(self._stats, f)
