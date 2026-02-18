import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import config


class SessionLogger:
    def __init__(self):
        short_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{timestamp}_{short_id}"
        self.session_dir = config.LOGS_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.session_dir / "session.jsonl"
        self._file = open(self.log_file, "a")
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._status_file = config.BASE_DIR / "agent_status.json"

    def log(self, event_type: str, data: dict):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **data,
        }
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def log_user_message(self, content: str):
        self.log("user_message", {"content": content})

    def log_llm_request(self, messages: list, tools: list):
        self.log("llm_request", {
            "message_count": len(messages),
            "tool_count": len(tools),
        })

    def log_llm_response(self, content: str, tool_calls: list, usage: dict):
        self.total_input_tokens += usage.get("input_tokens", 0)
        self.total_output_tokens += usage.get("output_tokens", 0)
        self.log("llm_response", {
            "content": content,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in tool_calls
            ],
            "usage": usage,
            "cumulative_usage": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
            },
        })

    def log_tool_exec(self, tool_name: str, arguments: dict, result: str):
        self.log("tool_exec", {
            "tool": tool_name,
            "arguments": arguments,
            "result": result[:2000],
        })

    def log_external_event(self, source: str, event_type: str, payload: dict):
        self.log("external_event", {
            "source": source,
            "event_type": event_type,
            "payload": payload,
        })

    def log_error(self, error: str, context: str = ""):
        self.log("error", {"error": error, "context": context})

    def update_status(self, state: str, current_tool: str | None = None):
        """Write agent_status.json with current state."""
        status = {
            "state": state,
            "session_id": self.session_id,
            "last_activity": datetime.now(timezone.utc).isoformat(),
            "current_tool": current_tool,
            "pid": os.getpid(),
        }
        try:
            self._status_file.write_text(json.dumps(status, indent=2))
        except Exception:
            pass

    def close(self):
        self._file.close()
        # Clean up status file on exit
        try:
            if self._status_file.exists():
                self._status_file.unlink()
        except Exception:
            pass
