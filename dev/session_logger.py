import json
from datetime import UTC, datetime
from pathlib import Path

_DEV_DIRECTORY = Path(__file__).resolve().parent
_DEFAULT_LOG_DIR = _DEV_DIRECTORY / "data" / "logs"


class SessionLogger:
    """Append-only JSONL log of mock agent chat and memory activity (one file per session)."""

    def __init__(
        self,
        session_id: str,
        user_id: str,
        log_dir: Path | None = None,
    ) -> None:
        self._session_id = session_id
        self._user_id = user_id
        self._log_dir = log_dir if log_dir is not None else _DEFAULT_LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / f"{session_id}.jsonl"
        self._append(
            {
                "event": "session_start",
                "session_id": session_id,
                "user_id": user_id,
            }
        )

    @property
    def log_path(self) -> Path:
        """Path to this session's log file."""
        return self._log_path

    def log_retrieved_facts_for_chat(
        self,
        user_input: str,
        retrieved_fact_contents: list[str],
    ) -> None:
        """Log graph-retrieved facts injected before the chat completion."""
        self._append(
            {
                "event": "retrieved_facts_for_chat",
                "query": user_input,
                "retrieved_facts": retrieved_fact_contents,
            }
        )

    def log_chat_completion(
        self,
        messages: list[dict[str, str]],
        assistant_response: str,
    ) -> None:
        """Log the full message list sent to the chat model and the assistant reply."""
        self._append(
            {
                "event": "chat_completion",
                "messages": messages,
                "assistant_response": assistant_response,
            }
        )

    def log_conversation(self, messages: list[dict[str, str]]) -> None:
        """Log the persistent conversation transcript (excludes injected memory messages)."""
        self._append(
            {
                "event": "conversation",
                "messages": messages,
            }
        )

    def log_memory_add(
        self,
        turn_messages: list[dict[str, str]],
        add_result: dict[str, object],
    ) -> None:
        """Log a memory add turn and its result (entities, stored facts, invalidations)."""
        self._append(
            {
                "event": "memory_add",
                "turn_messages": turn_messages,
                "add_result": add_result,
            }
        )

    def _append(self, record: dict[str, object]) -> None:
        record["timestamp"] = datetime.now(UTC).isoformat()
        with self._log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
