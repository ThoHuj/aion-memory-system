import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dev.mock_agent import MockAgent

PAST_TRACES_DIRECTORY = Path(__file__).resolve().parent / "past_traces"


class SlashCommandError(Exception):
    """Raised when a slash command cannot be executed."""


def parse_slash_command_line(line: str) -> tuple[str, str] | None:
    """Return (command_name, arguments) if line is a slash command, else None."""
    stripped = line.strip()
    if not stripped.startswith("/"):
        return None
    body = stripped[1:].strip()
    if not body:
        raise SlashCommandError("Empty slash command.")
    parts = body.split(maxsplit=1)
    command_name = parts[0].lower()
    arguments = parts[1].strip() if len(parts) > 1 else ""
    return command_name, arguments


def execute_slash_command(line: str, mock_agent: "MockAgent") -> str:
    """Run a slash command and return text to print to the user."""
    parsed = parse_slash_command_line(line)
    if parsed is None:
        raise SlashCommandError("Not a slash command.")
    command_name, arguments = parsed
    if command_name == "import":
        return _execute_import_command(arguments, mock_agent)
    raise SlashCommandError(f"Unknown command: /{command_name}")


def _execute_import_command(arguments: str, mock_agent: "MockAgent") -> str:
    if not arguments:
        raise SlashCommandError(
            "Usage: /import <filename>  or  /import -r"
        )
    if arguments == "-r":
        trace_path = _choose_random_trace_path()
    else:
        trace_path = _resolve_trace_path(arguments)
    payload = _load_trace_payload(trace_path)
    messages = _extract_dialogue_messages(payload)
    extraction_passes = mock_agent.import_conversation_history(messages)
    return (
        f"Imported {trace_path.name}: {len(messages)} dialogue messages "
        f"({_dialogue_character_count(messages)} characters), "
        f"{extraction_passes} memory extraction passes."
    )


def _choose_random_trace_path() -> Path:
    if not PAST_TRACES_DIRECTORY.is_dir():
        raise SlashCommandError(
            f"Past traces directory not found: {PAST_TRACES_DIRECTORY}"
        )
    trace_paths = sorted(PAST_TRACES_DIRECTORY.glob("*.json"))
    if not trace_paths:
        raise SlashCommandError("No trace files in past_traces.")
    return random.choice(trace_paths)


def _resolve_trace_path(filename: str) -> Path:
    if ".." in filename or "/" in filename or "\\" in filename:
        raise SlashCommandError("Import only accepts a file name, not a path.")
    basename = Path(filename).name
    if not PAST_TRACES_DIRECTORY.is_dir():
        raise SlashCommandError(
            f"Past traces directory not found: {PAST_TRACES_DIRECTORY}"
        )
    direct = PAST_TRACES_DIRECTORY / basename
    if direct.is_file():
        return direct
    with_json = PAST_TRACES_DIRECTORY / f"{basename}.json"
    if with_json.is_file():
        return with_json
    raise SlashCommandError(f"Trace file not found: {basename}")


def _load_trace_payload(trace_path: Path) -> dict[str, Any]:
    try:
        raw = trace_path.read_text(encoding="utf-8")
    except OSError as error:
        raise SlashCommandError(f"Could not read {trace_path.name}: {error}") from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SlashCommandError(
            f"Invalid JSON in {trace_path.name}: {error.msg}"
        ) from error
    if not isinstance(payload, dict):
        raise SlashCommandError(f"Trace file must be a JSON object: {trace_path.name}")
    return payload


def _extract_dialogue_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise SlashCommandError("Trace file is missing a 'messages' list.")
    dialogue_messages: list[dict[str, str]] = []
    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise SlashCommandError(f"Message at index {index} is not an object.")
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant"):
            raise SlashCommandError(
                f"Message at index {index} has unsupported role: {role!r}."
            )
        if not isinstance(content, str):
            raise SlashCommandError(
                f"Message at index {index} must have string content."
            )
        dialogue_messages.append({"role": role, "content": content})
    if not dialogue_messages:
        raise SlashCommandError("Trace file contains no messages.")
    return dialogue_messages


def _dialogue_character_count(messages: list[dict[str, str]]) -> int:
    return sum(len(message["content"]) for message in messages)
