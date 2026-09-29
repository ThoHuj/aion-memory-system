import json
import urllib.error
import urllib.request
from typing import TypeVar

from pydantic import BaseModel, ValidationError

CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

T = TypeVar("T", bound=BaseModel)


def _strict_json_schema(model: type[BaseModel]) -> dict[str, object]:
    """Build a JSON schema compatible with OpenRouter strict structured output."""
    schema = model.model_json_schema()

    def apply_strict(node: object) -> None:
        if not isinstance(node, dict):
            return
        properties = node.get("properties")
        if node.get("type") == "object" and isinstance(properties, dict):
            node["additionalProperties"] = False
            node["required"] = list(properties.keys())
        for value in node.values():
            apply_strict(value)

    apply_strict(schema)
    return schema


class OpenRouterClient:
    """Send chat completion requests to OpenRouter."""

    def __init__(
        self, api_key: str, model: str, embedding_model: str
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._embedding_model = embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for the given texts in a single request."""
        if not texts:
            return []

        request_body = json.dumps(
            {
                "model": self._embedding_model,
                "input": texts,
            }
        ).encode()
        request = urllib.request.Request(
            EMBEDDINGS_URL,
            data=request_body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                response_body = json.loads(response.read())
        except urllib.error.HTTPError as error:
            error_body = error.read().decode()
            raise RuntimeError(
                f"OpenRouter embedding request failed ({error.code}): {error_body}"
            ) from error

        data = response_body["data"]
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("OpenRouter embedding response count mismatch")

        sorted_data = sorted(data, key=lambda item: item["index"])
        embeddings: list[list[float]] = []
        for item in sorted_data:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise TypeError(
                    "OpenRouter embedding response missing embedding vector"
                )
            embeddings.append(embedding)

        return embeddings

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Send a message history and return the model response text."""
        response_body = self._chat_completion({"messages": messages})
        return response_body["choices"][0]["message"]["content"]

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
    ) -> T:
        """Send a message history and return a structured, validated model response."""
        response_body = self._chat_completion(
            {
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": _strict_json_schema(response_model),
                    },
                },
                "provider": {
                    "require_parameters": True,
                },
            }
        )
        content = response_body["choices"][0]["message"]["content"]
        try:
            return response_model.model_validate_json(content)
        except ValidationError as error:
            raise RuntimeError(
                f"Failed to parse structured model response: {error}"
            ) from error

    def _chat_completion(self, request_payload: dict[str, object]) -> dict[str, object]:
        """Send a chat completion request and return the response body."""
        request_body = json.dumps(
            {
                "model": self._model,
                **request_payload,
            }
        ).encode()
        request = urllib.request.Request(
            CHAT_COMPLETIONS_URL,
            data=request_body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                response_body = json.loads(response.read())
        except urllib.error.HTTPError as error:
            error_body = error.read().decode()
            raise RuntimeError(
                f"OpenRouter request failed ({error.code}): {error_body}"
            ) from error

        return response_body
