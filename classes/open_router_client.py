import json
import urllib.error
import urllib.request

CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient:
    """Send chat completion requests to OpenRouter."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Send a message history and return the model response text."""
        request_body = json.dumps(
            {
                "model": self._model,
                "messages": messages,
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

        return response_body["choices"][0]["message"]["content"]
