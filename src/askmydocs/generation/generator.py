"""Chat transport and answer generation.

The chat client is isolated here so the generator -- which owns the retry
decision, the malformed-output check, and token accounting -- is testable
against a fake with no network and no key.

Malformed output is not retried. An empty completion is the terminal outcome of a
*successful* request, not a transport fault, so retrying would spend tokens
re-asking a question the model already answered with silence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from askmydocs.config import Configuration
from askmydocs.embeddings.retry import RetryPolicy
from askmydocs.errors import AskMyDocsError


class ChatCompletionFailedError(AskMyDocsError):
    """The chat endpoint failed and the retries are exhausted."""


class MalformedModelOutputError(AskMyDocsError):
    """The response carried no usable message content."""


class MissingApiKeyError(AskMyDocsError):
    """The chat provider needs a key that is not set."""


@dataclass(frozen=True)
class ChatCompletion:
    """One raw response. Content and token counts are both nullable."""

    content: str | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class TokenUsage:
    """Token counts, with the total derived so it cannot disagree."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class GeneratedAnswer:
    """The answer text and what it cost."""

    text: str
    usage: TokenUsage
    model: str


class ChatClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str, model: str) -> ChatCompletion: ...


class OpenAIChatClient:
    """One request per call against the chat completions endpoint."""

    def __init__(self, api_key: str | None, timeout_seconds: int) -> None:
        if not api_key:
            raise MissingApiKeyError(
                "OPENAI_API_KEY is not set, and the chat provider 'openai' requires "
                "it. Embeddings run locally without a key, but answer generation "
                "does not."
            )
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, timeout=self._timeout)
        return self._client

    def complete(self, system_prompt: str, user_prompt: str, model: str) -> ChatCompletion:
        response = self._get_client().chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choices = getattr(response, "choices", None) or []
        content = choices[0].message.content if choices else None
        usage = getattr(response, "usage", None)
        return ChatCompletion(
            content=content,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )


@dataclass
class FakeChatModel:
    """A scripted client. Replies are consumed in order; the last one repeats."""

    replies: list[ChatCompletion] = field(default_factory=list)
    calls: list[tuple[str, str, str]] = field(default_factory=list)

    def complete(self, system_prompt: str, user_prompt: str, model: str) -> ChatCompletion:
        self.calls.append((system_prompt, user_prompt, model))
        if not self.replies:
            return ChatCompletion(content="No reply scripted.", prompt_tokens=0, completion_tokens=0)
        return self.replies[0] if len(self.replies) == 1 else self.replies.pop(0)


def normalize_usage(completion: ChatCompletion) -> TokenUsage:
    """Zero-fill only when the provider reported nothing at all.

    A negative or non-integer count is coerced to 0 rather than trusted.
    """
    def clean(value: int | None) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return 0
        return value

    return TokenUsage(
        prompt_tokens=clean(completion.prompt_tokens),
        completion_tokens=clean(completion.completion_tokens),
    )


class AnswerGenerator:
    """Sends one prompt and returns one answer."""

    def __init__(
        self,
        client: ChatClient,
        configuration: Configuration,
        chat_model: str = "gpt-4o-mini",
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._client = client
        self._model = chat_model
        self._retry = retry_policy or RetryPolicy(configuration.max_retry_attempts)

    def generate(self, prompt) -> GeneratedAnswer:
        """Generate, retrying transient transport failures only."""
        try:
            completion = self._retry.run(
                lambda: self._client.complete(
                    prompt.system_prompt, prompt.user_prompt, self._model
                ),
                f"chat completion with {self._model}",
            )
        except AskMyDocsError:
            raise
        except Exception as exc:
            raise ChatCompletionFailedError(
                f"The chat model {self._model} failed: {type(exc).__name__}: {exc}"
            ) from exc

        # Checked outside the retry wrapper: a blank answer is a successful call.
        if completion.content is None or not completion.content.strip():
            raise MalformedModelOutputError(
                f"The chat model {self._model} returned no usable message content."
            )

        return GeneratedAnswer(
            text=completion.content,
            usage=normalize_usage(completion),
            model=self._model,
        )
