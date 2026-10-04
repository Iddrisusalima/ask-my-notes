"""Ask a question, get a cited answer or an honest refusal.

The whole pipeline in one command. The outcome branch happens before any prompt
exists, so a refusal provably never reaches the chat endpoint.

Usage:
    python scripts/09_ask.py "why does chunk size matter?"
    python scripts/09_ask.py "..." --top-k 3
    python scripts/09_ask.py "..." --chat      # abstractive, needs OPENAI_API_KEY
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from askmydocs.config import (
    load_configuration,
    load_retrieval_settings,
    load_store_settings,
)
from askmydocs.errors import AskMyDocsError
from askmydocs.generation.citations import CitationValidator, build_source_list
from askmydocs.generation.generator import (
    AnswerGenerator,
    ChatCompletion,
    FakeChatModel,
    OpenAIChatClient,
)
from askmydocs.generation.extractive import build_extractive_answer
from askmydocs.generation.presenter import render_answer, render_refusal
from askmydocs.generation.prompting import DEFAULT_CONTEXT_BUDGET, PromptBuilder
from askmydocs.reporting import Reporter
from askmydocs.retrieval.logging import RetrievalLogWriter
from askmydocs.retrieval.retriever import Retriever
from askmydocs.stores.chroma_store import ChromaStore

DRY_RUN_ANSWER = (
    "A chunk is embedded as a single unit, so a chunk covering several topics "
    "averages them and matches no question strongly [1]. Overlap keeps a fact "
    "that straddles a boundary intact in at least one chunk [1]."
)


def parse_args(argv: list[str]) -> tuple[str, int | None, bool]:
    use_chat = "--chat" in argv
    argv = [a for a in argv if a != "--chat"]
    top_k: int | None = None
    if "--top-k" in argv:
        at = argv.index("--top-k")
        top_k = int(argv[at + 1])
        del argv[at : at + 2]
    return " ".join(argv).strip(), top_k, use_chat


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    question, requested_k, use_chat = parse_args(argv)

    if not question:
        print(
            'Usage: python scripts/09_ask.py "your question" [--top-k N] [--chat]',
            file=sys.stderr,
        )
        return 2

    try:
        configuration = load_configuration()
        store_settings = load_store_settings()
        retrieval_settings = load_retrieval_settings()
    except AskMyDocsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    reporter = Reporter(api_key=configuration.api_key)
    top_k = requested_k or retrieval_settings.top_k

    try:
        from askmydocs.embeddings.factory import build_embedder

        embedder = build_embedder(configuration)
        store = ChromaStore(
            store_settings.persist_directory,
            store_settings.collection_name,
            store_settings.distance_metric,
            model_name=configuration.model_name,
        )
        retriever = Retriever(
            store, embedder, retrieval_settings.relevance_threshold,
            configuration.max_input_length,
        )
        result = retriever.retrieve(question, top_k)

        RetrievalLogWriter(
            retrieval_settings.retrieval_log, configuration.api_key
        ).append(result, configuration.model_name, embedder.dimensionality)

        reporter.info(f'Question: "{question}"')

        # The refusal decision happens here, before a prompt exists.
        if not result.should_answer:
            for line in render_refusal(result):
                reporter.info(line)
            return 0

        builder = PromptBuilder(DEFAULT_CONTEXT_BUDGET)
        prompt = builder.build(result)

        if use_chat:
            # Abstractive: a chat model paraphrases the retrieved context.
            # Requires an API key, so it is opt-in rather than the default.
            client = OpenAIChatClient(
                configuration.api_key, configuration.request_timeout_seconds
            )
            generated = AnswerGenerator(
                client, configuration, "gpt-4o-mini"
            ).generate(prompt)
        else:
            # Extractive: the answer is composed from the retrieved sentences
            # themselves, ranked against the question by the same cosine
            # similarity used to rank the chunks. No API key, and grounded by
            # construction because every sentence is text from the notes.
            generated = build_extractive_answer(
                question, prompt, result, embedder
            )

        report = CitationValidator().validate(generated.text, prompt)
        sources = build_source_list(report, prompt)

        for line in render_answer(
            generated.text, report, sources, prompt, DEFAULT_CONTEXT_BUDGET
        ):
            reporter.info(line)
        reporter.info(
            f"Model: {generated.model}  Tokens: {generated.usage.total_tokens} "
            f"({generated.usage.prompt_tokens} prompt + "
            f"{generated.usage.completion_tokens} completion)"
        )
        return 0
    except AskMyDocsError as exc:
        reporter.error(str(exc))
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
