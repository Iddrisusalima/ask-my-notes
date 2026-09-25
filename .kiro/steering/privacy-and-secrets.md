# Privacy and secrets

## Never commit

`.env`, personal note files under `my-notes/`, `.chroma/`,
`logs/`, `reports/relevance-review*.csv`, and the answer log. These hold API keys or verbatim
personal note text.

## The API key

The key is read from the environment only. It never appears in a log record, an error message, a
traceback, a prompt, or a report. All console output goes through the `Reporter`, which substitutes
a fixed redaction marker, including for any substring of the key at least 8 characters long.

## Two notes folders, and why

`sample-notes/` is **committed**. It holds small, non-sensitive notes so a mentor or reviewer can
clone the repository and run the tool immediately. A submission whose corpus is empty does not
function as described.

`my-notes/` is **git-ignored** and holds the learner's own notes. Point `ASKMYDOCS_NOTES_FOLDER` at
it for real use. Never place personal notes in `sample-notes/`.

## Before any commit

Check the staged file list for `.env`, note files, database directories, and log files, and flag
anything that matches.

## Default provider

The default embedding provider is the local sentence-transformers model, so the paid remote path is
always an explicit choice.
