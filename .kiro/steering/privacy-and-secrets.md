# Privacy and secrets

## Never commit

`.env`, personal note files under `sample-notes/` (only the folder README is tracked), `.chroma/`,
`logs/`, `reports/relevance-review*.csv`, and the answer log. These hold API keys or verbatim
personal note text.

## The API key

The key is read from the environment only. It never appears in a log record, an error message, a
traceback, a prompt, or a report. All console output goes through the `Reporter`, which substitutes
a fixed redaction marker, including for any substring of the key at least 8 characters long.

## Sample notes

The committed `sample-notes/` folder must hold small, non-sensitive placeholder notes so a mentor
can clone the repository and run the tool.

## Before any commit

Check the staged file list for `.env`, note files, database directories, and log files, and flag
anything that matches.

## Default provider

The default embedding provider is the local sentence-transformers model, so the paid remote path is
always an explicit choice.
