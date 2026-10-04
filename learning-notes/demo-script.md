# Demo video shot list

**Target 1:50.** Minimum 30 seconds, hard cap 3:00 - nothing past 3:00 is watched.

Five shots. Every one of the seven lessons appears, because the `.kiro` tour carries six
of them at once and the test run carries the seventh. Dropping either loses credits.

## Recording

OBS is installed. Select the `Display Capture` source, Alt+drag the right edge to crop
out the Kiro chat panel, then **Settings - Video - Base Canvas** about `1150x820` and
**Ctrl+F** to fit. Output `mp4`, 30 fps, 2500 Kbps.

Record a 10 second test first and open it from `C:\Users\hp\Videos` to confirm the frame
shows your work and not the chat panel.

## Before you hit record

Run all four commands once so they are in your shell history. Then on camera you press
the up arrow instead of typing, which is faster and cannot typo.

```powershell
python scripts/05_ingest.py
python scripts/09_ask.py "Why does chunk size affect retrieved context quality?" --top-k 2
python scripts/09_ask.py "What is the capital city of Mongolia?"
python -m pytest -q
```

Each script spawns a fresh Python process that imports PyTorch and loads the embedding
model, so there is a wait with nothing on screen. **Hit Enter and start talking
immediately** - the wait becomes your explanation time rather than dead air.

## Shot list

| Time | Shot | Runtime | Lessons |
|---|---|---|---|
| 0:00-0:10 | `README.md` open | - | - |
| 0:10-0:32 | Expand `.kiro/`: `specs/`, `steering/`, `hooks/`, `agents/`, `skills/`, `settings/mcp.json` | - | **1, 2, 3, 5, 6, 7** |
| 0:32-0:50 | `python scripts/05_ingest.py` | 3s | - |
| 0:50-1:23 | `python scripts/09_ask.py "Why does chunk size affect retrieved context quality?" --top-k 2` | 33s | - |
| 1:23-1:42 | `python scripts/09_ask.py "What is the capital city of Mongolia?"` | 19s | - |
| 1:42-1:52 | `python -m pytest -q` | 10s | **4** |

## Do not cut

The `.kiro` tour and the test run. Between them they are the only evidence for six of the
seven lessons plus the seventh. The refusal shot is the most persuasive twenty seconds in
the video - it is the moment that proves the tool reads your notes rather than reciting
the internet.

## Publishing

Public on YouTube, or attached natively to the X or LinkedIn post. The post needs the
repo link, a 2 to 3 sentence description, `#KiroUniversity`, `#BuildWithKiro`, and
`@kirodotdev` on X or `@kiro` on LinkedIn. All drafted in `submission-text.md`.

---

# Narration, word for word

About 250 words, which lands near 1:50 at a normal pace. Keep the numbers - they are what
make it credible.

## 0:00-0:10 - README on screen

"This is Ask My Docs. It answers questions about my own notes using only what it retrieves,
and cites the exact file and character range every answer came from. No LangChain, no
framework hiding a step."

## 0:10-0:32 - expanding the .kiro folder

"I built it with Kiro, spec-first. Three specs with requirements, designs and task lists.
Steering files pin the rules every task follows - the banned-dependency rule, and which
modules are frozen. Six hooks, three custom agents with different permissions, three
skills, and an MCP server."

## 0:32-0:50 - the ingest

"Ingest is incremental. A SHA-256 hash per file means re-running embeds nothing - zero
embedder calls, three seconds, where the first run took fifty."

## 0:50-1:23 - the cited answer

"Now the full pipeline. It retrieves, builds a numbered context block, then composes the
answer from the highest-scoring retrieved sentences. Verified: yes - and each sentence
carries the marker of the chunk it came from, so a miscitation is structurally impossible
rather than something I asked a model not to do. No API key either."

## 1:23-1:42 - the refusal

"And the part I care about most. I ask something my notes do not cover. The best match
scored 0.06 against a threshold of 0.30, so it refuses rather than answering from noise.
An assistant that always answers is useless, because you cannot tell the grounded answers
from the invented ones."

## 1:42-1:52 - the tests

"Nineteen property-based tests, two hundred generated cases each. They check what examples
cannot: that chunks always reconstruct the original text exactly, whatever the size and
overlap."
