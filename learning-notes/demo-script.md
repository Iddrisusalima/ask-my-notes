# Demo video shot list

**Hard limit: 3 minutes.** Nothing past 3:00 is watched. Minimum 30 seconds.
Target 2:40, which leaves margin.

The video must do two things at once: show the project working, and show where each
lesson lives. The order below interleaves them so no time is spent on a tour with
nothing happening.

## Recording on Windows, no install

1. Open the window you want to record (Kiro, or a maximised terminal).
2. Press **Win + Alt + R** to start recording. A small timer appears.
3. Press **Win + Alt + R** again to stop.
4. The file lands in `C:\Users\hp\Videos\Captures` as MP4.

Press **Win + G** first if you want the capture widget, volume, or mic toggle.
Game Bar records one application window, not the desktop or File Explorer, so start
it with the terminal or Kiro focused.

Talk over it live, or record silent and add narration in Clipchamp, which ships with
Windows 11.

## Before you hit record

Run the ingest once so the collection already exists:

```powershell
python scripts/05_ingest.py
```

The first ingest takes about 50 seconds because it loads the embedding model and
embeds 20 chunks. That is too slow for a 3 minute video. Showing the *second* run is
both faster and a better story: it proves the ingest is incremental.

Have these ready in your terminal history, in order, so you are not typing during the
recording.

## Shot list

| Time | Shot | Say this | Lesson shown |
|---|---|---|---|
| 0:00-0:12 | `README.md` open | "Ask My Docs answers questions about my own notes, using only what it retrieves, with citations. Every RAG stage is hand-written - no LangChain, no framework hiding a step." | - |
| 0:12-0:32 | Expand `.kiro/` in the sidebar: `specs/`, `steering/`, `hooks/`, `agents/`, `skills/`, `settings/mcp.json` | "Three phase specs drove this: requirements, design, and tasks. Steering files pin the rules every task follows. Six hooks, three scoped agents, three skills, and an MCP fetch server." | 1, 2, 3, 5, 6, 7 |
| 0:32-0:45 | Open `.kiro/agents/retrieval-tuner.json` | "The agents have different permission envelopes. This one reads and measures but cannot write, and ingest is gated behind a confirmation because it costs money." | 7 |
| 0:45-1:05 | `python scripts/05_ingest.py` | "Re-running ingest embeds nothing. A SHA-256 hash per file means unchanged notes are skipped - zero embedder calls, four seconds instead of fifty." | - |
| 1:05-1:30 | `python scripts/06_query.py "What overlap ratio did I settle on?" --top-k 2` | "Retrieval returns the source file, the chunk index, the exact character range, and a cosine score. The top hit is the chunk that actually contains the answer." | - |
| 1:30-1:55 | `python scripts/09_ask.py "Why does chunk size affect retrieved context quality?" --top-k 2` | "The full pipeline: retrieve, build a numbered context block, generate, then validate every citation against what was actually supplied. Verified: yes." | - |
| 1:55-2:20 | `python scripts/09_ask.py "What is the capital city of Mongolia?"` | "Asked something my notes do not cover, it refuses. The best score was 0.06 against a threshold of 0.30, so it never even calls the model. That refusal is what makes the other answers trustworthy." | - |
| 2:20-2:40 | `python -m pytest -q` | "Nineteen property-based tests, two hundred generated cases each. They check things examples cannot: that chunks always reconstruct the original text exactly, and that the Chroma store agrees with my reference implementation." | 4 |

## If you are over time

Cut the agents shot at 0:32 and fold it into the `.kiro` tour. Never cut the refusal
shot - it is the single most persuasive twenty seconds in the video.

## Publishing

The video must be publicly accessible. Upload to YouTube as Public, or attach it
natively to the X or LinkedIn post. Then the post needs: the repo link, a 2 to 3
sentence description, `#KiroUniversity`, `#BuildWithKiro`, and a tag of `@kirodotdev`
on X or `@kiro` on LinkedIn.

---

# Narration, word for word

About 390 words, which lands near 2:40 at a normal speaking pace. Read it as written or
loosen it to sound like you - just keep the numbers, because the numbers are what make
it credible.

## 0:00-0:12 - README on screen

"This is Ask My Docs. It answers questions about my own notes using only what it
retrieves from them, and it cites the exact file and character range every answer came
from. Every stage is hand-written. No LangChain, no framework hiding a step."

## 0:12-0:32 - expanding the .kiro folder

"I built it with Kiro, spec-first. Three phase specs: requirements in EARS form, a
design, and a task list. Steering files pin the rules every task has to follow, like the
banned-dependency rule and which modules are frozen. Then six hooks, three custom
agents, three skills, and an MCP server config."

## 0:32-0:45 - retrieval-tuner.json open

"The agents have deliberately different permissions. This one can read my retrieval logs
and run the measurement scripts, but it cannot write code, and ingest is gated behind a
confirmation because embedding costs money."

## 0:45-1:05 - running the ingest

"Ingest is incremental. It keeps a SHA-256 hash per file, so re-running it embeds nothing
- zero embedder calls, about four seconds, where the first run took fifty. And when a
file does change, it deletes the old chunks before writing the new ones, so a file that
shrinks does not leave orphans behind still answering queries."

## 1:05-1:30 - running the query

"Here is retrieval on its own. I ask what overlap ratio I settled on, and it returns the
source file, the chunk index, the exact character range, and a cosine score. The top hit
is the chunk that literally contains the answer - a ten percent overlap ratio."

## 1:30-1:55 - running the ask script

"Now the whole pipeline. It retrieves, builds a numbered context block, composes the
answer from the highest-scoring retrieved sentences, then checks every citation marker
against what was actually supplied. Verified: yes. If the model had invented a citation number, the whole answer would be flagged
unverified rather than quietly dropping the bad marker."

## 1:55-2:20 - the refusal

"And this is the part I care about most. I ask something my notes do not cover. The best
match scored 0.06 against a threshold of 0.30, so it refuses - and it never calls the
model at all. An assistant that always answers is useless, because you cannot tell the
grounded answers from the invented ones."

## 2:20-2:40 - running the tests

"Nineteen property-based tests, two hundred generated cases each. These check things
examples cannot: that chunks always reconstruct the original text exactly whatever the
size and overlap, and that the Chroma store agrees with my in-memory reference
implementation to seven decimal places."

## On answering

Answers are extractive: composed from the retrieved sentences themselves, ranked against
the question by the same cosine similarity that ranked the chunks. Nothing is scripted and
nothing is mocked, so there is no caveat to state. Say what it does and why it is a
strength:

"The answer is composed from the retrieved sentences themselves, ranked by the same cosine
similarity that ranked the chunks. No API key, and it cannot hallucinate, because every
sentence is text from my notes."

The abstractive path is still implemented behind a --chat flag for anyone with an
OPENAI_API_KEY set, which is worth one sentence if you have time.
