# Kiro University Challenge — exam submission checklist

Authoritative source: the Challenge Terms and Conditions at
<https://kiro.dev/2026/university/terms/>. This file records the requirements as they
apply to Ask My Docs, and doubles as a draft of the per-lesson writeup the entry form
asks for.

**Hard deadline: Monday 5 October 2026, 23:59 PT.** The entry form must be submitted by
then. This is separate from the mentor project deadline of Sunday 4 October 2026, 23:59.

## Eligibility

- [x] First GitHub commit on or after Mon 21 Sept 2026, 09:00 PT — commit `9b81ece`,
      2026-09-24 17:29 UTC. No earlier commits exist in this repository.
- [ ] GitHub account is at least 3 months old.
- [ ] GitHub account pairs with the same person's X or LinkedIn account used for the post.
- [ ] Entrant is 18 or older, and does not live in an excluded territory. The excluded
      list includes Argentina, Australia, Brazil, Hong Kong, Indonesia, Italy, Malaysia,
      Philippines, Thailand, Vietnam, Singapore, Russia, Cuba, Iran, North Korea, Syria,
      Belarus, Crimea, the DNR and LNR regions, and the United Arab Emirates.
- [ ] Worked individually. One entry per person.
- [x] Kiro used as the primary development tool.

## Deliverables

- [ ] **Public GitHub repository** named `ask-my-docs`, owned by the entrant.
- [x] **`.kiro` folder committed**, showing the configuration for every lesson.
- [ ] **A working project.** Functional, not a static mockup: ingest runs, a question
      returns a cited answer, and a question with no answer in the notes is refused.
- [ ] **Demo video, 30 seconds to 3 minutes**, publicly accessible. Nothing past the
      3 minute mark is watched, so the pipeline explanation must fit inside it.
- [ ] **Public social post** on X or LinkedIn carrying `#KiroUniversity` and
      `#BuildWithKiro`, tagging `@kirodotdev` on X or `@kiro` on LinkedIn, and including
      the repository link, a 2 to 3 sentence description, and the video.
- [ ] **Entry form** at <https://kiro.dev/2026/university> with the repository link, the
      video link, the live social post link, a correct email address, and the per-lesson
      writeup below.

## After submitting

- [ ] **No commits** between 5 October 23:59 PT and the end of judging, 19 October 23:59
      PT, or the arrival of the award email, whichever comes first. Committing in that
      window is grounds for disqualification.

## Per-lesson writeup (draft for the entry form)

| Lesson | Credits | Where it lives | What it does here |
|---|---|---|---|
| 1 — Specs | 250 | `.kiro/specs/week1-embeddings-chunking/`, `week2-vector-db-retrieval/`, `week3-generation-citations/` | Three weekly specs, each with requirements in EARS form, a design, and a task list. 48 requirements and 72 correctness properties drove the build. |
| 2 — Steering | 250 | `.kiro/steering/` | Five files: product, tech, structure, privacy-and-secrets always on, plus testing scoped to `tests/**`. They pin the banned-dependency rule and the frozen-module rule that later weeks must respect. |
| 3 — Hooks | 250 | `.kiro/hooks/` | Six hooks: tests after a spec task, tests on source save, a secret and privacy write guard, a frozen-module write guard, re-ingest prompting when a note is added, and a session-start readiness report. |
| 4 — Property-based testing | 500 | `tests/test_*_properties.py` | Every correctness property in each design maps to a Hypothesis test at 200 examples. Chunk reconstruction, the overlap invariant, cosine bounds and symmetry, store atomicity, and top-K against an independent reference ranking. |
| 5 — Powers and skills | 500 | `.kiro/skills/`, `powers/ask-my-docs-rag/` | Three skills — tune-retrieval, verify-grounding, explain-pipeline — encoding the judgement calls the project needs. |
| 6 — MCP | 1,000 | `.kiro/settings/mcp.json` | A `fetch` server for reading vector database and provider documentation, and a `git` server available but disabled. |
| 7 — Custom agents | 1,000 | `.kiro/agents/` | Three agents with deliberately different permission envelopes: `spec-implementer` (read, write, narrow shell), `retrieval-tuner` (read plus measurement scripts, ingest gated behind ask), `grounding-auditor` (read only). |
| Bonus 2 — Package a power | 250 | `powers/ask-my-docs-rag/` | A schema-validated `plugin.json` against `agent-plugins.org/schemas/1.0.0`, bundling the three skills, an MCP server, and a retrieval reference. |
| Bonus 1 — Kiro Web and cloud | 250 | — | Requires a paid plan. Pending. |
