# Submission text, ready to copy

Replace the two placeholders before posting:

- `REPO_URL`  -> https://github.com/YOUR-USERNAME/ask-my-docs
- `VIDEO_URL` -> the public link to your 30s-3min demo

## Short description (2-3 sentences)

Required in both the social post and the entry form.

> Ask My Docs is a retrieval-augmented generation tool over my own PDF and markdown
> notes, built from scratch with no RAG framework - chunking, embedding, vector search,
> and citation validation are all hand-written. It answers only from the chunks it
> retrieves, cites the exact file and character range behind every claim, and refuses
> outright when the notes do not cover the question instead of inventing an answer. I
> built it in Kiro across three weekly specs, with 72 correctness properties driving
> property-based tests.

## Social post - X version

> Built Ask My Docs for #KiroUniversity: a RAG tool over my own notes with no framework
> - hand-written chunking, embedding, vector search, and citation validation.
>
> It cites the exact file and character range for every claim, and refuses when my notes
> do not cover the question rather than inventing an answer.
>
> Built in Kiro from three weekly specs: steering rules, hooks, scoped custom agents, an
> MCP server, and property-based tests that check chunks always reconstruct the original
> text exactly.
>
> Code: REPO_URL
> Demo: VIDEO_URL
>
> @kirodotdev #KiroUniversity #BuildWithKiro

## Social post - LinkedIn version

> I built Ask My Docs during the Kiro University Challenge: a retrieval-augmented
> generation tool over my own PDF and markdown notes, written from scratch without
> LangChain or any RAG framework. Chunking, embedding, vector search, and citation
> validation are all code I can explain line by line.
>
> Two things I care about in it. Every answer cites the exact source file and character
> range it came from, so a claim can be checked against the original note. And when the
> notes do not cover a question, it refuses - the closest chunk has to clear a relevance
> threshold before the model is called at all. An assistant that always answers is
> useless, because you cannot tell the grounded answers from the invented ones.
>
> I built it in Kiro using spec-driven development: three weekly specs with requirements,
> designs, and task lists, steering files pinning the rules every task follows, hooks
> automating the test runs, three custom agents with deliberately different permissions,
> an MCP server, and property-based tests covering 72 stated correctness properties.
>
> Code: REPO_URL
> Demo: VIDEO_URL
>
> @kiro #KiroUniversity #BuildWithKiro

## Entry form: how each lesson was incorporated

Paste this into the writeup field.

**Lesson 1, Specs.** Three weekly specs under `.kiro/specs/` - week1-embeddings-chunking,
week2-vector-db-retrieval, week3-generation-citations - each with requirements in EARS
form, a design document, and a task list. 48 requirements and 72 correctness properties
drove every implementation decision.

**Lesson 2, Steering.** Five files in `.kiro/steering/`. product, tech, structure, and
privacy-and-secrets apply always; testing is scoped to `tests/**` by fileMatch. They pin
the banned-dependency rule that keeps RAG frameworks out, and the frozen-module rule that
stops later weeks editing Week 1 code.

**Lesson 3, Hooks.** Six hooks in `.kiro/hooks/`: run the suite after a spec task, run it
on source save, a secret-and-privacy write guard, a frozen-module write guard, re-ingest
prompting when a note file is added, and a session-start readiness report.

**Lesson 4, Property-based testing.** 19 Hypothesis property tests at 200 generated
examples each, mapped to numbered properties in the specs. They cover what examples
cannot: chunks always reconstruct the source text exactly for any size and overlap, the
overlap invariant between consecutive chunks, cosine similarity bounds and symmetry,
store add atomicity, and top-K against an independently computed reference ranking.

**Lesson 5, Skills and powers.** Three skills in `.kiro/skills/` encoding the judgement
calls this project needs: tune-retrieval, verify-grounding, explain-pipeline. The same
three are packaged as a shareable Kiro power in `powers/ask-my-docs-rag/`.

**Lesson 6, MCP.** `.kiro/settings/mcp.json` registers a fetch server for reading vector
database and embedding provider documentation while building, plus a git server available
but disabled.

**Lesson 7, Custom agents.** Three agents in `.kiro/agents/` with deliberately different
permission envelopes. spec-implementer can read, write, and run a narrow shell allowlist
with destructive git commands denied. retrieval-tuner can read and run the measurement
scripts but cannot write, and ingest is gated behind a confirmation because embedding
costs money. grounding-auditor is read-only with all shell denied.

**Bonus Lesson 2, Package a power.** `powers/ask-my-docs-rag/` with a `plugin.json`
validated against the Agent Plugins 1.0.0 schema, bundling the three skills, an MCP server
config, and a retrieval reference.

## Final compliance pass

- [ ] Repo is **public** and owned by you
- [ ] Repo name `ask-my-docs`
- [ ] `.kiro/` folder committed - 31 files
- [x] First commit after 21 Sept 09:00 PT, none before
- [ ] GitHub account at least 3 months old
- [ ] GitHub account matches the social account you post from
- [ ] Not resident in an excluded territory
- [ ] Video is 30 seconds to 3 minutes and publicly accessible
- [ ] Social post has repo link, description, both hashtags, and the tag
- [ ] Entry form has repo link, video link, social post link, correct email, writeup
- [ ] **Stop committing** once submitted, until judging concludes
