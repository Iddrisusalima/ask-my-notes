---
inclusion: fileMatch
fileMatchPattern: 'tests/**'
---

# Testing conventions

- `pytest` plus `hypothesis`. One command runs everything: `python -m pytest -q`. The whole-suite
  budget is 300 seconds.
- Hypothesis profiles: `pure` (200 examples), `filesystem` (100 examples, 3 s deadline), `chroma`
  (100 examples, no deadline).
- Every correctness property in a spec design gets its own property test running at least 100
  generated examples. A failure must report the seed and the shrunk input so the run repeats.
- Name each property test after the design's property number, and carry a comment naming the
  acceptance criteria it validates.
- Embeddings in tests come from `FakeEmbedder`: deterministic, SHA-256 seeded, fixed
  dimensionality. Never call a remote provider from a test.
- No network. A session autouse fixture patches `socket`, and Chroma telemetry is disabled. A test
  that needs a socket is a test that is wrong.
- Each Chroma test case gets its own temporary persist directory under `tmp_path` and removes it
  afterwards.
- Repair degenerate generated inputs by rescaling rather than filtering them with `assume()`, so
  the example count stays honest.
- The store conformance suite runs the same test code once per store implementation.
