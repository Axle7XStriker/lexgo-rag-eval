# Prompts

Every prompt used by the pipeline lives here, versioned. Prompts are source
code — treat them the same way.

## Layout

```
prompts/
  <role>/
    v1.md
    v2.md
    ...
```

- `role` = what the prompt does: `answer`, `judge`, `chunk_semantic`.
- Bump the version on any semantic change to the prompt. Never edit `v1.md`
  in place once it's been used in a captured eval run — that would break
  reproducibility.

## Front-matter

Each prompt starts with YAML front-matter capturing intent + provenance:

```yaml
---
name: answer
version: 1
model: claude-sonnet-4-6
purpose: >
  Given a question and retrieved chunks, produce a grounded answer with
  inline citations. Held constant across all 4 retrieval pipelines — the
  pipeline matrix isolates retrieval, not generation.
---
```

## Loading

Prompts are loaded by version at eval-run time and the version string is
captured in the run manifest alongside git SHA and config — a result row
is meaningless without knowing which prompt version produced it.
