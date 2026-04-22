---
title: "RAG Scaffold Crosswalk for MIB Viewer"
date: 2026-04-22
base_commit: 3e5d60f
scope_paths:
  - CLAUDE.md
  - docs/ai/
  - scripts/
stale_policy.max_commits_since_base: 30
---

# RAG Scaffold Crosswalk for MIB Viewer

## Purpose

Document the current AI scaffold in this repo and map it to the adjacent RAG project so future work can extend the same pattern without guessing.

## What already exists here

- `CLAUDE.md` as project-level agent instructions.
- `docs/ai/repo-map.md` and `docs/ai/symbol-index.json` as generated navigation artifacts.
- `docs/ai/notes/` as the curated session-note area.
- `scripts/gen_ai_index.py` and `scripts/ai_notes_hygiene.py` as the current refresh tools.

## What the RAG project has that this repo now mirrors

- A canonical project note file with current understanding and open questions.
- A dedicated development script inventory.
- A clear read-first sequence for agents.
- A lightweight freshness loop for generated navigation artifacts and notes.

## Gaps this mapping makes explicit

- The repo is still lighter than the RAG scaffold in terms of living design notes.
- We do not yet have a formal attic/current split for development scripts because there are only two active scripts.
- The export workflow is still future work and should be documented only after the implementation stabilizes.

## Next scaffold steps

1. Keep `docs/ai/mib-notes.md` current as the project-level context note.
2. Keep `docs/ai/dev-scripts.md` current as scripts evolve.
3. Add task-specific notes under `docs/ai/notes/` when a feature or fix needs long-lived context.
4. Regenerate the AI navigation artifacts after structure changes.

