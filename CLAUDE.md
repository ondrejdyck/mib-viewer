# Agent instructions (project rules)

This file is the **single source of truth** for project-specific agent instructions.

## Read-first artifacts

Read these first when orienting on a non-trivial task:

1. `README.md` - user + developer overview
2. `docs/ai/mib-notes.md` - current understanding, decisions, and open questions
3. `docs/ai/repo-map.md` - auto-generated repo map
4. `docs/ai/notes/INDEX.md` - curated AI note index with freshness flags
5. `docs/ai/dev-scripts.md` - current vs attic script inventory
6. `CHANGELOG.md` - release history
7. `tests/README.md` - testing organization

## AI navigation artifacts

This repo keeps the same lightweight scaffold used in the adjacent RAG project:

- `docs/ai/mib-notes.md`
- `docs/ai/repo-map.md`
- `docs/ai/symbol-index.json`
- `docs/ai/dev-scripts.md`
- `docs/ai/notes/`

Regenerate the navigation artifacts after structural changes:

```bash
uv run python scripts/gen_ai_index.py --include-tests
uv run python scripts/ai_notes_hygiene.py
```

## AI notes

High-signal notes live in `docs/ai/notes/` and include frontmatter with:

- `base_commit`: the git commit the note was generated against
- `scope_paths`: files/dirs the note claims to describe
- `stale_policy.max_commits_since_base`: default 30 for this repo

Policy:

- Do not delete notes automatically.
- Prefer marking notes as stale when their scope has changed since `base_commit`.

## Code and docs hygiene

- Keep changes minimal and consistent with existing patterns.
- Do not commit large data or generated binaries.
- Avoid editing CI/workflows unless explicitly requested.
- Keep `docs/ai/mib-notes.md` short and decision-oriented.
