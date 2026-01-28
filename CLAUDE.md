# Agent instructions (project rules)

This file is the **single source of truth** for project-specific agent instructions.

## Read-first artifacts (always)

1. `README.md` – user + developer overview
2. `docs/ai/repo-map.md` – auto-generated repo map (entrypoints/modules)
3. `docs/ai/notes/INDEX.md` – curated, AI-generated session/port notes (auto-indexed)
4. `CHANGELOG.md` – release history
5. `tests/README.md` – testing organization

## AI navigation artifacts (keep current)

This repo maintains two lightweight, auto-generated AI navigation artifacts:

- `docs/ai/repo-map.md`
- `docs/ai/symbol-index.json`

Regenerate them after structural changes:

```bash
uv run python scripts/gen_ai_index.py --include-tests
```

## AI notes (keep relevant)

High-signal notes live in `docs/ai/notes/` and include frontmatter with:

- `base_commit`: the git commit the note was generated against
- `scope_paths`: files/dirs the note claims to describe
- `stale_policy.max_commits_since_base`: default 30 (for this repo)

Update the notes index and staleness flags with:

```bash
uv run python scripts/ai_notes_hygiene.py
```

Policy:
- Do not delete notes automatically.
- Prefer marking notes as *stale* when their scope has changed since `base_commit`.

## Coding rules

- Keep changes minimal and consistent with existing patterns.
- Do not commit large data or generated binaries.
- Avoid editing CI/workflows unless explicitly requested.
