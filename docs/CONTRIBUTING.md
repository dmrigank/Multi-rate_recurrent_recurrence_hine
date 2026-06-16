# Contributing to MSR-HINE-2D

## One prompt per PR

Each pull request should correspond to one focused build prompt (see `BUILD_PROMPTS.md`).
Keep changes small and reviewable. Do not bundle multiple milestones into a single PR.

## Before committing

Run the test suite and confirm it passes:

```bash
conda activate 2d_hine
pytest
```

If you add new functionality, add a corresponding test in `tests/`.

## Coding conventions

- Python >= 3.10; type hints on all public functions and class constructors.
- One concern per file; follow the layout in `CLAUDE.md §10` exactly.
- No comments unless the *why* is non-obvious. No multi-line docstring blocks beyond
  one-line summaries for internal helpers.
- Do not break the invariants in `CLAUDE.md §4`. If you think a change requires violating
  one, stop and open a discussion first.

## Design questions

`docs/DESIGN.md` is the source of truth for the method. If `CLAUDE.md` and `DESIGN.md`
appear to conflict, do not silently reconcile — raise it for resolution before writing code.
