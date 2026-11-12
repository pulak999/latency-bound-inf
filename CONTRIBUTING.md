# Contributing

## Issue or pull request?

- Open an issue when scope, ownership, or the desired outcome needs discussion.
- Open a pull request directly for a small, clear change.
- Use a draft pull request when code exists but you want early feedback.
- Link planned work with `Closes #<issue-number>`.

## Before review

```bash
uv sync --locked
uv run pytest -q
uv run ruff check
uv run ruff format --check
```

Keep each pull request focused. Its description should say:

1. what changed;
2. why;
3. how it was verified;
4. whether timing, scheduling, records, or reported metrics changed.

## Merging

Changes to `main` should:

- use a pull request;
- pass required checks;
- receive one approval, including code-owner review;
- resolve review conversations;
- be squash-merged.

The repository owner may push or merge to `main` directly. Other people and agents use a pull request. GitHub enforces the pull-request rules for everyone except the owner.
