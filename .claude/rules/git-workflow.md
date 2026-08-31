# Git Workflow

- Do not call `git commit`.
- Staging and inspection commands (`git status`, `git diff`, `git log`) are fine.
- Prefix commits with `uv run` so pre-commit hooks use the correct environment, e.g.
  `uv run git commit -m "message"`.
