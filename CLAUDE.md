# Repository conventions

## No AI attribution in commits

This repository's commit history must not contain any AI/assistant attribution,
regardless of which tool or account is used to make the commit:

- No `Co-Authored-By: Claude ...`, `Co-Authored-By: Happy ...`, or similar
  trailers in any commit message.
- No "Generated with Claude Code" / "via Happy" / similar lines anywhere in a
  commit message.
- No `claude.ai` URLs (or links to any other AI tool's hosted output) in a
  commit message or in committed file content (README, docs, etc.) -- even a
  plain content link reveals AI involvement to anyone browsing the repo. If an
  AI-hosted artifact (e.g. a chart or dashboard) needs to be shared, save it as
  a plain file in the repo (e.g. under `results/figures/html/`) and reference
  that path instead.

This applies to every commit, by every contributor, regardless of which
account or machine makes it. If you are using Claude Code (or any other AI
coding assistant) in this repository: write commit messages as plain text with
no trailer, and do not let the tool's default commit-message template add one.

Real human co-authors (e.g. `Co-Authored-By: <a real person's name and
email>`) are fine and expected on a multi-person project -- the rule above is
about AI/assistant attribution specifically, not about limiting the repo to a
single human author.
