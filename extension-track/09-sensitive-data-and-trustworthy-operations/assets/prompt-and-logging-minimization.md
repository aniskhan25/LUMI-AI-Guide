# Prompt and Logging Minimization Note

## Prompt Minimization Rules

- Include only context needed to answer the task.
- Remove direct personal identifiers unless required by objective.
- Prefer short evidence snippets over full raw records.
- Keep prompt templates deterministic and auditable.

## Logging Minimization Rules

- Log stable IDs, status, and timing by default.
- Avoid storing full raw prompts/responses in routine logs.
- Store raw text only for short-lived review samples when required.
- Separate debug artifacts from operational logs.

## Why This Helps

- Reduces exposure footprint
- Simplifies review and retention policies
- Improves debugging signal-to-noise in trust-critical workflows

