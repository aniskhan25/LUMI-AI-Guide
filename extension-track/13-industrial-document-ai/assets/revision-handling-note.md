# Revision Handling Note

Use revision-aware ingestion rules:

- ingest new revisions as new records, never overwrite silently
- mark superseded revisions explicitly
- preserve retrieval links to exact revision used in answer evidence
- re-evaluate affected question slices when revisions change

This keeps answers traceable and auditable in technical domains.
