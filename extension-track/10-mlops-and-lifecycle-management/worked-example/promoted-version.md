# Worked Example: Promoted Version

## Promotion Candidate

Source run: `run-20260312-001`

Gate decision:

- evaluation attached: yes
- provenance complete: yes
- intended use documented: yes
- known limitations documented: yes

## Promoted Layout

```text
promoted/
  version-rag-2026-03-12-a/
    manifest.yaml
    config/
      rag-config-v1.4.yaml
    model-or-adapter/
      adapter.bin
    evaluation-summary/
      eval-summary.md
    release-notes.md
```

## Promotion Record

- promoted_version_id: `version-rag-2026-03-12-a`
- source_run_id: `run-20260312-001`
- intended_use: internal technical document assistant
- known_limitations: weak performance on long multi-hop questions
- review_date: `2026-03-12`
