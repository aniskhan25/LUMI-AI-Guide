# Promotion Checklist

Use this checklist before promoting any workflow artifact from `draft_experiment` or `reviewed_experiment` into `promoted`.

## Required Inputs

- source run ID exists and is immutable
- run manifest exists and validates
- dataset/config/model/container references are complete
- evaluation summary is attached to the run

## Promotion Gate

A version can be promoted only if all checks below are `yes`.

| Check | Yes/No | Notes |
|---|---|---|
| Evaluation attached and review date recorded |  |  |
| Quality gate passed for intended use |  |  |
| Known limitations documented |  |  |
| Owner and reviewer assigned |  |  |
| Output schema and paths verified |  |  |
| Rollback target identified |  |  |

## Promotion Record

Record these fields in the promoted artifact metadata:

- promoted_version_id
- source_run_id
- intended_use
- known_limitations
- reviewer
- review_date
- storage/share_path

## Retirement Rule

If a promoted version fails follow-up checks, mark it `retired` and point users to the replacement version.
