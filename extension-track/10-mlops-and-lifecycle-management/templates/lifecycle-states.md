# Lifecycle States

Use these states consistently in manifests, reports, and artifact folders.

## State Definitions

| State | Purpose | Required Evidence |
|---|---|---|
| `draft_experiment` | Initial exploratory run | manifest + outputs |
| `reviewed_experiment` | Run reviewed but not promoted | manifest + evaluation + review notes |
| `promoted` | Approved reusable workflow version | source run link + gate decision + release notes |
| `retired` | No longer recommended | deprecation note + replacement reference |
| `archived` | Stored for traceability only | immutable manifest + storage location |

## Allowed Transitions

- `draft_experiment -> reviewed_experiment`
- `reviewed_experiment -> promoted`
- `promoted -> retired`
- `retired -> archived`

Disallow direct `draft_experiment -> promoted` without review evidence.

## Minimal Metadata Per State

- `draft_experiment`: run_id, owner, versions, paths
- `reviewed_experiment`: draft fields + evaluation summary + reviewer
- `promoted`: reviewed fields + promoted_version_id + intended_use + known_limitations
- `retired`: promoted fields + retirement_reason + replacement_version
- `archived`: retired fields + archive_path + archive_date
