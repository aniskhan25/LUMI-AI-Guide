# Lifecycle Checklist

- run manifest exists and validates
- dataset/config/model/container versions are recorded
- evaluation summary is attached to the run
- lifecycle state is explicit (`draft_experiment`, `reviewed_experiment`, `promoted`, `retired`, `archived`)
- promoted artifacts link to exactly one source run ID
- ownership and review date are documented
- sharing path (if used) is documented and scoped to promoted artifacts
