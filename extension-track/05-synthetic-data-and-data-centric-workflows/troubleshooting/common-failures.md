# Common Failures

## 1) Candidates not tied to measured weaknesses

Symptoms:

- generated examples look generic
- downstream weak-case metrics do not improve

Fix:

- start from a versioned weak-case set
- include `source_case_id` and `gap_label` in all candidates

## 2) Filtering is skipped or too weak

Symptoms:

- accepted set contains malformed or low-value examples

Fix:

- enforce schema, dedup, and required-term checks
- keep explicit `accepted/rejected` status with reasons

## 3) Provenance is lost during merge

Symptoms:

- synthetic records cannot be traced back to source cases

Fix:

- preserve `synthetic_id` and `source_case_id` in augmented dataset
- version dataset outputs explicitly

## 4) Evaluation contamination

Symptoms:

- before/after comparison looks unrealistically strong

Fix:

- avoid copying evaluation references directly into accepted data
- keep evaluation set separate and unchanged

## 5) Volume-focused generation without impact checks

Symptoms:

- large candidate count but no measurable improvement

Fix:

- optimize for targeted gain, not raw volume
- require before/after comparison report in the main path

