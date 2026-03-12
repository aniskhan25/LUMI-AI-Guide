# Expected Schemas

## Weak cases (`weak_cases.jsonl`)

Required fields:

- `case_id`
- `input_text`
- `failure_type`
- `gap_label`
- `evidence_reference`
- `reference_answer`
- `required_terms` (array)

## Baseline dataset (`baseline_dataset.jsonl`)

Required fields:

- `record_id`
- `question`
- `answer`
- `gap_label`
- `source_flag` (`original` or `synthetic`)
- `dataset_version`

## Candidate synthetic records (`candidates.jsonl`)

Required fields:

- `synthetic_id`
- `source_case_id`
- `generated_input`
- `generated_target`
- `gap_label`
- `required_terms`
- `provenance`
- `filter_status` (`pending` before filtering)

## Filtered records (`filtered_candidates.jsonl`)

Required fields:

- all candidate fields
- `filter_status` (`accepted` or `rejected`)
- `filter_reasons` (array)

## Accepted synthetic records (`accepted_candidates.jsonl`)

Required fields:

- subset of filtered records where `filter_status=accepted`

## Augmented dataset (`augmented_dataset.jsonl`)

Required fields:

- baseline dataset fields
- provenance for synthetic records (`synthetic_id`, `source_case_id`)

