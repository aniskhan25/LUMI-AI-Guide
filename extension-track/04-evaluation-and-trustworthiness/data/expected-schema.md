# Expected Schemas

## Evaluation set (`eval_set.jsonl`)

Required fields:

- `query_id`
- `question`
- `expected_doc_id`
- `category`

## Reference answers (`reference_answers.jsonl`)

Required fields:

- `query_id`
- `reference_answer`
- `required_terms` (array of strings)

## System outputs (`system_outputs.jsonl`)

Required fields:

- `query_id`
- `question`
- `answer`
- `evidence_chunk_ids` (array)
- `variant`

## Scored records (`scored_records.jsonl`)

Required fields:

- `query_id`
- `variant`
- `retrieval_hit` (0/1)
- `answer_score` (0..1)
- `grounded` (0/1)
- `completion` (0/1)
- `failure_category`

## Summary (`summary.json`)

Required fields:

- `variant`
- `item_count`
- `retrieval_hit_rate`
- `answer_score_mean`
- `grounded_rate`
- `completion_rate`
- `pass_rate`

