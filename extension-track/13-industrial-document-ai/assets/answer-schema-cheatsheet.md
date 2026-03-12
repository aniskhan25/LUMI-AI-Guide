# Answer Schema Cheat Sheet

Minimum response record:

- `query_id`
- `answer_text`
- `evidence_doc_ids` (list)
- `evidence_chunk_ids` (list)
- `review_required` (bool)
- `workflow_version`

Rule: no delivery-grade answer without evidence IDs.
