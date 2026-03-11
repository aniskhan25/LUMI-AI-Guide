# Common Failures

## 1) Weak chunking hurts retrieval quality

Symptoms:

- retrieved chunks do not contain answer-relevant facts
- grounded answers look generic or off-topic

Fix:

- adjust `chunking.chunk_words` and `chunking.overlap_words`
- preserve meaningful sentence boundaries when possible
- rerun and inspect retrieval outputs before tuning generation

## 2) ID drift between artifacts

Symptoms:

- validation reports chunk/embedding/retrieval mismatch

Fix:

- keep stable `doc_id`, `chunk_id`, and `query_id`
- regenerate downstream artifacts after changing chunking
- avoid mixing files from different runs in one output directory

## 3) Accidental CPU execution

Symptoms:

- `GPU_VISIBLE_COUNT=0`
- embedding or generation steps are unexpectedly slow

Fix:

- submit to GPU partition
- load AI container bindings module in job script
- use `--require-gpu` in validation

## 4) Empty or malformed retrieval context

Symptoms:

- retrieval records have empty context text
- generation answers ignore corpus content

Fix:

- verify chunk manifest content and index build success
- ensure retrieval `top_k` is > 0
- validate retrieval file before answer generation

## 5) Prompt ignores evidence

Symptoms:

- answer style is generic with no grounding

Fix:

- keep explicit evidence section in prompt template
- include evidence chunk IDs in output records
- inspect `retrieved_contexts` alongside answers

