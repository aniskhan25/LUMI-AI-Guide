# Expected Schemas

## Corpus (`sample_corpus.jsonl`)

Required fields:

- `doc_id` (string)
- `title` (string)
- `text` (string)
- `metadata` (object, optional but recommended)

## Queries (`sample_queries.jsonl`)

Required fields:

- `query_id` (string)
- `query` (string)

## Chunk Manifest (`chunks.jsonl`)

Required fields:

- `chunk_id`
- `doc_id`
- `chunk_index`
- `chunk_text`
- `start_word`
- `end_word`
- `metadata` (optional passthrough)

## Embeddings (`embeddings.jsonl`)

Required fields:

- `chunk_id`
- `embedding` (array of float, fixed dimension)

## Retrieval Results (`retrieval_results.jsonl`)

Required fields:

- `query_id`
- `query`
- `retrieved` (array)
  - each item: `chunk_id`, `score`, `chunk_text`, `doc_id`

## Answers (`answers.jsonl`)

Required fields:

- `query_id`
- `query`
- `answer`
- `evidence_chunk_ids` (array)
- `retrieved_contexts` (array of selected snippets)

