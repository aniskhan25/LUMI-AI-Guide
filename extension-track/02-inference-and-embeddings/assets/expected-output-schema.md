# Expected Output Schema

## Embeddings output (`embeddings.jsonl`)

Each line is a JSON object:

- `id` (string): original input identifier (must be preserved)
- `embedding` (array of float): dense vector with consistent dimension
- `metadata` (object, optional): copied metadata from input

Example:

```json
{"id":"doc-0001","embedding":[0.0123,-0.9987,0.1244],"metadata":{"domain":"energy","lang":"en"}}
```

## Generation output (`generation_outputs.jsonl`)

Each line is a JSON object:

- `id` (string): original input identifier (must be preserved)
- `prompt` (string): source prompt
- `output_text` (string): generated output, non-empty

Example:

```json
{"id":"gen-0001","prompt":"Summarize...","output_text":"Deployment completed with temporary latency increase."}
```

## Run summary (`run_summary.json`)

Required fields:

- `mode`
- `run_name`
- `model_name`
- `device`
- `gpu_visible_count`
- `records_written`
- `input_jsonl`
- `output_jsonl`
- runtime parameters such as batch size and sequence settings

