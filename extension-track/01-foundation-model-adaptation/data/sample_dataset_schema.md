# Sample Dataset Schema

The lesson uses JSONL files for train/eval data.

## File format

- One JSON object per line
- UTF-8 encoding
- Required keys:
  - `text` (string)
  - `label` (integer, class id)

Example record:

```json
{"text":"training job completed with stable loss","label":1}
```

## Expected files

- `data/sample_data/train.jsonl`
- `data/sample_data/eval.jsonl`

Generate them with:

```bash
python data/prepare_sample_data.py --output data/sample_data
```

