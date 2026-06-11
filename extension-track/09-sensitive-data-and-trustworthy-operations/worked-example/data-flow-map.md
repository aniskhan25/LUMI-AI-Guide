# Worked Example: Data Flow Map

| Stage | Input | Output | Sensitivity | Risk | Control |
|---|---|---|---|---|---|
| Ingestion | Restricted internal documents | Staged corpus snapshot | Red | broad raw exposure | restricted staging + access boundary |
| Preprocessing | Raw corpus snapshot | Minimized document records | Red -> Amber | retaining unnecessary fields | drop non-essential fields |
| Chunking | Minimized records | Pseudonymized chunk manifest | Amber | accidental identifier carry-over | surrogate IDs and schema checks |
| Retrieval | Chunk index + query | Retrieved contexts | Amber | over-retrieval of sensitive context | top-k and metadata filters |
| Prompt Build | Query + retrieved context | Model prompt | Amber | leakage in prompt text | constrained prompt template |
| Generation | Prompt | Candidate response | Amber | unsupported/sensitive output | trust gate checks |
| Output Logging | Candidate response + metadata | Reviewed output artifact | Amber/Green | raw content retention | ID-based logging + reduced text retention |

## Review Notes

- Highest-risk transitions are ingestion -> preprocessing and retrieval -> prompt build.
- Trust gate is mandatory before publishing outputs outside controlled pilot users.

