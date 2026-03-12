# Technical Corpus Schema Template

## Document-Level Fields

| Field | Type | Required | Description |
|---|---|---|---|
| doc_id | string | yes | Stable document identifier |
| doc_revision | string | yes | Revision/version tag |
| approval_state | enum | yes | `draft`, `approved`, `retired` |
| doc_type | enum | yes | `manual`, `procedure`, `report`, `bulletin` |
| title | string | yes | Document title |
| source_uri | string | yes | Source location or identifier |
| effective_date | date | no | Valid-from date |

## Chunk-Level Fields

| Field | Type | Required | Description |
|---|---|---|---|
| chunk_id | string | yes | Stable chunk identifier |
| doc_id | string | yes | Parent document ID |
| doc_revision | string | yes | Parent revision |
| section_ref | string | no | Section or heading |
| chunk_order | int | yes | Sequence position |
| chunk_text | string | yes | Chunk content |
| procedure_step | string | no | Procedure step reference |

## Rules

- no chunk without document revision linkage
- approved and draft documents must be separable
- procedure documents must preserve sequence metadata
