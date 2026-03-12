# Worked Example: Sensitive-Data Variant (After Redesign)

## Revised Pattern Goal

Keep architecture benefits of retrieval-grounded responses while reducing unnecessary data exposure.

## Revised Flow

restricted corpus -> selective extraction/minimization -> pseudonymized chunking -> retrieval -> constrained prompt build -> grounded generation -> reviewed outputs

## Key Redesign Moves

1. Field minimization before chunking:
   - Remove non-essential identifying attributes.
2. Pseudonymous identifiers:
   - Replace direct IDs with stable surrogate IDs for workflow stages.
3. Prompt discipline:
   - Include only fields required for answer quality.
4. Logging minimization:
   - Store structured status + IDs by default; avoid raw full-text logs except controlled review windows.
5. Trust gate:
   - Run sensitivity leak check + unsupported-answer check before broader downstream use.

## Resulting Operational Change

The workflow remains usable for grounded QA while producing fewer high-risk artifacts and clearer review checkpoints.

