# Worked Example: Corpus Design

## Document Classes

- equipment manuals
- approved operating procedures
- maintenance bulletins

## Metadata Requirements

- stable `doc_id`
- explicit `doc_revision`
- `approval_state` with draft/approved separation
- section references and procedure step markers

## Chunking Note

Procedure documents are chunked with sequence preservation to avoid step-order breakage.
Manual references are chunked by section for retrieval precision.
