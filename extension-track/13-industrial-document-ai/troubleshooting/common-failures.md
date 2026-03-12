# Common Domain Failure Modes

## 1) Step-Order Breakage In Procedures

Symptoms:

- retrieved chunks omit required step sequence
- generated answer reorders operational steps

Fix:

- preserve procedure sequence metadata
- enforce sequence-aware answer formatting

## 2) Unsupported Confident Answers

Symptoms:

- fluent answer returned with weak or missing evidence

Fix:

- enforce evidence-required answer schema
- fail with review flag when evidence is insufficient

## 3) Revision Identity Loss

Symptoms:

- answer cites document ID without revision
- draft and approved guidance mixed in retrieval

Fix:

- require revision and approval fields in corpus and outputs
- filter retrieval to approved scope for operational answers

## 4) Draft/Approved Source Mixing

Symptoms:

- draft procedure content appears in production responses

Fix:

- split draft and approved indexes or filter strictly at query time
- include approval state checks in promotion gate

## 5) Fluency-Only Evaluation

Symptoms:

- high readability accepted despite weak technical support

Fix:

- evaluate support, correctness, omission risk, and failure taxonomy
- treat fluency as secondary metric
