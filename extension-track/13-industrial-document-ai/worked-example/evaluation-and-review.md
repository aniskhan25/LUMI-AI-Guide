# Worked Example: Evaluation and Review

## Scorecard

- evidence relevance
- support completeness
- technical correctness
- omission risk
- unsupported-answer rate

## Review Rule

Any answer with weak retrieval confidence or missing critical evidence receives `review_required=true`.

## Promotion Gate

Promote only when:

- failure categories are reviewed
- high-risk unsupported answers are below threshold
- update owner and approver are recorded
