# Worked Example: Collaboration Pattern

## Selected Team Model

Split-role model with clear boundaries:

- data custodian owns dataset versions
- workflow developer owns experiment runs
- evaluator owns benchmark report and recommendation
- promotion approver owns go/no-go decision
- delivery owner owns external handoff bundle

## Review Flow

1. Workflow developer submits candidate run with manifest and outputs.
2. Evaluator validates benchmark results and tags failure classes.
3. Promotion approver checks evidence and approves or rejects promotion.
4. Delivery owner assembles handoff package for consuming team.

## Promotion Rule

No artifact is promoted unless:

- source run is traceable
- evaluation report is attached
- owner and approver are recorded
- known limitations are documented
