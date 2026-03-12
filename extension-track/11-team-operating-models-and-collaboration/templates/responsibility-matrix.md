# Responsibility Matrix Template

Use this matrix to avoid ambiguous ownership.

| Task | Data Custodian | Workflow Developer | Evaluator | Promotion Approver | Delivery Owner |
|---|---|---|---|---|---|
| Define dataset version | A | C | I | I | I |
| Prepare working dataset copy | R | C | I | I | I |
| Run workflow experiment | I | R | C | I | I |
| Produce evaluation report | I | C | R | C | I |
| Approve promotion decision | I | C | C | A | I |
| Package promoted artifact | I | C | C | A | R |
| Publish handoff bundle | I | I | I | C | R |

Legend:

- `R`: responsible
- `A`: accountable
- `C`: consulted
- `I`: informed
