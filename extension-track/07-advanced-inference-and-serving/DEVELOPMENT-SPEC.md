# Development Spec Template

Use this document to lock decisions and track implementation for Lesson 07.

## 1. Lesson Identity

- Lesson id: `EXT-07`
- Title: `Advanced Inference and Serving Patterns on LUMI-G`
- Short nav title: `Serving and advanced inference on MI250X`
- Owner: `<team or person>`
- Target branch: `feature/advanced-onboarding-lessons`

## 2. Decision Lock (Must Complete First)

| Decision | Options | Chosen | Rationale | Owner | Date |
|---|---|---|---|---|---|
| Primary inference mode | batched, service-style, hybrid | `TBD` | `TBD` | `TBD` | `TBD` |
| Engine path | transformers, vLLM-style, custom | `TBD` | `TBD` | `TBD` | `TBD` |
| Controlled comparison | batch size, concurrency, both | `TBD` | `TBD` | `TBD` | `TBD` |
| Output schema contract | strict response+error schema | `TBD` | `TBD` | `TBD` | `TBD` |
| Deployment recommendation rule | throughput/latency thresholds | `TBD` | `TBD` | `TBD` | `TBD` |

## 3. Scope Boundaries

### In Scope

- High-throughput batched inference
- Service-style loop inside scheduled GPU jobs
- Request/response schema and metrics logging
- Throughput/latency comparison
- Short decision framework for pattern selection

### Out Of Scope

- Public API production operations
- Full cloud-native deployment architecture
- Enterprise auth/gateway engineering
- Kubernetes and autoscaling operations

## 4. Learning Outcomes

Learner can:

1. choose between batch-style and service-style inference on LUMI-G
2. run repeatable inference loops with stable request/response IDs
3. collect throughput and latency metrics
4. compare controlled configuration changes responsibly
5. decide if LUMI-G pattern fits workload needs

## 5. Lesson Structure Contract

The lesson must include:

- A. What this lesson enables
- B. When to use this workflow
- C. Prerequisites
- D. Workflow at a glance
- E. Minimal working example
- F. Verification
- G. Which serving pattern to choose
- H. Throughput and latency tradeoffs
- I. Output and logging design
- J. Common failure modes
- K. Operational checklist
- L. Next lesson

## 6. Mandatory Deliverables

| Deliverable | Required | Path |
|---|---|---|
| Lesson README | Yes | `README.md` |
| Request schema and sample set | Yes | `data/sample_requests.jsonl`, `data/expected-schema.md` |
| Batched inference runner | Yes | `scripts/run_batched_inference.py` |
| Service loop runner | Yes | `scripts/run_service_loop.py` |
| Metrics collector | Yes | `scripts/collect_metrics.py` |
| Summary/comparison script | Yes | `scripts/summarize_results.py` |
| Canonical jobs | Yes | `jobs/run_batched_inference.sh`, `jobs/run_service_style_inference.sh` |
| Troubleshooting page | Yes | `troubleshooting/common-failures.md` |

## 7. Acceptance Criteria

### Content

- Golden path is clear and procedural
- Batch vs service distinction is explicit
- Throughput/latency tradeoffs are practical

### Technical

- Structured request/response artifacts produced
- GPU visibility confirmable
- Controlled configuration comparison included
- Summary report reproducible

### Pedagogical

- Teaches one capability: advanced inference packaging on LUMI-G
- Learner can justify pattern choice with metrics
- Learner can explain configuration tradeoffs

## 8. Testing Plan

| Gate | Command | Pass/Fail | Notes |
|---|---|---|---|
| Batched run | `python scripts/run_batched_inference.py --config configs/inference.yaml` | `TBD` | `TBD` |
| Service run | `python scripts/run_service_loop.py --config configs/service.yaml` | `TBD` | `TBD` |
| Metrics collect | `python scripts/collect_metrics.py --config configs/inference.yaml --mode batched` | `TBD` | `TBD` |
| Summary compare | `python scripts/summarize_results.py --compare-config configs/compare.yaml` | `TBD` | `TBD` |
| LUMI batched | `sbatch jobs/run_batched_inference.sh` | `TBD` | `TBD` |
| LUMI service | `sbatch jobs/run_service_style_inference.sh` | `TBD` | `TBD` |

## 9. Review Sign-Off

| Role | Name | Status | Date | Notes |
|---|---|---|---|---|
| Content reviewer | `TBD` | `Pending` | `TBD` | `TBD` |
| Technical reviewer | `TBD` | `Pending` | `TBD` | `TBD` |
| Platform reviewer (inference/serving focus) | `TBD` | `Pending` | `TBD` | `TBD` |

