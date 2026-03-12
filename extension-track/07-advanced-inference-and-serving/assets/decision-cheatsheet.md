# Decision Cheatsheet: Batch vs Service-Style vs Cloud-Native

## Choose Batch-Style on LUMI-G When

- requests are queued and large in volume
- throughput is primary goal
- latency per individual request is secondary

## Choose Service-Style in Scheduled LUMI-G Jobs When

- repeated internal requests arrive during allocation windows
- you need response turnaround faster than offline batch
- endpoint exposure is limited to internal testing/workflow use

## Prefer Cloud-Native Path When

- continuously available public endpoints are required
- full web-platform concerns dominate (auth, gateway, autoscaling)
- integration requirements exceed job-scheduled service constraints

## Practical Rule

Start with LUMI-G batch/service-style for heavy internal inference close to training/evaluation workflows. Move outward only when endpoint lifecycle requirements dominate compute concerns.

