# Common Lifecycle Failures

## 1) Mystery Runs

Symptoms:

- output folders exist without manifests
- no reliable link to dataset/config/model versions

Fix:

- require `manifest.yaml` in every run directory
- reject run folders that fail manifest validation

## 2) Multiple "Final" Artifacts

Symptoms:

- more than one artifact labeled final
- no formal promotion decision

Fix:

- enforce lifecycle states
- promote only versions that pass checklist and review

## 3) Provenance Drift

Symptoms:

- copied artifacts in new locations without source run ID
- evaluation summaries detached from outputs

Fix:

- keep source run ID in promoted metadata
- colocate evaluation summary with promoted artifacts

## 4) Environment Drift

Symptoms:

- reruns differ because container/env is not recorded

Fix:

- store container reference in manifest
- record execution command and relevant runtime notes

## 5) Unbounded Recompute

Symptoms:

- expensive reruns happen because prior outputs are hard to find

Fix:

- separate draft, reviewed, and promoted directories
- maintain stable naming conventions and index files
