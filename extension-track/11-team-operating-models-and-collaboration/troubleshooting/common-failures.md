# Common Collaboration Failures

## 1) Duplicate "Authoritative" Datasets

Symptoms:

- two or more dataset locations are treated as source-of-truth
- team members run from inconsistent copies

Fix:

- assign one dataset source-of-truth per version
- document derivative working copies explicitly

## 2) Ownerless Promoted Artifacts

Symptoms:

- promoted outputs exist without accountable owner or approver
- no clear path for rollback or update

Fix:

- require owner and approver fields in promotion package
- reject handoff if ownership is missing

## 3) Evaluation Detached From Handoff

Symptoms:

- receiving team gets outputs without benchmark context
- known limitations are lost in transfer

Fix:

- include evaluation summary and known limitations in every handoff bundle
- require checklist completion before handoff

## 4) Overbroad Sharing Scope

Symptoms:

- full project storage exposed when only one dataset/artifact is needed

Fix:

- use scoped read sharing to minimum required prefixes
- separate working and promoted paths clearly

## 5) Collaboration Lives In Team Memory

Symptoms:

- processes vary by person
- project stalls when one contributor is unavailable

Fix:

- maintain written operating model and responsibility matrix
- define backup owners and escalation contacts
