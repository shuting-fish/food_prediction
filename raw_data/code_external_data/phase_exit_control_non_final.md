# Food Prediction - Phase Exit Control - NON_FINAL

## Status

Status: Non-final.

This document is a phase-boundary control record only. It does not close the project phase as final, validated, release-ready, or QA-passed.

## Active Phase

External Data Acquisition + Reference Mapping + Source QA.

## Scope Boundary

Allowed current-phase work:
- external source review
- source documentation before use
- raw storage and lineage review
- reference mapping review
- join feasibility review
- causal availability review
- leakage review
- source, lineage, geospatial, PLZ, ZIP, AGS, VG250, and municipality QA
- read-only repository verification
- documentation hardening inside the current phase
- TODO-VERIFY tracking

Deferred, not implemented:
- ML integration
- model training
- model comparison
- final training feature engineering
- SHAP
- feature importance
- Streamlit
- deployment
- business recommendations

## Phase-Blocking Before Non-final Phase Exit

| Item | Reason | Required next slice |
|---|---|---|
| Phase boundary record | The current phase needs a compact explicit boundary so TODO-VERIFY items do not create uncontrolled scope expansion. | Keep this document updated only from fresh evidence. |
| Fresh repository baseline before repo-facing work | Repository, branch, file, and diff state must not be inferred from handoff evidence. | Run fresh PowerShell/Git verification before any file edit, commit, push, or PR. |
| Source/TODO consistency review | Open TODO-VERIFY and No reliable evidence limits must remain visible and internally consistent. | Read-only consistency review of current source-QA and mapping documentation. |

## Preserve as TODO-VERIFY

The following unresolved areas remain TODO-VERIFY unless later resolved by direct source evidence, file evidence, reproducible checks, current command output, or explicit user confirmation:

- PLZ centroid source identity
- stable source URL, repository, API, or download path
- access date and reference date
- license or usage terms
- local lineage to `plz_centroids.csv` or `plz_centroids_nrw.csv`
- reproducible acquisition path
- coordinate method, precision, and quality
- temporal availability, publication lag, revision lag, and backfill behavior
- causal availability and leakage risk
- ZIP-to-municipality truth
- multi-municipality ZIP allocation logic
- Selfkant/Waldfeucht coverage-gap cause and correctness
- AGS/Gemeindeschluessel identity, format, length, and leading-zero source validation
- VG250 local cache version and reference date
- VG250 geometry validity, CRS/layer integrity, calculated spatial bounds, and full NRW boundary consistency
- store coordinate source quality
- store-to-municipality spatial assignment quality

## No Reliable Evidence

No reliable evidence supports the following claims in the current phase:

- PLZ source promotion
- PLZ centroid source quality
- local lineage from `yetzt/postleitzahlen` to local PLZ centroid files
- ZIP-to-municipality mapping correctness
- one-to-one ZIP-to-municipality truth
- predictive value
- forecast improvement
- feature value
- model impact
- operational benefit
- business benefit

## Minimum Remaining Slice Plan

Hard cap target: 3 to 5 small slices. Do not expand this plan only because a TODO-VERIFY item exists.

| Slice | Objective | Allowed files | Stop rule |
|---|---|---|---|
| Fresh repo baseline verification | Confirm branch, status, divergence, and diff safety before any repo-facing work. | No-file status. | Stop on dirty tree, unexpected branch, divergence, or unclear remotes. |
| Read-only phase document consistency review | Check that current documentation preserves TODO-VERIFY and No reliable evidence limits. | Read-only: `external_source_documentation.md`, `external_source_evidence_audit.md`, `zipcode_ags_reference_qa.md`, `external_source_qa_registry.csv`. | Stop if review would require new source research, source promotion, or value claims. |
| Optional documentation-only correction | Fix only a verified contradiction or wording drift found in the consistency review. | Only explicitly identified documentation file(s). | Stop if change touches data, scripts, registry semantics, notebooks, source promotion, modeling, or value claims. |

## Maintenance Rule

Do not resolve TODO-VERIFY by assumption.

Do not treat handoff evidence as live repository truth.

Do not use this document to claim source quality, mapping correctness, predictive value, forecast improvement, model impact, operational benefit, or business benefit.
