# PLZ Centroid Candidate Source Inventory Template

- Status: Non-final
- Slice: plz_centroid_candidate_source_inventory_template
- Phase: External Data Acquisition + Reference Mapping + Source QA
- Scope: blank evidence inventory template for future PLZ/postcode centroid source candidates

## Purpose

Provide a compact template for recording direct evidence before any future PLZ/postcode centroid source candidate is considered for download, regeneration, source replacement, source promotion, mapping use, or downstream integration.

This template does not select, rank, approve, promote, download, regenerate, transform, or use any source.

## Inventory template

| candidate_id | source_identity | stable_source_reference | publisher_or_maintainer | access_date | reference_date | license_or_usage_terms | attribution_requirements | redistribution_or_storage_constraints | spatial_level | temporal_update_semantics | raw_acquisition_path | reproducible_lineage_path | coordinate_derivation_method | crs_and_precision | limitations | temporal_availability | causal_availability | leakage_review | nrw_boundary_consistency | zipcode_to_municipality_mapping_qa | decision_label | todo_verify_fields | stop_condition |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TODO-VERIFY | TODO-VERIFY | TODO-VERIFY | TODO-VERIFY | TODO-VERIFY | TODO-VERIFY | TODO-VERIFY | TODO-VERIFY | TODO-VERIFY | TODO-VERIFY | TODO-VERIFY | Deferred, not implemented | TODO-VERIFY | TODO-VERIFY | TODO-VERIFY | TODO-VERIFY | TODO-VERIFY | TODO-VERIFY | TODO-VERIFY | TODO-VERIFY | TODO-VERIFY | Blocked | TODO-VERIFY | Stop before download, regeneration, source replacement, source promotion, mapping use, downstream integration, or TODO-VERIFY closure by assumption. |

## Evidence acceptance criteria

- Every non-empty candidate row must cite direct source or repository evidence.
- Missing, conflicting, or unsupported fields remain TODO-VERIFY.
- File presence, same-commit references, historical helper scripts, and candidate-source similarity are not sufficient evidence.
- Decision labels are limited to: Allowed to proceed, Blocked, Deferred, not implemented, TODO-VERIFY, No reliable evidence, Non-final.
- Default decision label is Blocked until all evidence required for a later phase-safe action is documented.

## Stop conditions

Stop if the slice would require web research, candidate source selection, source ranking, download, data regeneration, file transformation, source replacement, source promotion, mapping use, downstream integration, modeling, or TODO-VERIFY closure by assumption.

Stop if wording would imply source validation, license suitability, proven upstream provenance, known coordinate method, proven mapping correctness, or resolved leakage posture.

## Non-goals

- No web research.
- No candidate source selection or ranking.
- No download.
- No data regeneration.
- No source replacement or source promotion.
- No mapping use or downstream integration.
- No modeling.
- No out-of-phase assertions.

## Current status

Blocked / TODO-VERIFY / No reliable evidence / Non-final.
