# PLZ Centroid Source Replacement Requirements Gate

- Status: Non-final
- Slice: plz_centroid_source_replacement_requirements_gate
- Phase: External Data Acquisition + Reference Mapping + Source QA
- Target artifact under control: raw_data/code_external_data/_reference_geo/plz_centroids_nrw.csv

## Purpose

Define the minimum evidence required before any future PLZ/postcode centroid source or replacement path is considered for download, regeneration, source promotion, mapping use, or downstream integration.

This gate references the existing evidence inventories in `external_source_documentation.md`, `zipcode_ags_reference_qa.md`, `external_source_evidence_audit.md`, `phase_exit_control_non_final.md`, and `external_source_qa_registry.csv`. It does not replace those records.

## Current controlling decision

Blocked.

The local `plz_centroids_nrw.csv` file is present and structurally inspected, but the source, license, upstream lineage, coordinate method, temporal basis, causal availability, leakage posture, and mapping quality remain TODO-VERIFY / No reliable evidence.

## Minimum source documentation requirements

Any future candidate source or replacement path must document:

- source identity;
- stable source URL, repository, API endpoint, or download path;
- access date and, where available, source reference date;
- source owner or publisher;
- license or usage terms;
- attribution requirements;
- spatial level and covered geography;
- update logic, publication lag, revision lag, and backfill behavior;
- raw storage path and reproducible acquisition command or procedure.

## Minimum lineage acceptance criteria

Allowed to proceed only if repository evidence records a reproducible chain from upstream source to raw local artifact, then from raw local artifact to any derived PLZ centroid output.

The lineage record must include input path, output path, transformation script or documented procedure, checksum or equivalent file identity, schema, row or feature count, and explicit notes for any manual step. File presence, same-commit references, or historical helper scripts alone are not enough.

## Minimum license / usage acceptance criteria

Allowed to proceed only if direct source evidence documents license or usage terms, attribution requirements, redistribution constraints, and compatibility limits for repository storage and later project use.

Until this evidence exists, license status remains TODO-VERIFY / No reliable evidence.

## Minimum coordinate-method acceptance criteria

Any candidate centroid path must document the coordinate derivation method, CRS, transformation steps, precision, rounding, geometry source, and quality checks.

If polygon-to-centroid generation is used, the centroid method must be explicit. PLZ centroids must remain documented as approximate reference coordinates, not precise store coordinates.

## Minimum temporal availability and leakage criteria

Any candidate path must document reference date, update cadence, publication lag, revision lag, backfill behavior, prediction-time availability, causal availability, and leakage review.

Until direct evidence exists, temporal availability, causal availability, and leakage posture remain TODO-VERIFY / No reliable evidence.

## Minimum mapping / geospatial QA criteria

Any candidate path must document join keys, ZIP/postcode normalization, duplicate key handling, missing key handling, multi-municipality ZIP handling, NRW filtering logic, coordinate bounds checks, geometry validity where applicable, and NRW boundary consistency checks.

Mapping correctness and ZIP/postcode-to-municipality correctness remain TODO-VERIFY unless supported by direct source evidence and reproducible QA.

## Explicit non-goals

- No web research.
- No downloads.
- No data regeneration.
- No source replacement.
- No source promotion.
- No modeling.
- No feature engineering for training.
- No downstream integration.
- No closure of TODO-VERIFY by assumption.
- No lineage inference from `yetzt/postleitzahlen` or any other candidate source to the local PLZ centroid file without direct evidence.

## Acceptance decision labels

- Allowed to proceed: all required evidence for a future phase-safe action is directly documented.
- Blocked: one or more required evidence areas remain missing, conflicting, or unsupported.
- Deferred, not implemented: action is outside the current approved slice.
- TODO-VERIFY: evidence is unresolved and must remain open.
- No reliable evidence: repository evidence does not support the claim.
- Non-final: project status while unresolved gate items remain open.

## Stop conditions

Stop if any future step would require source research, download, data regeneration, source replacement, source promotion, modeling, training feature engineering, downstream integration, or unsupported status closure.

Stop if wording would imply the source is selected, license is suitable, upstream provenance is proven, coordinate method is known, mapping correctness is proven, leakage posture is resolved, or outcome-scope support exists.

## Current status

Blocked / TODO-VERIFY / No reliable evidence / Non-final.

The next safe action is a phase-safe source-evidence decision gate. No download, regeneration, replacement, promotion, mapping use, downstream integration, commit, or push is authorized by this file.
