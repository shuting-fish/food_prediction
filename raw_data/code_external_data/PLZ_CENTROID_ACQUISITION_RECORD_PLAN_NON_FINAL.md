# PLZ Centroid Acquisition Record Plan - NON_FINAL

## Status

| Field | Value |
|---|---|
| Status | Non-final |
| Phase | External Data Acquisition + Reference Mapping + Source QA |
| Work mode | Documentation-only acquisition-record planning |
| Target current artifact | `raw_data/code_external_data/_reference_geo/plz_centroids_nrw.csv` |
| Candidate source context | `yetzt/postleitzahlen` as candidate context only |
| Current decision | Blocked / TODO-VERIFY / No reliable evidence |

## Evidence basis

This plan is based only on current repository documentation:

- `raw_data/code_external_data/PLZ_CENTROID_SOURCE_REPLACEMENT_REQUIREMENTS_GATE.md`
- `raw_data/code_external_data/PLZ_CENTROID_CANDIDATE_SOURCE_INVENTORY_TEMPLATE.md`
- `raw_data/code_external_data/PLZ_ODBL_APPROVAL_CRITERIA_AND_ACQUISITION_RECORD_TEMPLATE.md`
- `raw_data/code_external_data/external_source_documentation.md`

Repository documentation records that the current PLZ centroid file is structurally inspected, but source identity, license, upstream lineage, coordinate method, temporal basis, causal availability, leakage posture, and mapping quality remain `TODO-VERIFY` / `No reliable evidence`.

Repository documentation also records `yetzt/postleitzahlen` as candidate context only. No repository evidence ties `yetzt/postleitzahlen` or any pinned asset to the local `plz_centroids.csv` historical helper input or to `raw_data/code_external_data/_reference_geo/plz_centroids_nrw.csv`.

## Non-approval statement

This plan does not approve:

- data download;
- release asset download;
- raw asset storage;
- redistribution;
- centroid derivation;
- PLZ centroid replacement;
- ZIP/postcode-to-municipality joins;
- source promotion;
- downstream use;
- legal suitability;
- modelling;
- out-of-scope outcome assertions.

## Proposed acquisition record structure

Any future acquisition record must preserve unresolved fields as `TODO-VERIFY` until direct evidence exists.

| Field | Planned value or required status |
|---|---|
| `record_status` | Non-final / TODO-VERIFY |
| `source_identity` | `yetzt/postleitzahlen` candidate context only |
| `source_repository_url` | `https://github.com/yetzt/postleitzahlen` candidate context only |
| `release_tag` | `2026.02` candidate context only |
| `release_url` | `https://github.com/yetzt/postleitzahlen/releases/tag/2026.02` candidate context only |
| `asset_name` | `postleitzahlen.geojson.br` candidate context only |
| `asset_download_url` | `https://github.com/yetzt/postleitzahlen/releases/download/2026.02/postleitzahlen.geojson.br` candidate context only |
| `asset_published_at` | `2026-02-20T19:10:41Z` candidate context only |
| `asset_sha256_expected` | `3ead6646869a389ccd17a00bd4179b287d2a8440f3499f16983cb86aaee99dbd` candidate context only |
| `access_date` | TODO-VERIFY |
| `download_authorization_status` | Blocked / TODO-VERIFY |
| `local_raw_storage_path` | TODO-VERIFY |
| `local_file_sha256_observed` | TODO-VERIFY |
| `checksum_match_status` | TODO-VERIFY |
| `license_name` | ODbL 1.0 / TODO-VERIFY legal approval |
| `license_url` | `https://opendatacommons.org/licenses/odbl/1-0/` / TODO-VERIFY legal approval |
| `osm_copyright_url` | `https://www.openstreetmap.org/copyright` / TODO-VERIFY legal approval |
| `source_attribution_notice` | TODO-VERIFY |
| `storage_decision` | Blocked / TODO-VERIFY |
| `redistribution_decision` | Blocked / TODO-VERIFY |
| `derivative_database_decision` | TODO-VERIFY |
| `share_alike_decision` | TODO-VERIFY |
| `raw_asset_schema_inspection_status` | TODO-VERIFY |
| `geometry_type` | TODO-VERIFY until asset inspection |
| `geometry_crs_status` | TODO-VERIFY |
| `geometry_quality_status` | TODO-VERIFY |
| `duplicate_plz_check_status` | TODO-VERIFY |
| `missing_plz_check_status` | TODO-VERIFY |
| `nrw_filtering_status` | TODO-VERIFY |
| `centroid_derivation_status` | Blocked / TODO-VERIFY |
| `centroid_method` | TODO-VERIFY |
| `centroid_precision_policy` | TODO-VERIFY |
| `plz_to_municipality_ags_limitations` | TODO-VERIFY |
| `temporal_availability_status` | TODO-VERIFY |
| `publication_lag_status` | TODO-VERIFY |
| `revision_lag_status` | TODO-VERIFY |
| `backfill_behavior_status` | TODO-VERIFY |
| `causal_availability_status` | TODO-VERIFY |
| `leakage_review_status` | TODO-VERIFY |
| `lineage_to_current_plz_centroids_nrw_csv` | No reliable evidence |
| `remaining_todo_verify` | TODO-VERIFY |

## Blockers before any download

Stop before any download unless all of the following are resolved with direct evidence:

- legal approval for ODbL use: TODO-VERIFY;
- approved attribution and license-notice text: TODO-VERIFY;
- repository storage decision: TODO-VERIFY;
- redistribution decision: TODO-VERIFY;
- approved raw storage path: TODO-VERIFY;
- approved acquisition-record location: TODO-VERIFY;
- checksum verification procedure: TODO-VERIFY;
- explicit statement that download does not replace current PLZ centroid data: TODO-VERIFY;
- explicit statement that download does not approve centroid derivation: TODO-VERIFY.

Current download decision: Blocked. No asset download is approved by this plan.

## Blockers before centroid derivation

Stop before centroid derivation unless all of the following are resolved with direct evidence:

- approved legal, storage, and redistribution decision for ODbL-derived geometry: TODO-VERIFY;
- approved and verified asset download with observed SHA256 match: TODO-VERIFY;
- schema and layer inspection: TODO-VERIFY;
- geometry validity QA: TODO-VERIFY;
- CRS verification and transformation policy: TODO-VERIFY;
- explicit centroid derivation method: TODO-VERIFY;
- centroid precision and rounding policy: TODO-VERIFY;
- NRW filtering method and QA: TODO-VERIFY;
- duplicate and missing PLZ checks: TODO-VERIFY;
- PLZ-to-municipality and AGS limitation note: TODO-VERIFY;
- temporal availability, publication lag, revision lag, and backfill behavior review: TODO-VERIFY;
- causal availability and leakage review: TODO-VERIFY;
- explicit statement that derived centroids do not replace `plz_centroids_nrw.csv` without a separate approved replacement gate: TODO-VERIFY.

Current centroid derivation decision: Blocked. No centroid derivation is approved by this plan.

## Forbidden inferences

Do not infer that:

- `yetzt/postleitzahlen` is selected or approved;
- candidate source similarity proves current local lineage;
- download, storage, redistribution, derivation, replacement, joins, source promotion, or downstream use is authorized;
- any legal suitability claim can be made;
- coordinate method, CRS, precision, or quality is resolved;
- NRW boundary consistency or ZIP/postcode-to-municipality correctness is resolved;
- temporal availability, causal availability, or leakage posture is resolved.

## Remaining TODO-VERIFY

- ODbL/legal approval;
- attribution wording and sufficiency;
- repository storage decision;
- redistribution decision;
- approved raw storage path;
- approved acquisition-record location;
- checksum procedure and observed checksum;
- schema and layer inspection;
- geometry CRS and quality checks;
- NRW filtering QA;
- centroid derivation method and precision policy;
- ZIP/postcode and AGS limitations;
- temporal availability, publication lag, revision lag, and backfill behavior;
- causal availability and leakage review;
- lineage from candidate source or pinned asset to current `plz_centroids_nrw.csv`.

## Smallest safe next step

Use this file only as a Non-final documentation plan. Any future acquisition-record implementation, download, storage decision, derivation, replacement, join, or source promotion requires a separate explicit request, fresh repository verification, and direct evidence sufficient to keep unresolved fields as `TODO-VERIFY` until closed by evidence.
