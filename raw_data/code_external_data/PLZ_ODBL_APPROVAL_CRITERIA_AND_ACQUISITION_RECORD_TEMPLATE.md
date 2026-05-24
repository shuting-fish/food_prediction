# PLZ ODbL Approval Criteria and Acquisition Record Template

- Status: Non-final
- Phase: External Data Acquisition + Reference Mapping + Source QA
- Scope: documentation-only approval criteria and acquisition-record template for one pinned `yetzt/postleitzahlen` release asset
- Pinned source candidate: `yetzt/postleitzahlen`
- Target current PLZ centroid artifact under project control: `raw_data/code_external_data/_reference_geo/plz_centroids_nrw.csv`

## Non-Approval Statement

This document does not approve legal suitability, repository storage, redistribution, download, data regeneration, PLZ centroid replacement, mapping use, downstream integration, or centroid derivation.

This document is a non-final source-QA control record only. It preserves unresolved legal, storage, redistribution, derivative-data, attribution, temporal, leakage, and mapping questions as `TODO-VERIFY`.

## Phase Boundary

Allowed in the current phase:

- documentation-only source review;
- approval criteria definition;
- reproducibility template definition;
- TODO-VERIFY preservation for legal, storage, redistribution, derivative-data, attribution, temporal, leakage, and mapping questions.

Not allowed by this document:

- data download;
- release asset download;
- raw asset storage;
- data regeneration;
- centroid derivation;
- replacement of current PLZ centroid data;
- ZIP/postcode-to-municipality joins;
- modelling;
- validation or claims about predictive, operational, or business value.

## Pinned Asset Metadata

| Field | Value |
|---|---|
| Source identity | `yetzt/postleitzahlen` |
| Repository URL | `https://github.com/yetzt/postleitzahlen` |
| Release tag | `2026.02` |
| Release URL | `https://github.com/yetzt/postleitzahlen/releases/tag/2026.02` |
| Asset name | `postleitzahlen.geojson.br` |
| Asset URL | `https://github.com/yetzt/postleitzahlen/releases/download/2026.02/postleitzahlen.geojson.br` |
| Published at | `2026-02-20T19:10:41Z` |
| Expected SHA256 | `3ead6646869a389ccd17a00bd4179b287d2a8440f3499f16983cb86aaee99dbd` |
| Asset inspection status | Not downloaded; not inspected |
| Current suitability decision | Blocked / TODO-VERIFY / No reliable evidence for legal suitability, storage, redistribution, download, or centroid derivation approval |

## ODbL / OSM Attribution and License-Notice Criteria

Before any future download, storage, redistribution, or derived output is considered, direct approval evidence must document:

- legal approval for ODbL 1.0 use in this project: TODO-VERIFY;
- required OpenStreetMap attribution wording: TODO-VERIFY;
- required `yetzt/postleitzahlen` source attribution wording: TODO-VERIFY;
- required ODbL license name and URL notice: TODO-VERIFY;
- required source URL, release tag, asset URL, and expected SHA256 notice: TODO-VERIFY;
- required notice location in repository documentation, data directories, metadata, or generated outputs: TODO-VERIFY;
- whether attribution must appear in README files, data dictionaries, generated reports, maps, notebooks, or downstream artifacts: TODO-VERIFY;
- no-warranty and data-quality disclaimer wording: TODO-VERIFY.

Minimum candidate notice fields for later approval:

| Notice field | Required value or status |
|---|---|
| Source credit | `TODO-VERIFY` |
| OpenStreetMap credit | `TODO-VERIFY` |
| License name | `Open Database License (ODbL) 1.0` / TODO-VERIFY legal approval |
| License URL | `https://opendatacommons.org/licenses/odbl/1-0/` / TODO-VERIFY legal approval |
| OSM copyright URL | `https://www.openstreetmap.org/copyright` / TODO-VERIFY legal approval |
| Repository source URL | `https://github.com/yetzt/postleitzahlen` |
| Release URL | `https://github.com/yetzt/postleitzahlen/releases/tag/2026.02` |
| Asset URL | `https://github.com/yetzt/postleitzahlen/releases/download/2026.02/postleitzahlen.geojson.br` |
| Expected SHA256 | `3ead6646869a389ccd17a00bd4179b287d2a8440f3499f16983cb86aaee99dbd` |
| No-warranty disclaimer | TODO-VERIFY |
| Attribution sufficiency | TODO-VERIFY |

## Repository Storage and Redistribution Approval Criteria

No raw asset, extracted geometry, derived centroid file, or derivative database may be stored, committed, published, or redistributed unless all of the following are resolved with direct approval evidence:

- repository storage approval for the raw ODbL-derived asset: TODO-VERIFY;
- decision whether the raw asset may be committed to Git: TODO-VERIFY;
- decision whether the raw asset must remain local-only and untracked: TODO-VERIFY;
- decision whether only source metadata and checksums may be stored in Git: TODO-VERIFY;
- approved local raw storage path: TODO-VERIFY;
- approved `.gitignore` or equivalent exclusion rule if local-only storage is required: TODO-VERIFY;
- redistribution approval for raw asset copies: TODO-VERIFY;
- redistribution approval for extracted geometry: TODO-VERIFY;
- redistribution approval for derived centroid tables: TODO-VERIFY;
- required license and attribution notice location for any permitted storage or redistribution: TODO-VERIFY.

Current storage and redistribution decision:

Blocked. This document does not approve committing, storing, redistributing, publishing, or externally sharing the raw asset or any derived output.

## Derivative Database / Share-Alike Decision Points

Before any centroid derivation, extracted geometry, NRW subset, PLZ-to-municipality mapping, or public output is considered, direct approval evidence must classify the output and obligations:

- whether an NRW-filtered geometry subset is a derivative database: TODO-VERIFY;
- whether a centroid table generated from PLZ polygons is a derivative database: TODO-VERIFY;
- whether a centroid table is a produced work, derivative database, collective database, or another legally approved category: TODO-VERIFY;
- whether combining ODbL-derived PLZ data with project reference data creates share-alike, disclosure, or compatibility obligations: TODO-VERIFY;
- whether any public use requires providing derivative database access or alteration instructions: TODO-VERIFY;
- whether internal-only use changes obligations for this repository and future artifacts: TODO-VERIFY;
- whether current project documentation can mention the source without storing ODbL data: TODO-VERIFY.

Risk note:

ODbL-derived PLZ geometry and any generated centroid table may create derivative-data and share-alike obligations if publicly used, redistributed, published, or combined with other project data. Legal classification is not decided here.

## Acquisition-Record Template Fields

Any future acquisition record for this pinned asset must include the following fields before a download is considered:

| Field | Required status |
|---|---|
| record_status | Non-final / TODO-VERIFY |
| source_identity | `yetzt/postleitzahlen` |
| source_repository_url | `https://github.com/yetzt/postleitzahlen` |
| release_tag | `2026.02` |
| release_url | `https://github.com/yetzt/postleitzahlen/releases/tag/2026.02` |
| asset_name | `postleitzahlen.geojson.br` |
| asset_download_url | `https://github.com/yetzt/postleitzahlen/releases/download/2026.02/postleitzahlen.geojson.br` |
| asset_published_at | `2026-02-20T19:10:41Z` |
| asset_sha256_expected | `3ead6646869a389ccd17a00bd4179b287d2a8440f3499f16983cb86aaee99dbd` |
| access_date | TODO-VERIFY |
| download_authorization_status | TODO-VERIFY |
| local_raw_storage_path | TODO-VERIFY |
| local_file_sha256_observed | TODO-VERIFY |
| checksum_match_status | TODO-VERIFY |
| license_name | ODbL 1.0 / TODO-VERIFY legal approval |
| license_url | `https://opendatacommons.org/licenses/odbl/1-0/` / TODO-VERIFY legal approval |
| osm_copyright_url | `https://www.openstreetmap.org/copyright` / TODO-VERIFY legal approval |
| source_attribution_notice | TODO-VERIFY |
| storage_decision | TODO-VERIFY |
| redistribution_decision | TODO-VERIFY |
| derivative_database_decision | TODO-VERIFY |
| share_alike_decision | TODO-VERIFY |
| raw_asset_schema_inspection_status | TODO-VERIFY |
| geometry_type | TODO-VERIFY until asset inspection |
| geometry_crs_status | TODO-VERIFY |
| geometry_quality_status | TODO-VERIFY |
| duplicate_plz_check_status | TODO-VERIFY |
| missing_plz_check_status | TODO-VERIFY |
| nrw_filtering_status | TODO-VERIFY |
| centroid_derivation_status | Blocked / TODO-VERIFY |
| centroid_method | TODO-VERIFY |
| centroid_precision_policy | TODO-VERIFY |
| plz_to_municipality_ags_limitations | TODO-VERIFY |
| temporal_availability_status | TODO-VERIFY |
| publication_lag_status | TODO-VERIFY |
| revision_lag_status | TODO-VERIFY |
| backfill_behavior_status | TODO-VERIFY |
| causal_availability_status | TODO-VERIFY |
| leakage_review_status | TODO-VERIFY |
| lineage_to_current_plz_centroids_nrw_csv | No reliable evidence |
| remaining_todo_verify | TODO-VERIFY |

## Blockers Before Any Download

Stop before download unless all of the following are resolved:

- legal approval for ODbL use: TODO-VERIFY;
- approved attribution and license-notice text: TODO-VERIFY;
- repository storage decision: TODO-VERIFY;
- redistribution decision: TODO-VERIFY;
- approved raw storage path: TODO-VERIFY;
- approved acquisition-record location: TODO-VERIFY;
- checksum verification procedure: TODO-VERIFY;
- explicit statement that download does not replace current PLZ centroid data: TODO-VERIFY;
- explicit statement that download does not approve centroid derivation: TODO-VERIFY.

Current download decision:

Blocked. No asset download is approved by this document.

## Blockers Before Centroid Derivation

Stop before centroid derivation unless all of the following are resolved:

- approved legal/storage/redistribution decision for ODbL-derived geometry: TODO-VERIFY;
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

Current centroid derivation decision:

Blocked. No centroid derivation is approved by this document.

## Remaining TODO-VERIFY

- legal suitability for ODbL use in this project;
- OSM attribution wording and sufficiency;
- `yetzt/postleitzahlen` attribution wording and sufficiency;
- repository storage approval;
- raw asset redistribution approval;
- derived geometry or centroid redistribution approval;
- derivative database classification for any future centroid output;
- share-alike and derivative-data obligations;
- approved raw storage path;
- approved acquisition-record location;
- checksum verification after any future approved download;
- schema and layer inspection after any future approved download;
- geometry CRS, validity, duplicate PLZ, and missing PLZ QA;
- NRW filtering method and QA;
- centroid derivation method and precision policy;
- PLZ-to-municipality and AGS limitations;
- temporal availability, publication lag, revision lag, and backfill behavior;
- causal availability and leakage review;
- lineage from `yetzt/postleitzahlen` or the pinned asset to current `plz_centroids_nrw.csv`: No reliable evidence.

## Current Decision

Blocked / TODO-VERIFY / No reliable evidence / Non-final.

This document authorizes no data download, no asset storage, no redistribution, no centroid derivation, no PLZ centroid replacement, no mapping use, no modelling, no validation, and no predictive, operational, or business value claim.
