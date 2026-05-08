# External Source Documentation

## Scope

This document records source documentation for high-priority external data and reference mapping artifacts in the Food Prediction project.

Current phase only:
- External Data Acquisition
- Reference Mapping
- Source QA

This document does not perform ML integration, model training, model comparison, SHAP, feature importance, deployment, Streamlit work, business recommendations, feature-value claims, forecast-improvement claims, or operational-benefit claims.

Canonical raw data remain exactly:
- sales
- stores
- weather
- holidays

External data remain candidate enrichment data only.

Predictive value status: TODO-VERIFY.

Conclusion on predictive value: No reliable evidence.

## Evidence Basis

This document is based only on repository evidence available in the current branch:
- verified branch for this update: feature/next-external-source-qa
- raw_data/code_external_data/external_source_qa_registry.csv
- raw_data/code_external_data/external_source_evidence_audit.md
- raw_data/code_external_data/build_municipality_census_feature_base.py
- raw_data/code_external_data/fill_municipality_census_population_area.py
- raw_data/code_external_data/build_zipcode_to_municipality_nrw_csv.py
- raw_data/code_external_data/build_store_municipality_reference.py
- existing QA summaries and headers under raw_data/code_external_data

No web research was used.

Do not treat TODO-VERIFY fields as resolved unless later supported by source evidence, file evidence, reproducible checks, or explicit user confirmation.

## Source 1: Census Workbook Candidate

| Field | Status |
|---|---|
| Source name | TODO-VERIFY |
| Source file or path | raw_data/code_external_data/census_raw/destatis_gvisys_31122024.xlsx |
| Source URL | TODO-VERIFY |
| Access or creation date | TODO-VERIFY |
| License or usage terms | TODO-VERIFY |
| Spatial level | municipality candidate context, TODO-VERIFY |
| Temporal level | TODO-VERIFY |
| Update logic | TODO-VERIFY |
| Join key | municipality_ags, TODO-VERIFY |
| Known limitations | workbook provenance, source URL, access date, license, publication lag, revision lag, and causal availability are unresolved |
| QA status | file exists; source documentation incomplete |
| Current phase status | candidate external enrichment only |
| Predictive value status | TODO-VERIFY |

### Notes

The repository contains a local workbook at the documented path. The local file presence does not prove source validity, source URL, license status, publication lag, revision lag, causal availability, leakage safety, or predictive value.

## Source 2: Municipality Population and Area Source CSV

| Field | Status |
|---|---|
| Source name | TODO-VERIFY |
| Source file or path | raw_data/code_external_data/census_raw/municipality_population_area_source.csv |
| Source URL | TODO-VERIFY |
| Access or creation date | TODO-VERIFY |
| License or usage terms | TODO-VERIFY |
| Spatial level | municipality |
| Temporal level | TODO-VERIFY |
| Update logic | generated or updated from local Destatis workbook by fill_municipality_census_population_area.py |
| Join key | municipality_ags |
| Known limitations | workbook provenance, source URL, access date, license, and source update behavior remain TODO-VERIFY |
| QA status | header verified; AGS handling present in script; full source documentation incomplete |
| Current phase status | candidate external enrichment only |
| Predictive value status | TODO-VERIFY |

### Notes

The script normalizes municipality_ags to 8 characters and checks duplicate or unresolved target municipalities. This supports basic technical handling of the join key, but it does not fully verify source authority, license, temporal availability, or leakage safety.

## Source 3: Municipality Census Raw CSV

| Field | Status |
|---|---|
| Source name | TODO-VERIFY |
| Source file or path | raw_data/code_external_data/census_raw/municipality_census_raw.csv |
| Source URL | TODO-VERIFY |
| Access or creation date | TODO-VERIFY |
| License or usage terms | TODO-VERIFY |
| Spatial level | municipality |
| Temporal level | reference date 2022-05-15 appears in build_municipality_census_feature_base.py; source temporal validity remains TODO-VERIFY |
| Update logic | used by build_municipality_census_feature_base.py |
| Join key | municipality_ags |
| Known limitations | source provenance, license, temporal availability, publication lag, revision lag, and causal availability remain TODO-VERIFY |
| QA status | header verified; script checks missing, invalid, and duplicate municipality_ags |
| Current phase status | candidate external enrichment only |
| Predictive value status | TODO-VERIFY |

### Notes

The script records RAW_SOURCE_REFERENCE_DATE = 2022-05-15 and MAX_ALLOWED_REFERENCE_DATE = 2025-06-30. This is repository metadata only. It does not prove source access date, publication lag, revision lag, or causal availability at prediction time.

## Source 4: BKG VG250 Boundary Cache

| Field | Status |
|---|---|
| Source name | BKG VG250 boundary dataset |
| Source file or path | raw_data/code_external_data/_reference_geo/vg250_cache/DE_VG250.gpkg |
| Source URL | https://daten.gdz.bkg.bund.de/produkte/vg/vg250_ebenen_0101/aktuell/vg250_01-01.utm32s.gpkg.ebenen.zip |
| Access or creation date | TODO-VERIFY |
| License or usage terms | TODO-VERIFY |
| Spatial level | municipality boundary |
| Temporal level | TODO-VERIFY |
| Update logic | script can download or refresh VG250 boundary cache if executed; no execution in this documentation step |
| Join key | municipality_ags, municipality_ars |
| Known limitations | license, boundary version, access date, NRW boundary consistency, and source update timing remain TODO-VERIFY |
| QA status | file exists; script filters NRW by AGS prefix 05 and normalizes AGS to width 8 and ARS to width 12 |
| Current phase status | reference mapping candidate |
| Predictive value status | TODO-VERIFY |

### Notes

The script uses the VG250 municipality layer `vg250_gem` and filters NRW municipalities by AGS prefix 05. This supports reference mapping work, but it does not fully resolve license status, boundary version, source access date, or NRW boundary consistency.

## Source 5: NRW PLZ Centroids

| Field | Status |
|---|---|
| Source name | TODO-VERIFY |
| Source file or path | raw_data/code_external_data/_reference_geo/plz_centroids_nrw.csv |
| Source URL | TODO-VERIFY |
| Access or creation date | TODO-VERIFY |
| License or usage terms | TODO-VERIFY |
| Spatial level | ZIP or postcode centroid |
| Temporal level | TODO-VERIFY |
| Update logic | TODO-VERIFY |
| Join key | zip, lat, lng |
| Known limitations | centroid approximation, coordinate source quality, source URL, license, and update logic remain TODO-VERIFY |
| QA status | header verified; script checks valid ZIPs and duplicate ZIP coordinate pairs |
| Current phase status | reference coordinate candidate |
| Predictive value status | TODO-VERIFY |

### Notes

ZIP centroids are approximate reference points. They must not be treated as exact store coordinates or exact municipality membership proof without documented allocation logic and QA evidence.

## Source 6: ZIP to Municipality Reference

| Field | Status |
|---|---|
| Source name | derived ZIP-to-municipality reference candidate |
| Source file or path | raw_data/code_external_data/_reference_geo/zipcode_to_municipality_nrw.csv |
| Source URL | derived from PLZ centroids and VG250 boundary logic; upstream source URLs remain partly TODO-VERIFY |
| Access or creation date | TODO-VERIFY |
| License or usage terms | TODO-VERIFY |
| Spatial level | ZIP to municipality reference |
| Temporal level | TODO-VERIFY |
| Update logic | generated by build_zipcode_to_municipality_nrw_csv.py |
| Join key | zipcode, municipality_ags |
| Known limitations | ZIP/postcode ambiguity, centroid approximation, multi-municipality ZIPs, tie handling, license, and boundary consistency remain TODO-VERIFY |
| QA status | header verified; script checks duplicate ZIPs and non-NRW municipality keys |
| Current phase status | reference mapping candidate |
| Predictive value status | TODO-VERIFY |

### Mapping Risk

ZIP/postcode must not be treated as a one-to-one municipality key in general. The script documents deterministic tie handling for cases where one centroid intersects multiple municipality polygons by choosing the lexicographically smallest municipality_ags. This remains a mapping risk unless later resolved by explicit allocation logic and QA evidence.

### ZIP/AGS QA Update

Read-only QA on branch `feature/zipcode-ags-reference-qa` observed the current ZIP-to-municipality file has 864 rows, 864 unique ZIPs, 0 duplicate ZIP row groups, and 864 rows with `assignment_method = polygon_intersects`.

The current output showed 0 lexicographic resolution rows and 0 nearest municipality fallback rows. This is a current-file observation only. It does not prove true one-to-one ZIP-to-municipality mapping.

Source truth for multi-municipality ZIPs in repository evidence: No reliable evidence.

Multi-municipality ZIP ambiguity remains TODO-VERIFY.

Current AGS/Gemeindeschluessel shape checks found 0 invalid 8-digit `municipality_ags` values and 0 non-NRW-prefix values in the ZIP map. Authoritative municipality identity remains TODO-VERIFY.

## Source 7: Store Municipality Reference

| Field | Status |
|---|---|
| Source name | derived store-to-municipality reference candidate |
| Source file or path | raw_data/code_external_data/_external_data/store_geography/store_municipality_reference.csv |
| Source URL | derived from canonical stores and ZIP-to-municipality reference; source URL not applicable for derived file, upstream source references remain TODO-VERIFY |
| Access or creation date | TODO-VERIFY |
| License or usage terms | TODO-VERIFY |
| Spatial level | store to municipality |
| Temporal level | TODO-VERIFY |
| Update logic | generated by build_store_municipality_reference.py |
| Join key | store_id, store_zipcode, municipality_ags |
| Known limitations | current QA reports no valid store coordinates and ZIP fallback for all stores |
| QA status | QA summary exists; current output reports 84 stores, 0 valid coordinates, 84 ZIP fallback assignments, 0 unassigned stores |
| Current phase status | reference mapping candidate |
| Predictive value status | TODO-VERIFY |

### Notes

The current store municipality reference is not based on valid store coordinate spatial joins. It uses ZIP fallback for all stores according to the QA summary. This is acceptable as documented reference mapping work, but it must remain limited and must not be treated as precise geospatial truth. Spatial assignment quality remains TODO-VERIFY.

### Store Fallback QA Update

Read-only QA on branch `feature/zipcode-ags-reference-qa` observed:
- `assignment_method = zipcode_fallback_no_valid_coordinates`: 84 rows
- `qa_has_valid_coordinates = False`: 84 rows
- `qa_assignment_used_fallback = True`: 84 rows
- rows unassigned or missing ZIP reference: 0

This confirms the current store municipality reference depends on ZIP fallback for all stores. ZIP fallback can create false precision if treated as verified store-level municipality truth.

Store coordinate source quality remains TODO-VERIFY.

Spatial assignment quality remains TODO-VERIFY.

OSM remains deferred until store coordinate quality is verified.

## ZIP/AGS Reference QA Artifact

A dedicated non-final QA artifact records the current ZIP, AGS, centroid, VG250, and store fallback findings:
- raw_data/code_external_data/zipcode_ags_reference_qa.md

This artifact is source and reference mapping QA only. It does not validate predictive value, forecast improvement, operational benefit, model impact, or ML readiness.

## Deferred or Out-of-Scope Sources

The following sources remain deferred or registry-only in the current phase:
- OSM POI context: priority 2 but deferred until store coordinate quality is verified; current store geography QA reports 84 stores, 0 valid coordinates, and 84 ZIP fallback assignments. Source, license, lineage, leakage, and identical-coordinate effects also remain TODO-VERIFY.
- OpenLigaDB or event-like data: low-priority deferred context only.
- Bahn-Vorhersage downloader context: low-priority deferred context only.
- Zensus grid downloader stub: high-priority domain but deferred until source, license, lineage, file existence, QA, and causal availability are verified.
- VRR GTFS downloader stub: priority 5 transit context only, deferred until source, license, lineage, temporal coverage, QA, and causal availability are verified.

Do not activate, repair, execute, or prioritize deferred downloaders without explicit user approval.

## Leakage and Causal Availability Status

The following remain TODO-VERIFY for all listed external or reference artifacts unless later documented with source evidence:
- publication lag
- revision lag
- backfilled values
- post-event corrections
- update timing
- historical availability
- prediction-time availability
- causal availability
- leakage risk

## Remaining TODO-VERIFY Items

The following items remain unresolved:
- source URL or stable source reference for Census workbook
- access or creation date for external source material
- license or usage terms
- raw source provenance
- file lineage completeness
- publication lag
- revision lag
- backfill behavior
- post-event correction behavior
- temporal availability
- causal availability at prediction time
- leakage risk
- ZIP/postcode ambiguity
- multi-municipality ZIP handling
- AGS/Gemeindeschluessel identity
- AGS/Gemeindeschluessel format validation against source authority
- AGS/Gemeindeschluessel leading-zero preservation against source authority
- NRW boundary consistency
- ZIP centroid source quality
- centroid approximation risk
- store coordinate source quality
- duplicate coordinate risk
- identical OSM features caused by identical or centroid-derived coordinates
- predictive value
- feature value
- model impact
- forecast improvement
- operational benefit
- business benefit

## QA and Update Rules

Before changing any TODO-VERIFY status, require direct evidence from:
- repository files,
- source files,
- source documentation,
- reproducible checks,
- command output,
- or explicit user confirmation.

Do not resolve TODO-VERIFY by assumption.

Do not use this document to claim predictive value, forecast improvement, operational benefit, or business benefit.
