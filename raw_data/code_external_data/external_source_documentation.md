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

This document is based only on repository evidence available in the current post-PR30 branch state verified for the 2026-05-11 `store_municipality_reference` CSV/parquet registry/source-documentation consistency update:
- verified branch for this update: master
- verified HEAD for this update: 6cd6fa263cb4f66db7351ecda5f73a4d69745bca
- verified working tree for this update: clean before edits
- raw_data/code_external_data/external_source_qa_registry.csv
- raw_data/code_external_data/external_source_evidence_audit.md
- raw_data/code_external_data/build_municipality_census_feature_base.py
- raw_data/code_external_data/fill_municipality_census_population_area.py
- raw_data/code_external_data/build_zipcode_to_municipality_nrw_csv.py
- raw_data/code_external_data/build_store_municipality_reference.py
- raw_data/code_external_data/_external_data/store_geography/store_municipality_reference.csv
- raw_data/code_external_data/_external_data/store_geography/store_municipality_reference.parquet
- raw_data/code_external_data/_external_data/store_geography/store_municipality_reference_qa_summary.csv
- existing QA summaries and headers under raw_data/code_external_data
- official public source pages listed below

Official public source pages checked on 2026-05-08:
- Destatis GV-ISys Gemeindeverzeichnis: https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/_inhalt.html
- Destatis 31.12.2024 GV-ISys publication page: https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/Administrativ/Archiv/GVAuszugQ/AuszugGV4QAktuell.html
- Destatis standard copyright terms: https://www.destatis.de/DE/Service/Impressum/copyright-allgemein.html
- BKG VG250 product page: https://gdz.bkg.bund.de/index.php/default/open-data/verwaltungsgebiete-1-250-000-stand-01-01-vg250-01-01.html
- BKG PLZ product page, context only: https://gdz.bkg.bund.de/index.php/default/postleitzahlgebiete-deutschland-plz.html

Do not treat TODO-VERIFY fields as resolved unless later supported by source evidence, file evidence, reproducible checks, or explicit user confirmation.

## Source 1: Census Workbook Candidate

| Field | Status |
|---|---|
| Registry ID | `census_destatis_workbook` |
| Source name | Destatis GV-ISys Alle politisch selbständigen Gemeinden mit ausgewählten Merkmalen am 31.12.2024 (4. Quartal) |
| Source file or path | raw_data/code_external_data/census_raw/destatis_gvisys_31122024.xlsx |
| Source URL | https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/Administrativ/Archiv/GVAuszugQ/AuszugGV4QAktuell.html |
| Access or creation date | official source page checked 2026-05-08; local workbook modified timestamp remains repository file metadata only |
| License or usage terms | Reuse permitted with source attribution under Destatis standard copyright terms, unless product-specific third-party or deviating rights apply. Product-specific third-party or deviating rights remain TODO-VERIFY. |
| Spatial level | municipality |
| Temporal level | official product reference 2024-12-31; official page date 2025-01-07; local workbook metadata title says Gebietsstand 31.12.2024 and Stand August 2025 |
| Update logic | local workbook is read by fill_municipality_census_population_area.py; no download or refresh was performed in this audit |
| Join key | municipality_ags |
| Known limitations | product-specific third-party rights, local workbook acquisition path, publication lag, revision lag, and causal availability remain TODO-VERIFY |
| QA status | file exists; xlsx core metadata title references the 31.12.2024 GV-ISys product; script parses Satzart 60 municipality rows |
| Current phase status | candidate external enrichment only |
| Predictive value status | TODO-VERIFY |

### Notes

The repository contains a local workbook at the documented path. Official Destatis pages document the GV-ISys product and standard reuse terms, but repository evidence does not prove the exact acquisition event for the local workbook. The local workbook metadata shows a modified timestamp after the official page date, so publication lag and revision lag remain TODO-VERIFY.

## Source 2: Municipality Population and Area Source CSV

| Field | Status |
|---|---|
| Registry ID | `census_population_area_source_csv` |
| Source name | Destatis GV-ISys derived municipality population and area source CSV |
| Source file or path | raw_data/code_external_data/census_raw/municipality_population_area_source.csv |
| Source URL | derived from local Destatis GV-ISys workbook; official source page checked 2026-05-08 |
| Access or creation date | official source page checked 2026-05-08; local file timestamp remains repository file metadata only |
| License or usage terms | Reuse permitted with source attribution under Destatis standard copyright terms, unless product-specific third-party or deviating rights apply. Product-specific third-party or deviating rights remain TODO-VERIFY. |
| Spatial level | municipality |
| Temporal level | source workbook product reference 2024-12-31; revision lag remains TODO-VERIFY |
| Update logic | generated or updated from local Destatis workbook by fill_municipality_census_population_area.py |
| Join key | municipality_ags |
| Known limitations | local workbook acquisition event, product-specific third-party rights, source update behavior, publication lag, revision lag, and causal availability remain TODO-VERIFY |
| QA status | header verified; AGS handling present in script; full source documentation incomplete |
| Current phase status | candidate external enrichment only |
| Predictive value status | TODO-VERIFY |

### Notes

The script normalizes municipality_ags to 8 characters and checks duplicate or unresolved target municipalities. This supports basic technical handling of the join key, but it does not fully verify source authority, license, temporal availability, or leakage safety.

## Source 3: Municipality Census Raw CSV

| Field | Status |
|---|---|
| Registry ID | `census_raw_municipality_csv` |
| Source name | Destatis GV-ISys derived municipality census raw CSV |
| Source file or path | raw_data/code_external_data/census_raw/municipality_census_raw.csv |
| Source URL | derived from local Destatis GV-ISys workbook; official source page checked 2026-05-08 |
| Access or creation date | official source page checked 2026-05-08; local file timestamp remains repository file metadata only |
| License or usage terms | Reuse permitted with source attribution under Destatis standard copyright terms, unless product-specific third-party or deviating rights apply. Product-specific third-party or deviating rights remain TODO-VERIFY. |
| Spatial level | municipality |
| Temporal level | source workbook product reference 2024-12-31; build_municipality_census_feature_base.py still records RAW_SOURCE_REFERENCE_DATE = 2022-05-15 for derived feature outputs, so temporal metadata consistency remains TODO-VERIFY |
| Update logic | used by build_municipality_census_feature_base.py |
| Join key | municipality_ags |
| Known limitations | local workbook acquisition event, product-specific third-party rights, temporal metadata consistency, publication lag, revision lag, and causal availability remain TODO-VERIFY |
| QA status | header verified; script checks missing, invalid, and duplicate municipality_ags |
| Current phase status | candidate external enrichment only |
| Predictive value status | TODO-VERIFY |

### Notes

The script records RAW_SOURCE_REFERENCE_DATE = 2022-05-15 and MAX_ALLOWED_REFERENCE_DATE = 2025-06-30. That derived-script metadata conflicts with the official 2024-12-31 source product reference and must not be silently treated as resolved.

## Source 3a: Derived Census Feature Base Artifacts

| Field | Status |
|---|---|
| Registry ID | `census_municipality_feature_base`; `census_store_feature_base` |
| Source name | Destatis GV-ISys derived municipality and store census feature bases |
| Source file or path | raw_data/code_external_data/_external_data/census_features/municipality_census_feature_base.csv; raw_data/code_external_data/_external_data/census_features/store_census_feature_base.csv |
| Source URL | derived from local Destatis GV-ISys workbook; official source page checked 2026-05-08 |
| Access or creation date | official source page checked 2026-05-08; derived artifact timestamps remain repository file metadata only |
| License or usage terms | Reuse permitted with source attribution under Destatis standard copyright terms, unless product-specific third-party or deviating rights apply. Product-specific third-party or deviating rights remain TODO-VERIFY. |
| Spatial level | municipality; store-to-municipality candidate context |
| Temporal level | registry preserves script metadata reference date 2022-05-15; source workbook product reference is 2024-12-31; consistency remains TODO-VERIFY |
| Update logic | generated by build_municipality_census_feature_base.py from municipality_census_raw.csv and store_municipality_reference |
| Join key | municipality_ags; store_id |
| Known limitations | source-date conflict, name mismatch interpretation, publication lag, revision lag, causal availability, mapping quality, and predictive value remain TODO-VERIFY |
| QA status | QA summary reports 25 municipality feature rows, 19 municipality rows with name mismatch, 6 municipality rows without name mismatch, 84 store rows, and 0 store rows with missing municipality features |
| Current phase status | candidate external enrichment only |
| Predictive value status | TODO-VERIFY |

### Notes

These derived artifacts remain candidate enrichment only. The script metadata reference date must not be used to override the official GV-ISys workbook reference date without explicit source correction.

Read-only row-level review of the existing municipality census feature base found that the observed name-mismatch pattern is official/display municipality names with designation suffixes versus shorter store-reference municipality names. This documents the mismatch pattern only. It does not prove authoritative municipality identity, resolve AGS/Gemeindeschluessel authority checks, or resolve leading-zero preservation against an authoritative source.

## Source 4: BKG VG250 Boundary Cache

| Field | Status |
|---|---|
| Registry ID | `vg250_boundary_cache` |
| Source name | BKG Verwaltungsgebiete 1:250 000, `VG250_3112` / VG250 01.01. |
| Source file or path | raw_data/code_external_data/_reference_geo/vg250_cache/DE_VG250.gpkg |
| Source URL | product page: https://gdz.bkg.bund.de/index.php/default/open-data/verwaltungsgebiete-1-250-000-stand-01-01-vg250-01-01.html; direct URL in script: https://daten.gdz.bkg.bund.de/produkte/vg/vg250_ebenen_0101/aktuell/vg250_01-01.utm32s.gpkg.ebenen.zip |
| Access or creation date | official source page checked 2026-05-08; official BKG MIS metadata record date `05.05.2026`; local cache file timestamp remains repository file metadata only |
| License or usage terms | Datenlizenz Deutschland Namensnennung 2.0 from official BKG metadata; BKG source attribution and linked license/source notice required for public/external use; local file presence does not independently prove license status |
| Spatial level | municipality boundary |
| Temporal level | official BKG MIS metadata records `Letzte Änderung 31.12.2024`; official product page reports current reference 2025-01-01 and annual update cycle; local cache version/reference date remains TODO-VERIFY |
| Update logic | script can download or refresh VG250 boundary cache if executed; no execution in this documentation step |
| Join key | municipality_ags, municipality_ars |
| Known limitations | official BKG MIS metadata supports source metadata facts only; local cache version, CRS/layer integrity, NRW boundary consistency, and prediction-time suitability remain TODO-VERIFY |
| QA status | file exists; script filters NRW by AGS prefix 05 and normalizes AGS to width 8 and ARS to width 12 |
| Current phase status | reference mapping candidate |
| Predictive value status | TODO-VERIFY |

### Notes

The script uses the VG250 municipality layer `vg250_gem` and filters NRW municipalities by AGS prefix 05. Official BKG documentation resolves the public product page, license family, attribution requirement, and annual update cycle, but it does not prove the exact version of the local cached GeoPackage.

Official BKG/MIS metadata evidence is limited to source metadata facts:

- MIS docuuid `431406f6-1b31-48a9-b6db-dc4b38caf5ea`.
- Product identifier `VG250_3112`.
- Official `Letzte Änderung 31.12.2024`.
- Metadata record date `05.05.2026`.
- CRS/EPSG metadata `25832`.
- License family `Datenlizenz Deutschland Namensnennung 2.0`.

This official metadata does not prove that the local `DE_VG250.gpkg` cache is complete, unmodified, geometrically valid, CRS-transformation correct, fully NRW-boundary consistent, or fully equivalent to the official product.

Read-only SQLite metadata evidence from the local GeoPackage supports only technical file observations:

- Standard-library sqlite3 inspection completed successfully against the local GeoPackage.
- GeoPackage metadata tables are present, including `gpkg_contents`, `gpkg_geometry_columns`, `gpkg_spatial_ref_sys`, `gpkg_metadata`, and `gpkg_metadata_reference`.
- `gpkg_contents` declares VG250 feature layers with `srs_id = 25832`; local `gpkg_contents.description` date-like values show `2025-01-01`, but their exact meaning remains TODO-VERIFY.
- `gpkg_geometry_columns` declares VG250 geometry layers including `vg250_gem`, `vg250_krs`, `vg250_lan`, `vg250_li`, `vg250_pk`, `vg250_rbz`, `vg250_sta`, and `vg250_vwg`; `vg250_gem` is declared as `MULTIPOLYGON` with `srs_id = 25832`.
- `gpkg_spatial_ref_sys` lists EPSG:25832 / ETRS89 / UTM zone 32N.
- SQLite row counts observed include `vg250_gem = 11103`, `vg250_krs = 433`, `vg250_lan = 34`, `vg250_rbz = 21`, `vg250_sta = 11`, and `vg250_vwg = 4680`.
- `vg250_gem` contains municipality identity candidate columns including `AGS`, `ARS`, `GEN`, `BEZ`, `SN_L`, `SN_R`, `SN_K`, `NUTS`, and `geom`.
- Local SQLite checks found `vg250_gem` has 11103 rows, no null or blank `AGS` values, all checked `AGS` values are 8 characters, and no non-digit `AGS` values were observed.
- For the `SN_L = '05'` NRW subset, SQLite checks found 396 rows, 396 distinct `AGS`, 396 distinct `ARS`, 0 duplicate `AGS`, 0 duplicate `ARS`, and AGS/SN_L prefix consistency for the checked predicates.
- `gpkg_metadata` points to BKG-MIS metadata URL `https://mis.bkg.bund.de/trefferanzeige?docuuid=431406f6-1b31-48a9-b6db-dc4b38caf5ea`.
- Local `gpkg_metadata_reference` timestamps such as `2025-07-01T10:34:48Z` or `2025-07-01T10:34:49Z` are local GeoPackage metadata-reference evidence only, not an official VG250 reference date by themselves.

These SQLite observations and official metadata references do not prove actual geometry validity, calculated spatial bounds, CRS transformation correctness, full layer integrity, full NRW boundary consistency, authoritative AGS/Gemeindeschluessel identity, source-authority leading-zero validation, exact official local cache version or reference date, ZIP-to-municipality correctness, causal availability, leakage safety, predictive value, forecast improvement, feature value, model impact, operational benefit, or business benefit. `BEGINN` and `WSK` contain multiple historical or administrative date values and must not be interpreted as one uniform dataset reference date without official documentation. No reliable evidence supports value or benefit claims from this metadata inspection.

## Source 5: NRW PLZ Centroids

| Field | Status |
|---|---|
| Registry ID | `nrw_plz_centroids` |
| Source name | TODO-VERIFY; upstream source identity: No reliable evidence. |
| Source file or path | raw_data/code_external_data/_reference_geo/plz_centroids_nrw.csv |
| Source URL | TODO-VERIFY; stable upstream URL, download path, or source repository: No reliable evidence. |
| Access or creation date | TODO-VERIFY; repository evidence shows local file presence and repository-addition commit date only, not upstream access, creation, or reference date |
| License or usage terms | TODO-VERIFY; local PLZ centroid license: No reliable evidence. |
| Spatial level | ZIP or postcode centroid |
| Temporal level | TODO-VERIFY; reference date, publication lag, revision lag, and backfill behavior: No reliable evidence. |
| Update logic | TODO-VERIFY; historical helper evidence supports only partial local subset logic, not reproducible upstream acquisition |
| Join key | zip, lat, lng |
| Known limitations | centroid approximation, coordinate source quality, source URL, license, lineage, precision, update logic, temporal availability, causal availability, leakage risk, NRW boundary consistency, and ZIP/postcode mapping ambiguity remain TODO-VERIFY |
| QA status | schema `zip,lat,lng`; current file has 864 rows, 864 unique ZIPs, 0 duplicate ZIP groups, 0 duplicate coordinate groups, 0 missing coordinates, 0 nonnumeric coordinates, and 0 broad NRW-bounds outliers |
| Repository lineage evidence | Git history adds `plz_centroids_nrw.csv` in commit `9be4a742a538d22e2ed6b98c278b490ec5b4a40f` on 2026-03-09; this is repository-addition evidence only, not upstream access or reference-date evidence; the historical helper script `make_plz_centroids_nrw_subset_from_list.py` supports only partial local subset lineage from a local `plz_centroids.csv` and a user-provided NRW PLZ list; no tracked upstream `plz_centroids.csv` history was found; upstream centroid source remains TODO-VERIFY |
| Current phase status | reference coordinate candidate |
| Predictive value status | TODO-VERIFY |

### Notes

ZIP centroids are approximate reference points. They must not be treated as exact store coordinates or exact municipality membership proof without documented allocation logic and QA evidence.

Official BKG PLZ product context was checked on 2026-05-08 at https://gdz.bkg.bund.de/index.php/default/postleitzahlgebiete-deutschland-plz.html. That page describes a restricted product requiring a license agreement and does not prove the lineage of the local `plz_centroids_nrw.csv`; local PLZ centroid provenance remains TODO-VERIFY.

Current-file QA is structural only. The row count, unique-ZIP count, duplicate checks, missing-coordinate check, numeric-coordinate check, and broad NRW-bounds sanity check do not prove source identity, license, coordinate method, precision, temporal availability, causal availability, leakage safety, NRW boundary consistency, or true ZIP-to-municipality validity.

### Upstream Source Research Outcome

Read-only upstream source research on 2026-05-09 reviewed official or source-linked candidate pages only. BKG Postleitzahlgebiete Deutschland remains restricted context only, not local lineage. OpenPLZ and the OpenPLZ API data GitHub repository remain plausible candidates only, not proven local lineage. Open.NRW / CKAN Deutschland Postleitzahlen remains a plausible candidate only, not proven local lineage. OpenStreetMap copyright/license evidence provides ODbL context only, not local lineage.

No candidate source is tied by repository evidence to the local `plz_centroids.csv` or `plz_centroids_nrw.csv`. No tracked Git history was found for local `plz_centroids.csv`. Same-commit scripts and notebook references use PLZ centroids as an input but do not prove source URL, license, access date, upstream provenance, precision, coordinate method, or reference date.

The local `plz_centroids.csv` is proven only as a historical helper input, not as a sourced or provenanced artifact. Upstream provenance conclusion for `plz_centroids_nrw.csv`: No reliable evidence.

Repository evidence does not resolve the PLZ centroid source name, source URL, upstream access date, license or usage terms, upstream provenance, precision, coordinate quality, reference date, update logic, temporal availability, causal availability, publication lag, revision lag, leakage risk, mapping quality, or predictive value. These fields remain TODO-VERIFY.

### Public Source Candidate: yetzt/postleitzahlen

Direct source inspection on 2026-05-19 reviewed `https://github.com/yetzt/postleitzahlen`, `datapackage.json`, `opendata.json`, and the releases page. The repository is a public source candidate for German PLZ/postcode geometry provenance review only.

Observed candidate-source facts: the repository describes German postcode area shapes in compressed GeoJSON and TopoJSON formats, with extraction from OpenStreetMap via Overpass; metadata records `license = ODbL-1.0`, describes German postcode areas 2025, and lists `data/postleitzahlen.geojson` and `data/postleitzahlen.topojson`; the releases page shows release `2026.02` dated 2026-02-20 with four assets.

Boundary: no repository evidence ties `yetzt/postleitzahlen` to the local `plz_centroids.csv` historical helper input or to `raw_data/code_external_data/_reference_geo/plz_centroids_nrw.csv`. Local lineage, license suitability for this project, reproducible acquisition path, reference-date alignment, geometry quality, coordinate derivation method, temporal availability, causal availability, leakage review, and ZIP/postcode-to-municipality mapping quality remain TODO-VERIFY.

### Source-Selection Requirements

Any later PLZ centroid source-selection or replacement preparation must document source identity, stable URL or repository, license or usage terms, access date, spatial level, temporal or reference level, update logic, coordinate method, coordinate precision, join keys, limitations, raw lineage plan, QA status, causal availability, and leakage review before any download, regeneration, replacement, or promotion is considered. This documentation-only note does not select, validate, replace, or promote a PLZ source or derived ZIP-to-municipality mapping.

Until those requirements are supported by direct evidence, PLZ centroid source quality, ZIP-to-municipality mapping correctness, predictive value, operational value, and business value remain: No reliable evidence.

## Source 6: ZIP to Municipality Reference

| Field | Status |
|---|---|
| Registry ID | `zipcode_to_municipality_reference` |
| Source name | derived ZIP-to-municipality reference candidate |
| Source file or path | raw_data/code_external_data/_reference_geo/zipcode_to_municipality_nrw.csv |
| Source URL | derived from PLZ centroids and VG250 boundary logic; VG250 official source checked 2026-05-08; PLZ centroid upstream source remains TODO-VERIFY |
| Access or creation date | TODO-VERIFY |
| License or usage terms | VG250 license resolved as Datenlizenz Deutschland Namensnennung 2.0; PLZ centroid license remains TODO-VERIFY, so derived artifact license status remains partial/TODO-VERIFY |
| Spatial level | ZIP to municipality reference |
| Temporal level | TODO-VERIFY |
| Update logic | generated by build_zipcode_to_municipality_nrw_csv.py |
| Join key | zipcode, municipality_ags, municipality_ars; assignment_method documents assignment path |
| Known limitations | ZIP/postcode ambiguity, centroid approximation, multi-municipality ZIPs, tie handling, PLZ provenance/license, and boundary consistency remain TODO-VERIFY |
| QA status | current output has 864 rows, 864 unique ZIPs, 0 duplicate ZIP row groups, 864 `polygon_intersects` rows, 0 lexicographic resolution rows, 0 nearest municipality fallback rows, 0 invalid 8-digit `municipality_ags` values, 0 non-NRW AGS prefixes, 0 invalid 12-digit `municipality_ars` values, and 0 non-`05` federal_state_code values; local read-only coverage-gap QA observed 396 local VG250 NRW municipalities, 394 ZIP-map municipalities, and two local VG250 NRW municipalities not referenced by the ZIP map |
| Current phase status | reference mapping candidate |
| Predictive value status | TODO-VERIFY |

### Mapping Risk

ZIP/postcode must not be treated as a one-to-one municipality key in general. The script assigns centroids by point-in-polygon/intersects first, resolves multiple polygon hits by choosing the lexicographically smallest `municipality_ags`, and uses nearest municipality fallback only for unmatched centroids. These paths remain mapping risks unless later resolved by explicit allocation logic and QA evidence.

### ZIP/AGS QA Update

Read-only QA on branch `feature/zipcode-ags-reference-qa` observed the current ZIP-to-municipality file has 864 rows, 864 unique ZIPs, 0 duplicate ZIP row groups, and 864 rows with `assignment_method = polygon_intersects`.

Post-PR29 read-only verification on `master` at HEAD `586460e1e85647d346901e8c76d924dab1d53e95` confirmed the same current-file ZIP metrics.

Additional local read-only coverage-gap QA observed 396 local VG250 NRW municipalities and 394 ZIP-map municipalities. The observed local VG250 NRW municipalities not referenced by the ZIP map were `05370024 / Selfkant` and `05370032 / Waldfeucht`. The same observation recorded 8 Heinsberg ZIP-map context rows, all with `assignment_method = polygon_intersects`.

This coverage-gap observation is local read-only QA evidence only. It does not prove the ZIP-to-municipality mapping is correct or wrong. The cause and correctness of the Selfkant/Waldfeucht gap remain TODO-VERIFY. PLZ centroid provenance/license and ZIP-to-municipality truth remain TODO-VERIFY.

The current output showed 0 lexicographic resolution rows and 0 nearest municipality fallback rows. This is a current-file observation only. It does not prove true one-to-one ZIP-to-municipality mapping.

Source truth for multi-municipality ZIPs in repository evidence: No reliable evidence.

Multi-municipality ZIP ambiguity remains TODO-VERIFY.

Current AGS/Gemeindeschluessel shape checks found 0 invalid 8-digit `municipality_ags` values, 0 non-NRW-prefix AGS values, 0 invalid 12-digit `municipality_ars` values, and 0 non-`05` federal_state_code values in the ZIP map. Authoritative municipality identity and leading-zero preservation against a source authority remain TODO-VERIFY.

## Source 7: Store Municipality Reference

| Field | Status |
|---|---|
| Registry ID | `store_municipality_reference`; `store_municipality_reference_parquet` |
| Source name | derived store-to-municipality reference candidate |
| Source file or path | raw_data/code_external_data/_external_data/store_geography/store_municipality_reference.csv; raw_data/code_external_data/_external_data/store_geography/store_municipality_reference.parquet |
| Source URL | derived from canonical stores and ZIP-to-municipality reference; source URL not applicable for derived files; upstream PLZ centroid provenance remains TODO-VERIFY |
| Access or creation date | TODO-VERIFY |
| License or usage terms | upstream license status is partial: VG250 resolved, PLZ centroid license remains TODO-VERIFY |
| Spatial level | store to municipality |
| Temporal level | TODO-VERIFY |
| Update logic | generated by build_store_municipality_reference.py; script writes the final dataframe to parquet and CSV and writes the QA summary from the same final dataframe |
| Join key | store_id, store_zipcode, municipality_ags |
| Known limitations | current QA reports no valid store coordinates and ZIP fallback for all stores; parquet file presence and shared script lineage do not prove content-level parquet-to-CSV equivalence or precise store geography |
| QA status | post-PR30 QA confirms the CSV and parquet artifacts are present; QA summary reports 84 stores, 0 valid coordinates, 0 spatial join assignments, 84 ZIP fallback assignments, and 0 unassigned stores; current CSV check reports 84 rows, 84 `zipcode_fallback_no_valid_coordinates` assignments, 84 `qa_has_valid_coordinates = False` rows, 84 `qa_assignment_used_fallback = True` rows, 0 missing `store_zipcode` values, and 0 missing `municipality_ags` values |
| Current phase status | reference mapping candidate |
| Predictive value status | TODO-VERIFY |

### Notes

The current store municipality reference is not based on valid store coordinate spatial joins. It uses ZIP fallback for all stores according to the QA summary. This is acceptable as documented reference mapping work, but it must remain limited and must not be treated as precise geospatial truth. Spatial assignment quality remains TODO-VERIFY.

The parquet artifact is script-supported as an output from the same final dataframe as the CSV artifact. Current evidence supports file lineage and CSV-level QA only. Content-level parquet-to-CSV equivalence, downstream use, precise geospatial truth, causal availability, leakage risk, mapping quality, predictive value, forecast improvement, feature value, model impact, operational benefit, and business benefit remain TODO-VERIFY.

### Store Fallback QA Update

Read-only QA on branch `feature/zipcode-ags-reference-qa` observed:
- `assignment_method = zipcode_fallback_no_valid_coordinates`: 84 rows
- `qa_has_valid_coordinates = False`: 84 rows
- `qa_assignment_used_fallback = True`: 84 rows
- rows unassigned or missing ZIP reference: 0

Post-PR30 live verification on `master` at HEAD `6cd6fa263cb4f66db7351ecda5f73a4d69745bca` confirmed the current QA summary metrics and observed the current CSV output has 84 rows, 84 `zipcode_fallback_no_valid_coordinates` assignments, 84 `qa_has_valid_coordinates = False` rows, 84 `qa_assignment_used_fallback = True` rows, 0 missing `store_zipcode` values, and 0 missing `municipality_ags` values.

This confirms the current store municipality reference depends on ZIP fallback for all stores. ZIP fallback can create false precision if treated as verified store-level municipality truth.

Store coordinate source quality remains TODO-VERIFY.

Spatial assignment quality remains TODO-VERIFY.

OSM remains deferred until store coordinate quality is verified.

### Parquet Output Lineage QA Update

Post-PR30 file existence checks confirmed both current store municipality reference artifacts are present:

- `raw_data/code_external_data/_external_data/store_geography/store_municipality_reference.csv`
- `raw_data/code_external_data/_external_data/store_geography/store_municipality_reference.parquet`

Script evidence shows `build_store_municipality_reference.py` writes `final_df` to `store_municipality_reference.parquet`, writes the same `final_df` to `store_municipality_reference.csv`, and writes `build_qa_summary(final_df)` to `store_municipality_reference_qa_summary.csv`.

This supports shared generation lineage only. It does not prove content-level parquet-to-CSV equivalence if files are modified independently after generation, downstream use, source validity, mapping truth, precise geospatial truth, causal availability, leakage safety, mapping quality, predictive value, forecast improvement, feature value, model impact, operational benefit, or business benefit.

### Store Source Schema QA Update

Read-only schema QA on branch `feature/store-coordinate-source-qa` observed the canonical stores parquet at `raw_data/20260218_144523_stores.parquet` with 84 rows and 5 columns:

- `subdivision_code`
- `country_code`
- `zipcode`
- `average_weekly_revenue_Q1`
- `store_id`

No latitude, longitude, coordinate, address, street, city, or precise store-location columns were observed. Only `zipcode` and `store_id` matched the geo/address search terms used in the schema check.

This confirms there is currently no repository evidence for precise store coordinates in the canonical stores file. Store coordinate source quality remains TODO-VERIFY. ZIP fallback remains candidate reference mapping only. OSM remains deferred until a precise and documented store coordinate source is verified.

### Deferred OSM / Identical Geospatial Output QA

PR25 store coordinate source QA confirms that the canonical stores file does not provide precise coordinate, address, or location columns. Therefore the deferred OSM POI artifact cannot currently be interpreted as precise store context. Any identical OSM feature outputs, duplicate geospatial outputs, or coordinate-derived context effects remain TODO-VERIFY because the current repository-supported store geography basis is ZIP fallback only.

OSM source name, source URL, license, raw source traceability, file lineage, temporal availability, causal availability, leakage risk, mapping quality, coordinate source quality, and predictive value remain TODO-VERIFY. Source, lineage, coordinate quality, causal availability, leakage safety, predictive value, and operational value conclusion: No reliable evidence.

## ZIP/AGS Reference QA Artifact

A dedicated non-final QA artifact records the current ZIP, AGS, centroid, VG250, and store fallback findings:
- raw_data/code_external_data/zipcode_ags_reference_qa.md

This artifact is source and reference mapping QA only. It does not validate predictive value, forecast improvement, operational benefit, model impact, or ML readiness.

## Deferred or Out-of-Scope Sources

The following registry-specific entries remain deferred and registry-only in the current phase. They are candidate enrichment entries only; their `source_name`, source reference, source documentation status, license status, file lineage, reference date, temporal availability, causal availability, leakage risk, mapping quality, and predictive value fields remain TODO-VERIFY where recorded as TODO-VERIFY in the registry.

| Registry ID | Source name | Artifact path | Phase status | Deferred reason |
|---|---|---|---|---|
| `osm_pois_overpass_store_context` | TODO-VERIFY | `raw_data/code_external_data/_external_data/osm_pois_overpass/store_static_context_osm.parquet` | `deferred_registry_only` | deferred until source license, coordinate quality, lineage, leakage, mapping quality, temporal availability, causal availability, and identical-coordinate effects are verified |
| `openligadb_event_like_matches` | TODO-VERIFY | `raw_data/code_external_data/_external_data/openligadb_matches/openligadb_matches_2025-04-01_2025-06-30.parquet` | `deferred_registry_only` | low-priority event-like feed deferred; source license, temporal availability, and leakage posture remain TODO-VERIFY |
| `bahnvorhersage_downloader_stub` | TODO-VERIFY | `raw_data/code_external_data/download_08_bahnvorhersage_parsed_delays.py` | `deferred_registry_only` | low-priority downloader context deferred; do not repair, execute, activate, or prioritize without explicit approval |
| `zensus_grid_downloader_stub` | TODO-VERIFY | `raw_data/code_external_data/download_09_zensus_grid_download_stub.py` | `deferred_registry_only` | high-priority domain deferred until source license, lineage, file existence, QA, and causal availability are verified |
| `vrr_gtfs_downloader_stub` | TODO-VERIFY | `raw_data/code_external_data/download_10_vrr_gtfs_download_stub.py` | `deferred_registry_only` | priority 5 transit feed deferred until source license, lineage, temporal coverage, QA, and causal availability are verified |

Registry presence, artifact-path presence, file presence, or downloader presence does not prove source validity, license status, usage-rights status, temporal availability, causal availability, leakage safety, mapping quality, ZIP-to-municipality truth, AGS/Gemeindeschluessel identity, coordinate quality, predictive value, forecast improvement, feature value, model impact, operational benefit, or business benefit. No reliable evidence supports promoting these deferred entries beyond registry-only candidate enrichment status.

Do not activate, repair, execute, prioritize, validate, or promote deferred downloaders without explicit later approval and phase-compliant evidence.

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
