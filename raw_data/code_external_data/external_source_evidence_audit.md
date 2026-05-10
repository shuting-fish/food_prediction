# External Source Evidence Audit

## Scope

This audit documents repository evidence for the external source QA registry.

Current phase only:
- External Data Acquisition
- Reference Mapping
- Source QA

This audit does not perform ML integration, model training, model comparison, SHAP, feature importance, deployment, Streamlit work, business recommendations, feature-value claims, forecast-improvement claims, or operational-benefit claims.

Canonical raw data remain exactly:
- sales
- stores
- weather
- holidays

External data remain candidate enrichments only.

Predictive value status: TODO-VERIFY.
Conclusion on predictive value: No reliable evidence.

## Verified Repository State During Audit

Verified branch:
- feature/source-license-hardening

Verified working tree:
- clean before source-license hardening edits
- after implementation modified files are limited to external_source_qa_registry.csv, external_source_documentation.md, and external_source_evidence_audit.md

Verified registry:
- raw_data/code_external_data/external_source_qa_registry.csv
- tracked by git
- 15 rows
- 26 columns
- predictive_value_status is TODO-VERIFY for all rows
- candidate_enrichment_status is candidate_external_enrichment_only for all rows
- 5 deferred registry rows
- no invalid phase_scope_status values found

## Current High TODO-VERIFY Inventory

The following current_high registry rows still contain TODO-VERIFY fields after applying repository-supported evidence and official public source evidence checked on 2026-05-08.

| Registry row | TODO-VERIFY fields |
|---|---|
| census_destatis_workbook | temporal_availability_status; causal_availability_status; leakage_risk_status; mapping_quality_status; predictive_value_status; notes |
| census_population_area_source_csv | temporal_availability_status; causal_availability_status; leakage_risk_status; mapping_quality_status; predictive_value_status; notes |
| census_raw_municipality_csv | temporal_availability_status; causal_availability_status; leakage_risk_status; mapping_quality_status; predictive_value_status; notes |
| census_municipality_feature_base | temporal_availability_status; causal_availability_status; leakage_risk_status; mapping_quality_status; predictive_value_status; notes |
| census_store_feature_base | temporal_availability_status; causal_availability_status; leakage_risk_status; mapping_quality_status; predictive_value_status; notes |
| zipcode_to_municipality_reference | source_reference_or_url; source_documentation_status; license_status; reference_date; temporal_availability_status; causal_availability_status; leakage_risk_status; mapping_quality_status; predictive_value_status; notes |
| nrw_plz_centroids | source_name; source_reference_or_url; source_documentation_status; license_status; file_lineage_status; reference_date; temporal_availability_status; causal_availability_status; leakage_risk_status; mapping_quality_status; predictive_value_status; notes |
| vg250_boundary_cache | reference_date; temporal_availability_status; causal_availability_status; leakage_risk_status; mapping_quality_status; predictive_value_status; notes |
| store_municipality_reference | source_reference_or_url; source_documentation_status; license_status; reference_date; temporal_availability_status; causal_availability_status; leakage_risk_status; mapping_quality_status; predictive_value_status |
| store_municipality_reference_parquet | source_reference_or_url; source_documentation_status; license_status; reference_date; temporal_availability_status; causal_availability_status; leakage_risk_status; mapping_quality_status; predictive_value_status; notes |

Fields are left as TODO-VERIFY where repository evidence and checked official public source pages do not resolve source URL, access date, license, temporal availability, causal availability, publication lag, revision lag, leakage risk, mapping validity, predictive value, feature value, forecast improvement, or business value.

## Official Public Source Evidence

Official pages checked on 2026-05-08:
- Destatis GV-ISys Gemeindeverzeichnis: https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/_inhalt.html
- Destatis 31.12.2024 GV-ISys publication page: https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/Administrativ/Archiv/GVAuszugQ/AuszugGV4QAktuell.html
- Destatis standard copyright terms: https://www.destatis.de/DE/Service/Impressum/copyright-allgemein.html
- BKG VG250 product page: https://gdz.bkg.bund.de/index.php/default/open-data/verwaltungsgebiete-1-250-000-stand-01-01-vg250-01-01.html
- BKG PLZ product page, context only: https://gdz.bkg.bund.de/index.php/default/postleitzahlgebiete-deutschland-plz.html

Destatis evidence:
- GV-ISys is published by the Statistical Offices of the Federation and the Länder and includes AGS, ARS, municipality names, postal code of the administrative seat, area, and population.
- The 31.12.2024 publication page for all politically independent municipalities is dated 2025-01-07.
- Destatis standard copyright terms permit reuse with source attribution unless product-specific third-party or deviating rights apply.
- Product-specific third-party or deviating rights remain TODO-VERIFY for the local workbook.

BKG evidence:
- VG250 01.01 provides administrative areas from state to municipality boundaries for Germany.
- The official product page reports reference status 2025-01-01 and a 1-year update cycle.
- VG250 is provided under Datenlizenz Deutschland Namensnennung 2.0 with BKG source attribution requirements.
- The local `DE_VG250.gpkg` cache version remains TODO-VERIFY because file presence and script URL do not prove exact local cache version.
- The BKG PLZ product is restricted and requires a license agreement; it is context only and does not prove lineage for the local `plz_centroids_nrw.csv`.

PLZ centroid repository evidence:
- Git history adds `raw_data/code_external_data/_reference_geo/plz_centroids_nrw.csv` in commit `9be4a742` on 2026-03-09.
- The historical helper script `make_plz_centroids_nrw_subset_from_list.py` supports only partial local subset lineage from a local `plz_centroids.csv` and a user-provided NRW PLZ list.
- No tracked Git history was found for local `plz_centroids.csv`.
- Same-commit scripts and notebook references use PLZ centroids as an input but do not prove source URL, license, access date, upstream provenance, precision, coordinate method, or reference date.
- Repository evidence does not prove the upstream centroid source, source URL, access date, license, reference date, precision, coordinate quality, update logic, temporal availability, causal availability, leakage risk, or mapping quality.

PLZ centroid upstream source research:
- Read-only upstream source research on 2026-05-09 reviewed official or source-linked candidate pages only.
- BKG Postleitzahlgebiete Deutschland remains restricted context only, not local lineage.
- OpenPLZ and the OpenPLZ API data GitHub repository remain plausible candidates only, not proven local lineage.
- Open.NRW / CKAN Deutschland Postleitzahlen remains a plausible candidate only, not proven local lineage.
- OpenStreetMap copyright/license evidence provides ODbL context only, not local lineage.
- No candidate source is tied by repository evidence to local `plz_centroids.csv` or `plz_centroids_nrw.csv`.
- The local `plz_centroids.csv` is proven only as historical helper input, not as a sourced or provenanced artifact.
- Upstream provenance conclusion for `plz_centroids_nrw.csv`: No reliable evidence.

## Evidence Path Audit

All qa_evidence_path entries in the registry existed during the audit.

Evidence paths verified as present:
- raw_data/code_external_data/census_raw/destatis_gvisys_31122024.xlsx
- raw_data/code_external_data/census_raw/municipality_population_area_source.csv
- raw_data/code_external_data/census_raw/municipality_census_raw.csv
- raw_data/code_external_data/_external_data/census_features/census_feature_base_qa_summary.csv
- raw_data/code_external_data/_reference_geo/zipcode_to_municipality_nrw.csv
- raw_data/code_external_data/_reference_geo/plz_centroids_nrw.csv
- raw_data/code_external_data/_reference_geo/vg250_cache/DE_VG250.gpkg
- raw_data/code_external_data/_external_data/store_geography/store_municipality_reference_qa_summary.csv
- raw_data/code_external_data/_external_data/osm_pois_overpass/store_static_context_osm.parquet
- raw_data/code_external_data/_external_data/openligadb_matches/openligadb_matches_2025-04-01_2025-06-30.parquet
- raw_data/code_external_data/download_08_bahnvorhersage_parsed_delays.py
- raw_data/code_external_data/download_09_zensus_grid_download_stub.py
- raw_data/code_external_data/download_10_vrr_gtfs_download_stub.py

File presence does not prove source validity, license status, causal availability, leakage safety, mapping validity, or predictive value.

## QA Summary Evidence

### Census feature QA summary

File:
- raw_data/code_external_data/_external_data/census_features/census_feature_base_qa_summary.csv

Observed metrics:
- store_count = 84
- municipality_count_in_store_universe = 25
- municipality_feature_rows = 25
- municipality_rows_missing_coverage = 0
- municipality_rows_with_name_mismatch = 19
- municipality rows with `qa_name_mismatch_store_vs_raw = False` = 6
- municipality_rows_with_missing_population = 0
- municipality_rows_with_missing_area = 0
- municipality_rows_with_incomplete_feature_row = 0
- store_rows_with_missing_municipality_features = 0

Interpretation limit:
- The metrics document generated QA output only.
- Read-only row-level review found the observed mismatch pattern is official/display municipality names with designation suffixes versus shorter store-reference municipality names.
- Name mismatch review does not prove authoritative municipality identity or resolve AGS/Gemeindeschluessel authority, format, or leading-zero preservation.
- They do not prove source license, source URL, publication lag, revision lag, causal availability, leakage safety, or predictive value.

### Store municipality reference QA summary

File:
- raw_data/code_external_data/_external_data/store_geography/store_municipality_reference_qa_summary.csv

Observed metrics:
- store_count = 84
- stores_with_valid_coordinates = 0
- stores_assigned_by_spatial_join = 0
- stores_assigned_by_zip_fallback = 84
- stores_unassigned = 0
- stores_with_duplicate_coordinates_exact = 0
- stores_with_spatial_zip_mismatch = 0
- stores_with_invalid_zipcode = 0

Interpretation limit:
- The metrics show that the current output used ZIP fallback for all stores because no valid store coordinates were available.
- Coordinate source quality remains TODO-VERIFY.
- ZIP-to-municipality ambiguity remains TODO-VERIFY.
- Spatial assignment quality remains TODO-VERIFY because no valid store-coordinate spatial join was used.
- OSM work remains deferred until store coordinate quality is verified.

## Header Evidence

Observed CSV headers support that expected key columns exist in the following files:

- raw_data/code_external_data/_reference_geo/zipcode_to_municipality_nrw.csv
- raw_data/code_external_data/_reference_geo/plz_centroids_nrw.csv
- raw_data/code_external_data/census_raw/municipality_census_raw.csv
- raw_data/code_external_data/census_raw/municipality_population_area_source.csv
- raw_data/code_external_data/_external_data/census_features/municipality_census_feature_base.csv
- raw_data/code_external_data/_external_data/census_features/store_census_feature_base.csv
- raw_data/code_external_data/_external_data/store_geography/store_municipality_reference.csv

Header presence does not prove semantic correctness, license status, causal availability, or mapping validity.

## Script Evidence

### build_municipality_census_feature_base.py

Relevant evidence:
- Loads store municipality reference from parquet or CSV.
- Loads municipality census raw CSV.
- Normalizes municipality_ags.
- Checks missing or invalid municipality_ags.
- Checks duplicate municipality_ags.
- Builds `municipality_name_store_reference` from the store municipality universe.
- Computes `qa_name_mismatch_store_vs_raw` by comparing stripped and casefolded raw and store-reference municipality names.
- Uses RAW_SOURCE_NAME = Official municipality census / demography extract.
- Uses RAW_SOURCE_REFERENCE_DATE = 2022-05-15.
- Uses MAX_ALLOWED_REFERENCE_DATE = 2025-06-30.
- Writes municipality and store census feature base outputs.
- Writes census_feature_base_qa_summary.csv.

Limit:
- Official Destatis pages resolve the source family and standard reuse terms for the GV-ISys source, but the derived script reference date conflicts with the official 2024-12-31 product reference.
- Name mismatch QA is a string-comparison check only and must not be used to resolve municipality identity by assumption.
- Product-specific third-party or deviating rights remain TODO-VERIFY.
- Publication lag and revision lag remain TODO-VERIFY.
- Predictive value remains TODO-VERIFY.

### fill_municipality_census_population_area.py

Relevant evidence:
- Reads local Destatis workbook path census_raw/destatis_gvisys_31122024.xlsx.
- Builds or updates municipality_population_area_source.csv.
- Updates municipality_census_raw.csv.
- Normalizes municipality_ags to 8 characters.
- Checks duplicate municipality_ags.
- Checks unresolved target municipalities.

Limit:
- Official Destatis pages resolve source family, source page, access date, and standard reuse terms.
- The exact local workbook acquisition event remains TODO-VERIFY.
- Product-specific third-party or deviating rights remain TODO-VERIFY.

### build_zipcode_to_municipality_nrw_csv.py

Relevant evidence:
- Documents that ZIP centroids are spatially assigned to official NRW municipalities.
- Uses BKG VG250 boundary dataset URL in code.
- Uses VG250 municipality layer.
- Filters NRW municipalities by AGS prefix 05.
- Normalizes AGS to width 8 and ARS to width 12.
- Uses point-in-polygon assignment.
- Uses nearest municipality fallback for unmatched ZIP centroids.
- Detects duplicate ZIPs in output.
- Checks for non-NRW municipality keys in output.

Mapping risk:
- The script contains deterministic tie handling for cases where one centroid intersects multiple municipality polygons by choosing the lexicographically smallest municipality_ags.
- This must remain a documented mapping risk unless there is explicit allocation logic and QA evidence.
- ZIP/postcode must not be treated as a one-to-one municipality key for all business or geospatial purposes.

Limit:
- VG250 source page, license family, update cycle, and source attribution requirement are resolved from official BKG evidence.
- The exact local cache version remains TODO-VERIFY.
- NRW boundary consistency remains TODO-VERIFY until independently QA-checked.
- ZIP centroid source quality remains TODO-VERIFY.
- Multi-municipality ZIP ambiguity remains TODO-VERIFY.

### build_store_municipality_reference.py

Relevant evidence:
- Reads canonical stores parquet.
- Reads zipcode_to_municipality_nrw.csv.
- Optionally reads store_coordinates.csv if present.
- Validates stores are DE and DE-NW.
- Checks coordinate columns, missing coordinates, coordinate helper usage, source conflicts, Germany range, NRW bbox, duplicate coordinates, invalid ZIPs, ZIP reference matches, spatial matches, ambiguous spatial matches, spatial-vs-ZIP mismatches, fallback usage, and unassigned stores.
- Current QA summary reports 0 valid coordinates and 84 ZIP fallback assignments.

Limit:
- Store coordinate quality remains TODO-VERIFY.
- ZIP fallback assignment remains candidate reference mapping only.
- Spatial assignment validity is limited because no valid coordinates were available.
- Geospatial join assumptions remain TODO-VERIFY.
- OSM remains deferred until coordinate quality is verified.

Additional read-only schema QA on branch `feature/store-coordinate-source-qa` observed:

- Canonical stores parquet path exists: `raw_data/20260218_144523_stores.parquet`.
- Shape: 84 rows and 5 columns.
- Columns: `subdivision_code`, `country_code`, `zipcode`, `average_weekly_revenue_Q1`, `store_id`.
- No latitude, longitude, coordinate, address, street, city, or precise store-location columns were observed.
- Only `zipcode` and `store_id` matched the geo/address search terms used in the schema check.

Limit:

- The canonical stores file provides ZIP-level location context only.
- There is no repository evidence for precise store coordinates in the canonical stores file.
- Store coordinate source quality remains TODO-VERIFY.
- OSM remains deferred until a precise and documented store coordinate source is verified.

### Deferred OSM / identical geospatial output blocker after PR25

Observed:
- PR25 store coordinate source QA found no latitude, longitude, coordinate, address, street, city, or precise store-location columns in canonical stores.
- Current store geography QA reports 84 stores, 0 valid coordinates, 84 ZIP fallback assignments, and 0 spatial joins.
- The registry row `osm_pois_overpass_store_context` remains deferred.

Interpretation limit:
- OSM work must remain deferred until precise documented store coordinates are verified.
- The existing OSM artifact must not be interpreted as precise store context.
- Duplicate geospatial output risk and identical OSM feature outputs remain TODO-VERIFY because current store geography is ZIP fallback only.
- OSM source, license, raw source traceability, file lineage, temporal availability, causal availability, leakage risk, mapping quality, coordinate source quality, and predictive value remain TODO-VERIFY.
- Source, lineage, coordinate quality, causal availability, leakage safety, predictive value, and operational value conclusion: No reliable evidence.

## Documentation Hardening Status

The repository source documentation artifact was updated for repository-supported evidence and checked official public source evidence only:
- raw_data/code_external_data/external_source_documentation.md

This does not fully resolve source URL, access date, license/usage terms, publication lag, revision lag, causal availability, leakage review, mapping validity, or predictive value for the high-priority Census and reference mapping artifacts.

## ZIP/AGS Reference QA Evidence

File:
- raw_data/code_external_data/zipcode_ags_reference_qa.md

Read-only QA was performed on branch:
- feature/zipcode-ags-reference-qa

### ZIP-to-municipality current output

File:
- raw_data/code_external_data/_reference_geo/zipcode_to_municipality_nrw.csv

Observed metrics:
- rows = 864
- unique ZIPs = 864
- duplicate ZIP row groups = 0
- assignment_method polygon_intersects = 864
- lexicographic resolution rows observed in current output = 0
- nearest municipality fallback rows observed in current output = 0

Interpretation limit:
- The current output has one row per ZIP, but that is not proof of true one-to-one ZIP-to-municipality mapping.
- Source truth for multi-municipality ZIPs in repository evidence: No reliable evidence.
- Multi-municipality ZIP ambiguity remains TODO-VERIFY.

### AGS/Gemeindeschluessel current output

Observed metrics:
- ZIP map municipality_ags values failing 8-digit shape check = 0
- ZIP map municipality_ags values without NRW prefix 05 = 0
- Store reference municipality_ags values failing 8-digit shape check = 0
- Store reference municipality_ags values without NRW prefix 05 = 0

Script evidence:
- build_zipcode_to_municipality_nrw_csv.py normalizes AGS to width 8 and ARS to width 12.
- build_zipcode_to_municipality_nrw_csv.py filters NRW municipalities using AGS prefix 05.
- build_store_municipality_reference.py reads the ZIP reference CSV with dtype="string".

Interpretation limit:
- Shape and prefix checks do not prove authoritative municipality identity.
- AGS/Gemeindeschluessel identity remains TODO-VERIFY.

### PLZ centroid current output

File:
- raw_data/code_external_data/_reference_geo/plz_centroids_nrw.csv

Observed metrics:
- schema = zip, lat, lng
- rows = 864
- unique ZIPs = 864
- duplicate ZIP groups = 0
- duplicate coordinate groups = 0
- missing coordinate rows = 0
- nonnumeric coordinate rows = 0
- broad NRW-bounds outlier rows = 0

Interpretation limit:
- PLZ centroid coordinates are approximate.
- Broad coordinate bounds are a sanity check only.
- Read-only upstream source research did not prove local lineage from BKG PLZ, OpenPLZ, OpenPLZ API data, Open.NRW / CKAN Deutschland Postleitzahlen, OpenStreetMap, or any other candidate source.
- Upstream provenance conclusion for `plz_centroids_nrw.csv`: No reliable evidence.
- PLZ centroid source provenance, source URL, access date, license, reference date, precision, coordinate quality, update logic, temporal availability, causal availability, leakage risk, mapping quality, and predictive value remain TODO-VERIFY.

### Store municipality fallback current output

File:
- raw_data/code_external_data/_external_data/store_geography/store_municipality_reference.csv

Observed metrics:
- assignment_method zipcode_fallback_no_valid_coordinates = 84
- qa_has_valid_coordinates False = 84
- qa_assignment_used_fallback True = 84
- rows unassigned or missing ZIP reference = 0

Interpretation limit:
- The current store municipality reference depends on ZIP fallback for all stores.
- ZIP fallback can create false precision if treated as verified store-level municipality truth.
- Store coordinate source quality remains TODO-VERIFY.
- Spatial assignment quality remains TODO-VERIFY.
- OSM remains deferred until coordinate quality is verified.

### VG250 boundary cache

Observed:
- raw_data/code_external_data/_reference_geo/vg250_cache/DE_VG250.gpkg exists.
- build_zipcode_to_municipality_nrw_csv.py references VG250 and the vg250_gem municipality layer.

Interpretation limit:
- File presence and script references do not prove license, boundary version, CRS, layer integrity, source access date, or full NRW boundary consistency.
- VG250 official source page, license family, update cycle, and attribution requirement are resolved from official BKG evidence.
- Local cache version, CRS/layer integrity, and full NRW boundary consistency remain TODO-VERIFY.

## Unresolved TODO-VERIFY Items

The following items remain unresolved:

- exact local acquisition event for the Census workbook
- product-specific third-party or deviating Destatis rights
- PLZ centroid source URL or stable source reference
- PLZ centroid license or usage terms
- PLZ centroid access date, reference date, precision, coordinate quality, update logic, temporal availability, causal availability, leakage risk, and mapping quality
- local VG250 cache reference date/version
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
- OSM source, license, lineage, temporal availability, causal availability, leakage risk, mapping quality, and predictive value
- predictive value
- feature value
- model impact
- forecast improvement
- operational benefit
- business benefit

## Remaining Required Work Before Further Downloads

Before further downloads, collect source evidence that directly resolves source URL or stable source reference, access date, license or usage terms, raw source provenance, publication lag, revision lag, temporal availability, causal availability, leakage risk, mapping validity, and store coordinate quality.

Do not change any TODO-VERIFY status to resolved unless supported by repository evidence, source evidence, reproducible checks, or explicit user confirmation.

## Stop Conditions

Stop before implementation if any proposed work would require:
- ML integration
- model training
- model comparison
- SHAP
- feature importance
- Streamlit
- deployment
- business recommendations
- forecast-improvement claims
- feature-value claims
- low-priority feed activation
- downloader repair
- undocumented source assumptions
- undocumented geospatial join assumptions
- resolving TODO-VERIFY by assumption
