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
- feature/external-context-features

Verified working tree:
- clean before read-only audit
- clean after read-only audit

Verified registry:
- raw_data/code_external_data/external_source_qa_registry.csv
- tracked by git
- 15 rows
- 26 columns
- predictive_value_status is TODO-VERIFY for all rows
- candidate_enrichment_status is candidate_external_enrichment_only for all rows
- 5 deferred registry rows
- no invalid phase_scope_status values found

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
- municipality_rows_with_missing_population = 0
- municipality_rows_with_missing_area = 0
- municipality_rows_with_incomplete_feature_row = 0
- store_rows_with_missing_municipality_features = 0

Interpretation limit:
- The metrics document generated QA output only.
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
- Uses RAW_SOURCE_REFERENCE_DATE = 2022-05-15.
- Uses MAX_ALLOWED_REFERENCE_DATE = 2025-06-30.
- Writes municipality and store census feature base outputs.
- Writes census_feature_base_qa_summary.csv.

Limit:
- The script contains a source-name metadata string, but source URL and license/usage terms are not fully documented in this audit.
- Source documentation remains TODO-VERIFY.
- License status remains TODO-VERIFY.
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
- Exact source URL, access date, license/usage terms, and workbook provenance remain TODO-VERIFY.
- The local workbook presence alone is not sufficient source documentation.

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
- VG250 URL is present in code, but license/usage terms remain TODO-VERIFY.
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

## Documentation File Search Result

No dedicated source documentation file was identified that fully resolves source URL, access date, license/usage terms, update logic, publication lag, revision lag, causal availability, or leakage review for the high-priority Census and reference mapping artifacts.

Files found by documentation-name search were registry, QA, source CSV, and QA summary artifacts, not complete source documentation.

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
- PLZ centroid source provenance and precision remain TODO-VERIFY.

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

### VG250 boundary cache

Observed:
- raw_data/code_external_data/_reference_geo/vg250_cache/DE_VG250.gpkg exists.
- build_zipcode_to_municipality_nrw_csv.py references VG250 and the vg250_gem municipality layer.

Interpretation limit:
- File presence and script references do not prove license, boundary version, CRS, layer integrity, source access date, or full NRW boundary consistency.
- VG250 lineage, license, and full NRW boundary consistency remain TODO-VERIFY.

## Unresolved TODO-VERIFY Items

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

## Recommended Next Step

Create or update a dedicated English source documentation artifact for high-priority Census and reference mapping sources before resolving registry TODO-VERIFY fields.

Recommended documentation target:
- raw_data/code_external_data/external_source_documentation.md

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
