# ZIP/AGS Reference QA

## Scope

This non-final QA artifact documents ZIP-to-municipality ambiguity QA and AGS/Gemeindeschluessel reference QA for the Food Prediction project.

Current phase only:
- External Data Acquisition
- Reference Mapping
- Source QA

This artifact does not perform ML integration, model training, model comparison, SHAP, feature importance, deployment, Streamlit work, business recommendations, feature-value claims, forecast-improvement claims, or operational-benefit claims.

Canonical raw data remain exactly:
- sales
- stores
- weather
- holidays

External data remain candidate enrichments only.

Predictive value status: TODO-VERIFY.

Conclusion on predictive value: No reliable evidence.

## Evidence Basis

This QA was originally based only on repository files and read-only PowerShell checks on branch `feature/zipcode-ags-reference-qa`. Post-PR29 current-file metrics for `zipcode_to_municipality_reference` were rechecked on `master` at HEAD `586460e1e85647d346901e8c76d924dab1d53e95`.

Files inspected:
- `raw_data/code_external_data/build_zipcode_to_municipality_nrw_csv.py`
- `raw_data/code_external_data/build_store_municipality_reference.py`
- `raw_data/code_external_data/_reference_geo/zipcode_to_municipality_nrw.csv`
- `raw_data/code_external_data/_reference_geo/plz_centroids_nrw.csv`
- `raw_data/code_external_data/_reference_geo/vg250_cache/DE_VG250.gpkg`
- `raw_data/code_external_data/_external_data/store_geography/store_municipality_reference.csv`
- `raw_data/code_external_data/_external_data/store_geography/store_municipality_reference_qa_summary.csv`

No downloader was run. No external output was regenerated.

## ZIP-to-Municipality Output QA

Observed current file:
- Path: `raw_data/code_external_data/_reference_geo/zipcode_to_municipality_nrw.csv`
- Rows: 864
- Unique ZIPs: 864
- Duplicate ZIP row groups: 0
- Assignment method counts:
  - `polygon_intersects`: 864
- Lexicographic resolution rows observed in current output: 0
- Nearest fallback rows observed in current output: 0

Interpretation limit:
- The current file has one output row per ZIP.
- One output row per ZIP is not proof of true one-to-one ZIP-to-municipality mapping.
- Deterministic centroid-to-polygon assignment is not validated ZIP-to-municipality source truth.
- Source truth for multi-municipality ZIPs in the repository: No reliable evidence.
- Multi-municipality ZIP ambiguity remains TODO-VERIFY.

## AGS/Gemeindeschluessel QA

Observed current files:
- `raw_data/code_external_data/_reference_geo/zipcode_to_municipality_nrw.csv`
- `raw_data/code_external_data/_external_data/store_geography/store_municipality_reference.csv`

Observed checks:
- ZIP map `municipality_ags` values failing 8-digit shape check: 0
- ZIP map `municipality_ags` values without NRW prefix `05`: 0
- Store reference `municipality_ags` values failing 8-digit shape check: 0
- Store reference `municipality_ags` values without NRW prefix `05`: 0

Script evidence:
- `build_zipcode_to_municipality_nrw_csv.py` normalizes AGS to width 8 and ARS to width 12.
- `build_zipcode_to_municipality_nrw_csv.py` filters NRW municipalities using AGS prefix `05`.
- `build_store_municipality_reference.py` reads the ZIP reference CSV with `dtype="string"`.
- ARS values must not be treated as AGS.

Interpretation limit:
- These checks verify current string shape and NRW prefix in the inspected files.
- They do not verify authoritative municipality identity against an external source authority.
- AGS/Gemeindeschluessel identity remains TODO-VERIFY.
- Leading-zero preservation against an authoritative source remains TODO-VERIFY.

## NRW Boundary QA

Observed current file:
- `raw_data/code_external_data/_reference_geo/vg250_cache/DE_VG250.gpkg` exists.

Script evidence:
- `build_zipcode_to_municipality_nrw_csv.py` references a VG250 GeoPackage URL.
- `build_zipcode_to_municipality_nrw_csv.py` uses the `vg250_gem` municipality layer.
- `build_zipcode_to_municipality_nrw_csv.py` filters NRW municipalities using AGS prefix `05`.

Official BKG/MIS metadata evidence:
- MIS docuuid `431406f6-1b31-48a9-b6db-dc4b38caf5ea`.
- Product identifier `VG250_3112`.
- Official `Letzte Änderung 31.12.2024`.
- Metadata record date `05.05.2026`.
- EPSG `25832`.
- License family `Datenlizenz Deutschland Namensnennung 2.0`.
- These facts support source metadata only and do not prove local GeoPackage/cache completeness, modification status, geometry validity, CRS transformation correctness, full NRW boundary consistency, or equivalence to the official product.

Local GeoPackage/cache metadata evidence:
- Local cache path: `raw_data/code_external_data/_reference_geo/vg250_cache/DE_VG250.gpkg`.
- Local `gpkg_contents.description` date-like values showing `2025-01-01` are local GeoPackage metadata evidence only; their exact meaning remains TODO-VERIFY.
- Local GeoPackage metadata-reference timestamps such as `2025-07-01T10:34:48Z` or `2025-07-01T10:34:49Z` are not an official VG250 reference date by themselves.

Interpretation limit:
- File presence and script references alone do not prove the exact local VG250 cache version/reference date, actual geometry validity, calculated spatial bounds, CRS transformation correctness, full layer integrity, local cache source access date, or full NRW boundary consistency.
- VG250 official source, license family, annual update cycle, and attribution requirement are documented from official BKG evidence.
- Local VG250 cache version/reference date remains TODO-VERIFY.
- Full NRW boundary consistency remains TODO-VERIFY.

## PLZ Centroid and Coordinate QA

Observed current file:
- Path: `raw_data/code_external_data/_reference_geo/plz_centroids_nrw.csv`
- Schema: `zip,lat,lng`
- Rows: 864
- Unique ZIPs: 864
- Duplicate ZIP groups: 0
- Duplicate coordinate groups: 0
- Missing coordinate rows: 0
- Nonnumeric coordinate rows: 0
- Broad NRW-bounds outlier rows: 0

Interpretation limit:
- Broad coordinate bounds are a sanity check only.
- PLZ centroid coordinates are approximate.
- PLZ centroids must not be interpreted as precise store coordinates.
- Git history adds `plz_centroids_nrw.csv` in commit `9be4a742` on 2026-03-09.
- Historical helper script evidence supports only partial local subset lineage from a local `plz_centroids.csv` and a user-provided NRW PLZ list.
- Read-only upstream source research on 2026-05-09 did not prove local lineage from BKG PLZ, OpenPLZ, OpenPLZ API data, Open.NRW / CKAN Deutschland Postleitzahlen, OpenStreetMap, or any other candidate source.
- Local `plz_centroids.csv` is proven only as historical helper input, not as a sourced or provenanced artifact.
- Upstream provenance conclusion for `plz_centroids_nrw.csv`: No reliable evidence.
- PLZ centroid source name, source URL, access date, license or usage terms, upstream provenance, reference date, precision, coordinate quality, update logic, temporal availability, causal availability, publication lag, revision lag, leakage risk, mapping quality, and predictive value remain TODO-VERIFY.

## Store Municipality Fallback QA

Observed current file:
- Path: `raw_data/code_external_data/_external_data/store_geography/store_municipality_reference.csv`

Observed QA summary:
- Store count: 84
- Stores with valid coordinates: 0
- Stores assigned by spatial join: 0
- Stores assigned by ZIP fallback: 84
- Stores unassigned: 0
- Stores with duplicate coordinates exact: 0
- Stores with spatial-ZIP mismatch: 0
- Stores with invalid ZIP code: 0

Observed row-level counts:
- `assignment_method = zipcode_fallback_no_valid_coordinates`: 84
- `qa_has_valid_coordinates = False`: 84
- `qa_assignment_used_fallback = True`: 84
- Rows unassigned or missing ZIP reference: 0

Interpretation limit:
- The current store municipality reference depends on ZIP fallback for all stores.
- ZIP fallback can create false precision if treated as verified store-level municipality truth.
- Store coordinate source quality remains TODO-VERIFY.
- Spatial assignment quality remains TODO-VERIFY because no valid store-coordinate spatial join was used.

### Store Source Schema QA Update

Read-only schema QA on branch `feature/store-coordinate-source-qa` observed the canonical stores parquet at `raw_data/20260218_144523_stores.parquet` with:

- Store rows: 84
- Columns: `subdivision_code`, `country_code`, `zipcode`, `average_weekly_revenue_Q1`, `store_id`
- Potential geo or address columns found by keyword search: `zipcode`, `store_id`

No latitude, longitude, coordinate, address, street, city, or precise store-location columns were observed.

Interpretation limit:

- The canonical stores file does not provide precise store coordinates.
- ZIP fallback remains the only current repository-supported store geography assignment basis.
- Store coordinate source quality remains TODO-VERIFY.
- OSM remains deferred until precise store coordinate quality is verified.
- Deferred OSM POI context, duplicate geospatial output risk, and identical OSM feature outputs remain TODO-VERIFY because current store geography is ZIP fallback only.
- OSM source, license, lineage, temporal availability, causal availability, leakage risk, mapping quality, coordinate source quality, and predictive value remain TODO-VERIFY.
- Source, lineage, coordinate quality, causal availability, leakage safety, predictive value, and operational value conclusion: No reliable evidence.

## Leakage and Causal Availability

The following remain TODO-VERIFY:
- publication lag
- revision lag
- update timing
- backfill behavior
- temporal coverage
- prediction-time availability
- causal availability
- leakage risk

No forecast-improvement, feature-value, operational-benefit, business-benefit, or model-impact claim is supported by this QA.

## Non-Final Status

This QA supports source and reference mapping documentation only. It does not approve ML integration or final training feature engineering.
