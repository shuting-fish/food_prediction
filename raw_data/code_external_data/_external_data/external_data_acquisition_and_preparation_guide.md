# External Data Acquisition and Preparation Guide - NON_FINAL

## 1. Purpose

This guide records how the current internal external-data candidate files were obtained or prepared.

These files are candidate enrichments only. They are not canonical raw data and are not approved for customer delivery, redistribution, joins, downstream model use, legal suitability, mapping safety, leakage safety, or predictive value.

Canonical raw concepts remain: `sales`, `stores`, `weather`, `holidays`.

## 2. Operational acquisition map

| Candidate data | Source and access | Reference period | Preparation route | Current outputs / status |
|---|---|---|---|---|
| Destatis GV100AD store-municipality candidate | Destatis Gemeindeverzeichnis / GV100AD publication page; structured file `AuszugGV4QAktuell.xlsx`; source XLSX kept in quarantine | Territorial reference date `2025-12-31`; exact HTTP download timestamp remains `TODO-VERIFY` | Download XLSX, keep original in quarantine, create NRW-only intermediate, filter by required `municipality_ags` from `store_geography/store_municipality_reference.csv`, preserve AGS leading zeros, write CSV and SHA256 sidecar | `destatis_gv100ad_store_municipalities_2025-12-31.csv`; 25 derived municipality rows; 0 missing required AGS values; 0 duplicate derived AGS values; candidate enrichment only |
| OpenStreetMap ZIP39 POI candidate | Geofabrik historical OpenStreetMap NRW PBF snapshot; source page and snapshot URL documented; OSM copyright page referenced; attribution: OpenStreetMap contributors | Historical snapshot `260531`; relation-reference completeness remains `TODO-OSM-RELATIONS` | Download PBF with `curl`, extract postal-code boundary relations with `osmium tags-filter`, build union of 39 required ZIP polygons, extract ZIP39 PBF with `osmium extract`, export features with `osmium export`, filter selected POIs with Python | ZIP39 PBF plus POI CSV; 84,186 POI rows; 364 `shop=bakery` rows; candidate enrichment only |
| Root CSV candidates | Repository-root files `Verbraucherpreisindex.csv`, `oil_price.csv`, and `event_data.csv`; hash/byte evidence only in `EXTERNAL_DATA_PROVENANCE.md` | `TODO-VERIFY` | No reproducible acquisition route documented yet because source identity, source URL/access method, access date, and usage terms remain unresolved | Keep as TODO-VERIFY candidates; do not promote or use downstream |

## 3. Destatis GV100AD store-municipality candidate

### Source and access

- Source organisation: Statistisches Bundesamt (Destatis)
- Source product family: Gemeindeverzeichnis / GV100AD municipal-directory data
- Stable publication page: https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/Administrativ/Archiv/GVAuszugQ/AuszugGV4QAktuell.html
- Structured source file: `AuszugGV4QAktuell.xlsx`
- Territorial reference date: `2025-12-31`
- Exact HTTP download timestamp: `TODO-VERIFY`
- Usage-term evidence: supporting PDF retained at `destatis_gv100ad_store_municipalities_2025-12-31/destatis_gv_isys_usage_terms_supporting_evidence.pdf`
- Usage-term status: provenance note states redistribution with source attribution is permitted; legal/customer-delivery suitability is not concluded.

### How the data was prepared

1. Download the Destatis GV100AD structured XLSX from the stable publication page.
2. Keep the original XLSX in quarantine.
3. Export or prepare the NRW-only municipality candidate.
4. Read the distinct `municipality_ags` values from `store_geography/store_municipality_reference.csv`.
5. Filter the NRW municipality candidate to those required `municipality_ags` values.
6. Preserve AGS leading zeros.
7. Do not use administrative-seat postcode as a municipality mapping key.
8. Write the store-municipality candidate CSV and SHA256 sidecar.

### Inputs and outputs

- Quarantine source XLSX: `C:\Users\simon\food_prediction_quarantine\gv100ad_cycle_2025_01_01_to_2026_03_31\AuszugGV4QAktuell.xlsx`
- NRW-only intermediate: `C:\Users\simon\food_prediction_quarantine\gv100ad_cycle_2025_01_01_to_2026_03_31\gv100ad_31122025_nrw_csv_non_final\destatis_gv100ad_municipalities_nrw_2025-12-31_NON_FINAL.csv`
- AGS filter input: `store_geography/store_municipality_reference.csv`
- Candidate output: `destatis_gv100ad_store_municipalities_2025-12-31/destatis_gv100ad_store_municipalities_2025-12-31.csv`
- Candidate output SHA256: `61F7EB1D04415343A9698C76D79D74C2C640FC0188FA3B81C252C163449382B7`
- SHA256 sidecar: `destatis_gv100ad_store_municipalities_2025-12-31/destatis_gv100ad_store_municipalities_2025-12-31.csv.sha256.txt`
- Provenance note: `destatis_gv100ad_store_municipalities_2025-12-31/provenance_destatis_gv100ad_store_municipalities_2025-12-31.txt`

### QA checks recorded

- NRW source rows: `396`
- Required unique store municipality AGS values: `25`
- Derived rows: `25`
- Missing required municipality AGS values: `0`
- Duplicate derived municipality AGS values: `0`
- Filter key: `municipality_ags`, not postcode.

### Open TODO-VERIFY

- Exact HTTP download timestamp.
- Stable public acquisition method remains documented but not fully re-executed in this guide slice.
- Legal suitability, customer-delivery readiness, redistribution approval, mapping-safety, leakage-safety, downstream join approval, source promotion, and predictive value remain not approved.

## 4. OpenStreetMap ZIP39 POI candidate

### Source and access

- Source provider: Geofabrik OpenStreetMap extract for Nordrhein-Westfalen
- Source page: https://download.geofabrik.de/europe/germany/nordrhein-westfalen.html
- Historical snapshot URL: https://download.geofabrik.de/europe/germany/nordrhein-westfalen-260531.osm.pbf
- License reference: https://www.openstreetmap.org/copyright
- Attribution: OpenStreetMap contributors
- Relation-reference completeness: `TODO-OSM-RELATIONS`

### How the data was prepared

1. Download the historical NRW PBF snapshot with `curl`.
2. Extract postal-code boundary relations with `osmium tags-filter`.
3. Build the union of the 39 required ZIP-code polygons.
4. Extract the ZIP39 PBF with `osmium extract --strategy=smart`.
5. Export OSM features with `osmium export`.
6. Retain selected POIs, including bakeries tagged as `shop=bakery`.
7. Verify output hashes and aggregate row counts.

Detailed reproduction commands are documented in:
`osm_geofabrik_nrw_zip39_poi_source_snapshot_260531/reproduce_osm_geofabrik_nrw_zip39_poi_source_snapshot_260531.md`

### Outputs and QA checks recorded

- PBF output: `osm_geofabrik_nrw_zip39_extract_source_snapshot_260531.osm.pbf`
- PBF SHA256: `6425EB1C5832F5569A39F1FF1A2A05B6C66F49C93890644EFFAF15282E5EFD67`
- POI CSV output: `osm_geofabrik_nrw_zip39_poi_including_bakeries_source_snapshot_260531.csv`
- POI CSV SHA256: `6888325638722CC2896F50FE607CA9A74F258241400EB9ECBB49BF98E7491AAC`
- POI rows: `84186`
- `shop=bakery` rows: `364`

### Open TODO-VERIFY

- `TODO-OSM-RELATIONS` remains open.
- License, redistribution, delivery-readiness, mapping-safety, leakage-safety, downstream join approval, source promotion, and predictive value remain not approved.

## 5. Root CSV candidates

The repository root contains three additional external candidate CSV files:

- `Verbraucherpreisindex.csv`
- `oil_price.csv`
- `event_data.csv`

Their hashes and bytes are recorded in `EXTERNAL_DATA_PROVENANCE.md`.

No reproducible acquisition guide is provided for these root CSV candidates because source identity, source URL or access method, access date, and usage terms remain `TODO-VERIFY`.

## 6. Operational boundary

This guide explains acquisition and preparation only.

It does not approve customer delivery, redistribution, legal suitability, source promotion, canonical overwrite, joins, downstream model use, mapping safety, leakage safety, predictive value, operational value, or business value.
