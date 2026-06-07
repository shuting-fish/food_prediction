# External Data Acquisition and Preparation Guide ÔÇö NON_FINAL

## 1. Scope and restrictions

This guide documents reproducible acquisition and preparation steps for isolated external-data candidate enrichments.

These artifacts are not canonical raw data. They must remain separate from `sales`, `stores`, `weather`, and `holidays`.

This guide does not approve source promotion, redistribution, downstream model use, customer delivery, legal suitability, mapping safety, leakage safety, or predictive value.

## 2. Destatis GV-ISys municipal-directory candidate

### Source

- Organisation: Statistisches Bundesamt (Destatis)
- Product family: Gemeindeverzeichnis / GV100AD
- Stable publication page: https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/Administrativ/Archiv/GVAuszugQ/AuszugGV4QAktuell.html
- Structured source file: `AuszugGV4QAktuell.xlsx`
- Territorial reference date: `2025-12-31`
- Exact original HTTP download timestamp: `TODO-VERIFY`

### Acquisition and preparation

1. Open the stable Destatis publication page.
2. Download the linked XLSX file for the fourth quarter of 2025.
3. Record the download timestamp and SHA256 hash.
4. Restrict the structured source data to NRW municipalities.
5. Read the distinct `municipality_ags` values from `store_municipality_reference.csv`.
6. Filter the NRW municipality candidate by `municipality_ags`.
7. Preserve AGS leading zeros.
8. Do not filter by administrative-seat postcode.

### Candidate output

- File: `destatis_gv100ad_store_municipalities_2025-12-31/destatis_gv100ad_store_municipalities_2025-12-31.csv`
- Expected candidate rows: `25`
- Join candidate: `municipality_ags`
- Schema:
  `reference_date, municipality_ags, municipality_ars, state_ars, regional_district_ars, district_ars, association_ars, municipality_component, municipality_name, area_km2, population_total, population_male, population_female, population_per_km2, administrative_seat_zipcode, centroid_longitude, centroid_latitude, travel_region_code, travel_region_name, urbanisation_code, urbanisation_name`

### Preserved blockers

- Exact field-level temporal semantics before downstream use: `TODO-VERIFY`
- Exact original HTTP download timestamp: `TODO-VERIFY`
- Delivery readiness: `Blocked / TODO-VERIFY`

## 3. OpenStreetMap ZIP39 POI candidate

### Source

- Source page: https://download.geofabrik.de/europe/germany/nordrhein-westfalen.html
- Historical snapshot URL: https://download.geofabrik.de/europe/germany/nordrhein-westfalen-260531.osm.pbf
- Attribution: `┬® OpenStreetMap contributors`
- License reference: https://www.openstreetmap.org/copyright

### Acquisition and preparation

1. Download the historical NRW PBF snapshot with `curl`.
2. Extract postal-code boundary relations with `osmium tags-filter`.
3. Build the union of the 39 required ZIP-code polygons.
4. Extract the ZIP39 PBF with `osmium extract --strategy=smart`.
5. Export OSM features with `osmium export`.
6. Retain selected POIs, including bakeries tagged as `shop=bakery`.
7. Verify output hashes and aggregate row counts.

Detailed commands are documented in:
`osm_geofabrik_nrw_zip39_poi_source_snapshot_260531/reproduce_osm_geofabrik_nrw_zip39_poi_source_snapshot_260531.md`

### Expected internal-candidate outputs

- PBF file: `osm_geofabrik_nrw_zip39_extract_source_snapshot_260531.osm.pbf`
- PBF SHA256: `6425EB1C5832F5569A39F1FF1A2A05B6C66F49C93890644EFFAF15282E5EFD67`
- CSV file: `osm_geofabrik_nrw_zip39_poi_including_bakeries_source_snapshot_260531.csv`
- CSV SHA256: `6888325638722CC2896F50FE607CA9A74F258241400EB9ECBB49BF98E7491AAC`
- Expected POI rows: `84186`
- Expected `shop=bakery` rows: `364`
- Schema:
  `feature_unique_id, osm_type, osm_id, geometry_type, representative_latitude, representative_longitude, coordinate_semantics, name, amenity, shop, tourism, leisure, office, craft, healthcare, public_transport, railway, highway, brand, operator, tags_json`

### Preserved blocker

`TODO-OSM-RELATIONS` remains open. It is non-blocking only for the previously reduced internal ZIP39 OSM POI export scope.

## 4. Separation from delivery artifacts

All documented outputs remain isolated internal candidate enrichments.

Before any delivery-readiness review, verify source terms, redistribution suitability, raw lineage, schema, sidecar metadata, temporal semantics, mapping safety, leakage safety, and QA evidence for the intended delivery artifact.
