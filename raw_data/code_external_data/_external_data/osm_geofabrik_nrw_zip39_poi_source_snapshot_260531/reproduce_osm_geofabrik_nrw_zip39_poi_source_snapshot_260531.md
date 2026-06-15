# OSM ZIP39 PBF and POI CSV - Step-by-Step Reproduction

## Output files

- `osm_geofabrik_nrw_zip39_extract_source_snapshot_260531.osm.pbf`
- `osm_geofabrik_nrw_zip39_poi_including_bakeries_source_snapshot_260531.csv`

The POI CSV includes the selected OpenStreetMap points of interest, including bakeries tagged as `shop=bakery`.

## Prerequisites

Use Linux or Windows Subsystem for Linux (WSL) with:

- `osmium-tool`
- `python3`
- `curl`
- `gzip`
- `sha256sum`

## 1. Create a working directory

```bash
mkdir -p osm_zip39_reproduction/work
cd osm_zip39_reproduction
```

## 2. Download the historical NRW OpenStreetMap snapshot

```bash
curl --location \
  --output nordrhein-westfalen-260531.osm.pbf \
  https://download.geofabrik.de/europe/germany/nordrhein-westfalen-260531.osm.pbf
```

Source page:

```text
https://download.geofabrik.de/europe/germany/nordrhein-westfalen.html
```

## 3. Create `required_zipcodes.txt`

```text
41812
41836
41849
52062
52064
52066
52068
52070
52072
52074
52076
52078
52080
52134
52146
52152
52156
52159
52222
52223
52224
52249
52349
52351
52353
52355
52379
52382
52385
52393
52428
52441
52445
52477
52499
52511
52525
52531
52538
```

## 4. Create `select_zip39_union.py`

```python
import json
import sys
from pathlib import Path

zip_list_path = Path(sys.argv[1])
source_geojson_path = Path(sys.argv[2])
output_geojson_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])

required_zipcodes = {
    line.strip()
    for line in zip_list_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
}

source = json.loads(source_geojson_path.read_text(encoding="utf-8"))
features = source.get("features")

if not isinstance(features, list):
    raise SystemExit("STOP: Postal boundary GeoJSON has no FeatureCollection features.")

matched_counts = {}
polygon_parts = []

for feature in features:
    properties = feature.get("properties") or {}
    zipcode = str(properties.get("postal_code", "")).strip()

    if zipcode not in required_zipcodes:
        continue

    geometry = feature.get("geometry") or {}
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")

    if geometry_type == "Polygon":
        polygon_parts.append(coordinates)
    elif geometry_type == "MultiPolygon":
        polygon_parts.extend(coordinates)
    else:
        raise SystemExit(f"STOP: Unsupported geometry type for ZIP {zipcode}: {geometry_type}")

    matched_counts[zipcode] = matched_counts.get(zipcode, 0) + 1

matched_zipcodes = set(matched_counts)
missing_zipcodes = sorted(required_zipcodes - matched_zipcodes)

if missing_zipcodes:
    raise SystemExit("STOP: Missing ZIP boundary relations: " + ", ".join(missing_zipcodes))

if len(matched_zipcodes) != 39:
    raise SystemExit("STOP: Expected 39 matched ZIP boundaries, observed: " + str(len(matched_zipcodes)))

output = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "properties": {"role": "zip39_union_candidate", "zip_count": len(matched_zipcodes)},
        "geometry": {"type": "MultiPolygon", "coordinates": polygon_parts},
    }],
}

summary = {
    "required_zip_count": len(required_zipcodes),
    "matched_zip_count": len(matched_zipcodes),
    "selected_feature_count": sum(matched_counts.values()),
    "polygon_part_count": len(polygon_parts),
    "multiple_features_per_zip": {
        zipcode: count
        for zipcode, count in sorted(matched_counts.items())
        if count > 1
    },
    "matched_zipcodes": sorted(matched_zipcodes),
}

output_geojson_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
```

## 5. Build the ZIP39 PBF

```bash
osmium tags-filter \
  --fsync \
  --output work/nrw_postal_code_boundaries.osm.pbf \
  nordrhein-westfalen-260531.osm.pbf \
  r/boundary=postal_code

osmium export \
  --geometry-types=polygon \
  --show-errors \
  --output work/nrw_postal_code_boundaries.geojson \
  work/nrw_postal_code_boundaries.osm.pbf

python3 \
  select_zip39_union.py \
  required_zipcodes.txt \
  work/nrw_postal_code_boundaries.geojson \
  work/required_zip39_union.geojson \
  work/zip39_selection_summary.json

osmium extract \
  --polygon work/required_zip39_union.geojson \
  --strategy=smart \
  --set-bounds \
  --fsync \
  --verbose \
  --output osm_geofabrik_nrw_zip39_extract_source_snapshot_260531.osm.pbf \
  nordrhein-westfalen-260531.osm.pbf
```

## 6. Create `export_zip39_pois.py`

```python
import csv
import gzip
import json
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

target_keys = {
    "amenity", "shop", "tourism", "leisure", "office", "craft",
    "healthcare", "public_transport",
}

target_values = {
    ("railway", "station"),
    ("railway", "halt"),
    ("railway", "tram_stop"),
    ("railway", "subway_entrance"),
    ("highway", "bus_stop"),
}

headers = [
    "feature_unique_id", "osm_type", "osm_id", "geometry_type",
    "representative_latitude", "representative_longitude",
    "coordinate_semantics", "name", "amenity", "shop", "tourism",
    "leisure", "office", "craft", "healthcare", "public_transport",
    "railway", "highway", "brand", "operator", "tags_json",
]

def is_target(tags):
    return any(key in tags for key in target_keys) or any(
        tags.get(key) == value for key, value in target_values
    )

def collect_pairs(value, output):
    if not isinstance(value, list):
        return
    if (
        len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        output.append((float(value[0]), float(value[1])))
        return
    for item in value:
        collect_pairs(item, output)

def representative_coordinate(geometry):
    if not geometry:
        return "", "", "NO_GEOMETRY"
    geometry_type = geometry.get("type", "")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Point" and isinstance(coordinates, list) and len(coordinates) >= 2:
        return coordinates[1], coordinates[0], "POINT_COORDINATE_FROM_OSM_GEOMETRY"
    pairs = []
    collect_pairs(coordinates, pairs)
    if not pairs:
        return "", "", "NO_REPRESENTATIVE_COORDINATE"
    min_lon = min(pair[0] for pair in pairs)
    max_lon = max(pair[0] for pair in pairs)
    min_lat = min(pair[1] for pair in pairs)
    max_lat = max(pair[1] for pair in pairs)
    return (
        (min_lat + max_lat) / 2,
        (min_lon + max_lon) / 2,
        "GEOMETRY_BOUNDING_BOX_CENTER_APPROXIMATE",
    )

feature_unique_ids = set()
row_count = 0

with input_path.open("r", encoding="utf-8") as source:
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=headers)
        writer.writeheader()

        for raw_line in source:
            raw_line = raw_line.lstrip("\x1e").strip()
            if not raw_line:
                continue

            feature = json.loads(raw_line)
            properties = feature.get("properties") or {}
            feature_unique_id = str(feature.get("id") or "").strip()
            osm_type = str(properties.get("@type") or "").strip()
            osm_id = str(properties.get("@id") or "").strip()

            if not feature_unique_id or not osm_type or not osm_id:
                raise RuntimeError("STOP: Missing OSM identity field.")

            if feature_unique_id in feature_unique_ids:
                raise RuntimeError("STOP: Duplicate feature_unique_id.")

            feature_unique_ids.add(feature_unique_id)

            tags = {
                str(key): str(value)
                for key, value in properties.items()
                if not str(key).startswith("@") and value is not None
            }

            if not is_target(tags):
                continue

            geometry = feature.get("geometry") or {}
            latitude, longitude, coordinate_semantics = representative_coordinate(geometry)

            writer.writerow({
                "feature_unique_id": feature_unique_id,
                "osm_type": osm_type,
                "osm_id": osm_id,
                "geometry_type": str(geometry.get("type") or ""),
                "representative_latitude": latitude,
                "representative_longitude": longitude,
                "coordinate_semantics": coordinate_semantics,
                "name": tags.get("name", ""),
                "amenity": tags.get("amenity", ""),
                "shop": tags.get("shop", ""),
                "tourism": tags.get("tourism", ""),
                "leisure": tags.get("leisure", ""),
                "office": tags.get("office", ""),
                "craft": tags.get("craft", ""),
                "healthcare": tags.get("healthcare", ""),
                "public_transport": tags.get("public_transport", ""),
                "railway": tags.get("railway", ""),
                "highway": tags.get("highway", ""),
                "brand": tags.get("brand", ""),
                "operator": tags.get("operator", ""),
                "tags_json": json.dumps(tags, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            })

            row_count += 1

if row_count == 0:
    raise RuntimeError("STOP: POI export produced zero rows.")

print(f"POI_ROW_COUNT={row_count}")
```

## 7. Build the POI CSV

```bash
osmium export \
  --add-unique-id=type_id \
  --attributes=type,id \
  --output-format=geojsonseq \
  --output work/osm_zip39_features.geojsonseq \
  osm_geofabrik_nrw_zip39_extract_source_snapshot_260531.osm.pbf

python3 \
  export_zip39_pois.py \
  work/osm_zip39_features.geojsonseq \
  work/osm_zip39_poi_candidate.csv.gz

gzip --decompress --stdout \
  work/osm_zip39_poi_candidate.csv.gz \
  > osm_geofabrik_nrw_zip39_poi_including_bakeries_source_snapshot_260531.csv
```

## 8. Verify the outputs

```bash
sha256sum \
  osm_geofabrik_nrw_zip39_extract_source_snapshot_260531.osm.pbf \
  osm_geofabrik_nrw_zip39_poi_including_bakeries_source_snapshot_260531.csv

python3 - <<'PY'
import csv

with open("osm_geofabrik_nrw_zip39_poi_including_bakeries_source_snapshot_260531.csv", encoding="utf-8", newline="") as source:
    rows = list(csv.DictReader(source))

print("POI_ROW_COUNT=" + str(len(rows)))
print("SHOP_BAKERY_ROW_COUNT=" + str(sum(row.get("shop") == "bakery" for row in rows)))
PY
```

Expected values for the supplied files:

```text
PBF_SHA256=6425EB1C5832F5569A39F1FF1A2A05B6C66F49C93890644EFFAF15282E5EFD67
POI_CSV_SHA256=6888325638722CC2896F50FE607CA9A74F258241400EB9ECBB49BF98E7491AAC
POI_ROW_COUNT=84186
SHOP_BAKERY_ROW_COUNT=364
```

## Minimal status note

This is a `NON_FINAL` internal candidate. Relation-reference completeness remains `TODO-VERIFY`.

OpenStreetMap attribution:

```text
OpenStreetMap contributors
```

License reference:

```text
https://www.openstreetmap.org/copyright
```

