Food Prediction - Destatis GV100AD Store-Municipality Candidate - NON_FINAL

Created UTC: 2026-06-07T12:07:38Z
Artifact role: Candidate external enrichment only.
Repository status: Isolated candidate repo intake. No source promotion.
Customer delivery status: Blocked / TODO-VERIFY.
Downstream model use: Blocked unless separately gated.

1. Source identity
------------------
Source organisation: Statistisches Bundesamt (Destatis)
Source product family: Gemeindeverzeichnis / GV100AD municipal-directory data
Structured source file: AuszugGV4QAktuell.xlsx
Structured source reference date: 2025-12-31 (4th quarter)

Important temporal distinction:
- Municipality territorial status: 2025-12-31.
- Area, population, postcode, centroid coordinates and related fields use the source worksheet reference semantics.
- No predictive-value, mapping-safety or leakage-safety conclusion is made.

2. Usage-terms supporting evidence
----------------------------------
Supporting PDF: GV-ISys_StatistischesBundesamt.pdf
Supporting PDF SHA256: 4691284CD5F800089A846E0E95CD42F205A0F543038BC01A24E552F06B4C6208

The attached Destatis response states that reproduction and distribution, including partial distribution, are permitted with source attribution.

This provenance note does not claim customer-delivery approval or legal suitability.

3. Lineage
----------
Structured source XLSX:
- Path: C:\Users\simon\food_prediction_quarantine\gv100ad_cycle_2025_01_01_to_2026_03_31\AuszugGV4QAktuell.xlsx
- SHA256: 9409B968FDFC58C6E57237DB556B48CCD31C845308BA046E0471E3FDE149935E

NRW-only quarantine intermediate:
- Path: C:\Users\simon\food_prediction_quarantine\gv100ad_cycle_2025_01_01_to_2026_03_31\gv100ad_31122025_nrw_csv_non_final\destatis_gv100ad_municipalities_nrw_2025-12-31_NON_FINAL.csv
- SHA256: 7B485F072BFAEAE75459C1273CC69DE7E58A671402EF07E56AE555A291F19BEB
- Row count: 396 NRW municipalities

Project store-municipality reference:
- Path: C:\Users\simon\food_prediction\raw_data\code_external_data\_external_data\store_geography\store_municipality_reference.csv
- SHA256: 25B24082E32508486C36677918D14512EBB01F3D3F6259F4322A82B1913D58DC
- Usage: Read-only AGS filter source

Derived project-municipality candidate CSV:
- Filename: destatis_gv100ad_store_municipalities_2025-12-31_NON_FINAL.csv
- Repository-candidate SHA256: 61F7EB1D04415343A9698C76D79D74C2C640FC0188FA3B81C252C163449382B7
- Quarantine-derived LF variant SHA256: CAC6A4A042AC61AB1BAC88D1CE2A93D661F2EEE8BE216ECE49C8857C569B8D03
- Structural QA note: Read-only comparison verified normalized UTF-8 text identity. The repository candidate uses CRLF line endings; the quarantine-derived variant uses LF line endings. Neither CSV was modified in the README-repair slice.
- Row count: 25 unique municipalities
- Filter key: municipality_ags
- Transformation: Select the distinct municipality_ags values required by store_municipality_reference.csv from the NRW-only GV100AD candidate CSV.

4. Minimal QA result
--------------------
- NRW source rows: 396
- Required unique store municipality AGS values: 25
- Derived rows: 25
- Missing required municipality AGS values: 0
- Duplicate derived municipality AGS values: 0
- AGS filter uses municipality_ags, not administrative-seat postcode.

5. Open TODO-VERIFY
-------------------
- Stable public source URL and reproducible acquisition method.
- Exact HTTP download timestamp.
- Delivery readiness.
- Source promotion.
- Downstream join approval.
- Mapping-safety conclusion.
- Leakage-safety conclusion.
- Predictive-value conclusion.

6. Restrictions
---------------
- Keep this artifact separate from canonical raw concepts: sales, stores, weather and holidays.
- Do not treat administrative-seat postcode as a complete postcode-to-municipality mapping.
- Do not overwrite canonical raw data.
- Do not use this candidate for downstream modeling unless separately approved and evidenced.