# Checks on real public data (26 Nov 2026)

These are small manual checks, not benchmarks. Nothing here says the tools are accurate in general.

## 1. Datasheet extractor (chemintel/datasheet.py, regex method) on 3 real public technical data sheets
Sources, downloaded 26 Nov 2026 (check the vendors' current terms before reuse; the PDFs are not stored in this repo):
- A: https://www.valsparindustrialmix.com.au/resources/docs/VIM-FP450_FP451-Epoxy-Primer-TDS-ANZ.pdf (4 pages). I tuned the patterns while looking at this sheet, so it is a development example.
- B: https://www.axalta.com/content/dam/NA/HQ/Public/Transportation/Documents/tds/TR-TDS-RV450-Epoxy-Primer-Eng.pdf (3 pages). Not used for tuning.
- C: https://marinecoatings.brand.akzonobel.com/m/7e348f0069a9ffd8/original/Intergard_7600_eng_A4_20160524.pdf (4 pages). Not used for tuning.

I compared the output with the sheet text (pdftotext -layout) by hand.

| Sheet | Fields in the sheet within the tool's scope | Extracted correctly | Missed or partial | Wrong values |
|---|---|---|---|---|
| A | density 1.75 - 1.77 kg/L, flash point 24 C, pot life 8 h, shelf life 24 months (p1), shelf life 2 years (p4) | all 5 | none | 0 |
| B | pot life 4 hours, shelf life 12 months (flash point says "See MSDS/SDS", correctly not extracted) | shelf life | pot life missed: the value sits on the line after the label | 0 |
| C | flash point (Part A 28 C; Part B 27 C; mixed 29 C), pot life (4 values by component and temperature), shelf life 12 months | shelf life; first flash value (28 C); first pot life value (6 hrs) | the other flash values and pot life values for other columns | 0 |

Reading: no wrong values in these 3 sheets, but it misses values and takes only the first number when a table row has several. 3 sheets is too few to say more.
Known limits: text PDFs only; tables are read with pypdf layout mode (default pypdf mode put the labels and values of sheet A in separate lines and missed density); multi-column rows and values on the next line are not handled; unit and name coverage is small.

## 2. Reaction precedents on a real ORD subset
- Data: first 750 reactions of data/00/ord_dataset-00005539a1e04c809a9a78647bea649c.parquet from https://github.com/open-reaction-database/ord-data (CC BY-SA 4.0), imported with scripts/07_import_ord.py. File sha256 5326c1f79ef6a71e6b2a91cbd07cd8521049427df79307686c7c4f44eb16b53a. The imported data is not stored in this repo (share-alike).
- Content: all 750 are N-arylation (mostly Buchwald-Hartwig) reactions from one source group, so this subset is narrow. 616 of 750 records have a recorded yield value; 134 do not.
- Query checks (3 results each, read by hand):
  - "Buchwald-Hartwig amination of aryl bromide with amine, cesium carbonate": 2 of 3 are bromo Buchwald-Hartwig; the third is a chloro reaction (not a bromide), so retrieval is only approximate.
  - "Iodo Buchwald-Hartwig amination anisole solvent": 3 of 3 are iodo Buchwald-Hartwig reactions.
  - "Suzuki coupling boronic acid palladium": 0 results (no such reactions in this subset). "zzzz qqqq": 0 results.
- The relevance guard requires at least half of the query's content words to occur in a record. This is a crude rule, not a similarity model. Retrieval works on words, not on chemical structure; SMILES strings are not searched structurally.
- Still not a feasibility, yield or safety prediction.
