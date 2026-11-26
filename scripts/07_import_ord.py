"""Import a BOUNDED subset of the Open Reaction Database (ORD) into the JSON-lines format used by chemintel/reactions.py.
ORD data licence: CC BY-SA 4.0 (https://github.com/open-reaction-database/ord-data). Attribution is kept in each record's "source" field.
We do NOT ship the imported data (share-alike); run this script to create it locally.
Needs: pip install pyarrow protobuf ; pip install --no-deps ord-schema   (ord-schema is only used for its protobuf definitions)
  python scripts/07_import_ord.py --dataset data/00/ord_dataset-00005539a1e04c809a9a78647bea649c.parquet --out data/ord_subset.jsonl --limit 300
  (the script downloads the file from the ord-data repo if --dataset is a repo path such as data/00/ord_dataset-....parquet)"""
import argparse, hashlib, json, os, sys, urllib.request

MEDIA = "https://media.githubusercontent.com/media/open-reaction-database/ord-data/main/"
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, help="repo path like data/00/ord_dataset-<id>.parquet, or a local parquet file")
ap.add_argument("--out", default="data/ord_subset.jsonl")
ap.add_argument("--limit", type=int, default=300)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
local = a.dataset
if not os.path.exists(local):
    local = os.path.join("data", "ord_download", os.path.basename(a.dataset)); os.makedirs(os.path.dirname(local), exist_ok=True)
    urllib.request.urlretrieve(MEDIA + a.dataset, local)
sha = hashlib.sha256(open(local, "rb").read()).hexdigest()
import pyarrow.parquet as pq
from ord_schema.proto import reaction_pb2
TEMP_UNITS = {1: "C", 2: "K", 3: "F"}
rows = pq.read_table(local).slice(0, a.limit).to_pylist()
n = 0
with open(a.out, "w", encoding="utf-8") as f:
    for row in rows:
        r = reaction_pb2.Reaction(); r.ParseFromString(row["reaction"])
        rtype = next((i.value for i in r.identifiers if i.type == 6 or "[" in i.value), "")  # e.g. "1.3.1 [N-arylation with Ar-X] ..."
        parts = []; solvents = []
        for role, inp in r.inputs.items():
            smi = [c.identifiers[0].value for c in inp.components if c.identifiers]
            parts.append(f"{role}: " + " . ".join(smi))
            if "solvent" in role.lower(): solvents += smi
        prods = []; yld = ""
        for o in r.outcomes:
            for p in o.products:
                prods.append(" . ".join(i.value for i in p.identifiers[:1]))
                for m in p.measurements:
                    if m.HasField("percentage") and m.percentage.value and not yld: yld = f"{m.percentage.value:.1f} % ({reaction_pb2.ProductMeasurement.ProductMeasurementType.Name(m.type)})"
        cond = []
        if r.conditions.temperature.setpoint.value:
            cond.append(f"{r.conditions.temperature.setpoint.value:g} {TEMP_UNITS.get(r.conditions.temperature.setpoint.units, '?')}")
        rec = {"id": row["reaction_id"], "reactants": "; ".join(parts), "products": "; ".join(prods),
               "conditions": " | ".join([x for x in [rtype] + cond if x]), "yield": yld,
               "source": f"ORD {row['reaction_id']} from {os.path.basename(local)} (CC BY-SA 4.0, https://github.com/open-reaction-database/ord-data)"}
        f.write(json.dumps(rec) + "\n"); n += 1
print(json.dumps({"records": n, "dataset_file": os.path.basename(local), "sha256": sha, "out": a.out}))
