"""Collect results/*.json into a markdown table. Only numbers present in the result files are reported."""
import glob, json, os, sys
d = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    r = json.load(open(f))
    o = r["overall"]
    bd = r["by_difficulty"]
    def cell(x):
        return f"{x['accuracy']:.3f} (n={x['n']})" if x else "-"
    rows.append(f"| {r['name']} | {r['backend']} | {r.get('model') or '-'} | {r.get('adapter') or '-'} | {'yes' if r['rag'] else 'no'} | "
                f"{r['mcq_questions']} | {o['accuracy']:.3f} [{o['ci95'][0]:.3f}, {o['ci95'][1]:.3f}] | {r['chance_expected']:.3f} | "
                f"{cell(bd.get('basic'))} | {cell(bd.get('advanced'))} |")
print("| run | backend | model | adapter | retrieval | questions | accuracy [95% CI] | chance | basic | advanced |")
print("|---|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
