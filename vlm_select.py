"""Turn the two pilot results into rules_vlm.json, with no hand-picking.

A class gets vlm_check at margin m only if, at that same m:
  half A (tuning):    fixed > broken and broken == 0
  half B (confirm):   fixed > broken
Among the margins that qualify, the one with the most fixes on both halves wins
(ties go to the larger, more cautious margin).  Classes that qualify at no
margin keep the base recipe unchanged.  If no class qualifies, the output is a
copy of the base rules, so the vision report equals the part-check report.

    python vlm_select.py pilot_A.json pilot_B.json --base rules_parts.json --out rules_vlm.json
"""
import argparse
import copy
import json
from pathlib import Path

import geo_canon as g

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("pilot_a")
ap.add_argument("pilot_b")
ap.add_argument("--base", default="rules_parts.json")
ap.add_argument("--out", default="rules_vlm.json")
args = ap.parse_args()

A = json.loads(Path(args.pilot_a).read_text())["report"]
B = json.loads(Path(args.pilot_b).read_text())["report"]
doc = json.loads(Path(args.base).read_text())
by_name = {c["name"]: c for c in doc["classes"]}
chosen = {}
for cls in sorted(set(A) & set(B)):
    best = None
    for m, a in A[cls]["margins"].items():
        b = B[cls]["margins"].get(m)
        if b is None:
            continue
        if a["broken"] == 0 and a["fixed"] > a["broken"] and b["fixed"] > b["broken"]:
            key = (a["fixed"] + b["fixed"], float(m))
            if best is None or key > best[0]:
                best = (key, float(m), a, b)
    if best:
        _, m, a, b = best
        chosen[cls] = m
        print(f"{cls:10s} ON  margin {m}: half A +{a['fixed']} -{a['broken']}, half B +{b['fixed']} -{b['broken']}")
    else:
        print(f"{cls:10s} off (no margin fixes more than it breaks on both halves)")

for cls, m in chosen.items():
    rec = by_name[cls]
    name = f"{rec['recipe']}+vlm_check_{cls}"
    doc["recipes"][name] = copy.deepcopy(doc["recipes"][rec["recipe"]]) + [
        {"op": "vlm_check", "params": {"name": cls, "margin": m}}]
    rec["recipe"] = name
text = json.dumps(doc, indent=2) + "\n"
g.RuleDatabase(json.loads(text), g.OPERATIONS)            # validate before writing
Path(args.out).write_text(text)
print(f"wrote {args.out}: vlm_check on {sorted(chosen) or 'no class (same as ' + args.base + ')'}")
