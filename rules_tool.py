"""Append a rule step (part_check / vlm_check) to chosen classes of a rules file.

    python rules_tool.py --base rules_parts.json --op vlm_check \
        --classes chair,monitor --params '{"margin": 2.0}' --out rules_vlm.json

Each chosen class gets its own copy of its recipe with the step appended, so
classes that share a recipe are unaffected.  vlm_check also gets name=<class>.
The result is validated by the rule database before it is written.
"""
import argparse
import copy
import json
from pathlib import Path

import geo_canon as g

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--base", default="rules.json")
ap.add_argument("--op", required=True, choices=["part_check", "vlm_check"])
ap.add_argument("--classes", required=True, help="comma-separated")
ap.add_argument("--params", default="{}", help="JSON object of step parameters")
ap.add_argument("--out", required=True)
args = ap.parse_args()

doc = json.loads(Path(args.base).read_text())
by_name = {c["name"]: c for c in doc["classes"]}
for cls in [c.strip() for c in args.classes.split(",") if c.strip()]:
    rec = by_name[cls]
    params = json.loads(args.params)
    if args.op == "vlm_check":
        params.setdefault("name", cls)
    name = f"{rec['recipe']}+{args.op}_{cls}"
    doc["recipes"][name] = copy.deepcopy(doc["recipes"][rec["recipe"]]) + [{"op": args.op, "params": params}]
    rec["recipe"] = name
text = json.dumps(doc, indent=2) + "\n"
g.RuleDatabase(json.loads(text), g.OPERATIONS)            # validate before writing
Path(args.out).write_text(text)
print("wrote", args.out)
