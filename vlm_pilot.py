"""Decide which classes the vision-model check may be switched on for.

For every object of the chosen classes this asks the model (vlm_cues.up_score)
about the six ways up, exactly as the vlm_check rule step does, and reports per
class and per margin how many objects would be FIXED (more than 45 degrees from
the class consensus before, within 45 after) and how many BROKEN (the other way
round).  The consensus and the error are measured the way class_report.py
measures consistency: frame relative to the stored pose, from one random input
rotation, modulo the symmetry group.

Half A = even files (tune here), half B = odd files (confirm here).  Switch a
class on only if, at the same margin, it fixes more than it breaks on BOTH
halves and breaks nothing on half A.  Answers are cached in vlm_cache/, so the
later class_report run reuses them.

    python vlm_pilot.py --classes chair,monitor,sofa --half A
    python vlm_pilot.py --classes chair,monitor,sofa --half B
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import geo_canon as g
import vlm_cues as vc

MARGINS = (0.5, 1.0, 2.0, 3.0, 4.0)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", default="processed_data")
ap.add_argument("--rules", default="rules.json", help="the rules the check is added on top of")
ap.add_argument("--classes", default="chair,monitor,car,sofa,bench,piano,lamp,toilet,bathtub,bed,guitar,laptop")
ap.add_argument("--half", default="A", choices=["A", "B"])
ap.add_argument("--max", type=int, default=0, help="at most this many objects per class (0 = all)")
ap.add_argument("--out", default=None, help="JSON with per-object scores")
args = ap.parse_args()

g.configure_rules(args.rules)
canon = g.make_canonicaliser({}, refine_classes=g.REFINE_CLASSES)
ds = g.load_real(Path(args.data), 100, 1024, 0)
turns = vc._up_turns()
report, rows = {}, []
for cls in [c.strip() for c in args.classes.split(",") if c.strip()]:
    if cls not in ds:
        print(f"{cls}: no data"); continue
    clouds = ds[cls][0]
    rng = np.random.default_rng(977 * g.CLASS_ORDER.index(cls))
    F, G, P = [], [], []
    for i, X in enumerate(clouds):
        A = g.random_rotations(1, rng)[0]
        if i % 2 != (0 if args.half == "A" else 1):
            continue
        R, info = canon(X @ A.T, cls)
        F.append(R @ A); G.append(info.get("symmetry", "I"))
        P.append(g.normalise_cloud(X @ A.T)[0] @ R.T)
    if args.max:
        F, G, P = F[:args.max], G[:args.max], P[:args.max]
    _, err, M = g.dispersion_deg(F, G)
    t0 = time.time()
    per = []
    for Pi, Fi, gi, e in zip(P, F, G, err):
        cands = [Pi @ O.T for O in turns]
        lim = vc._shared_lim(cands)
        sc = [vc.up_score(C, cls.replace("_", " "), lim) for C in cands]
        k = int(np.argmax(sc))
        e_new = g.group_distance_deg(turns[k] @ Fi, M, gi)
        per.append((sc[k] - sc[0], k, float(e), float(e_new)))
        rows.append({"cls": cls, "scores": sc, "err": float(e), "err_if_switched": float(e_new)})
    res = {}
    for m in MARGINS:
        sw = [(e0, e1) for gain, k, e0, e1 in per if k != 0 and gain > m]
        res[m] = {"switched": len(sw), "fixed": sum(e0 > 45 >= e1 for e0, e1 in sw),
                  "broken": sum(e1 > 45 >= e0 for e0, e1 in sw)}
    report[cls] = {"n": len(per), "gross": int((err > 45).sum()), "margins": res}
    print(f"{cls:10s} n={len(per):2d} gross={int((err > 45).sum()):2d} | " +
          "  ".join(f"m{m}: +{v['fixed']} -{v['broken']}" for m, v in res.items()) +
          f"   ({time.time() - t0:.0f}s)", flush=True)
if args.out:
    Path(args.out).write_text(json.dumps({"report": report, "rows": rows}, indent=1, default=float))
