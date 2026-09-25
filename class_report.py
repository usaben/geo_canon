"""
class_report.py -- stability and consistency for every class, plus figures.

Writes class_report.txt with one row per class, and the pair of figures
geo_canon already makes for its own five (<class>_consistency.png and
<class>_stability.png) for the five added here.

The two numbers answer different questions and a rule can pass one and fail the
other:

  stability    one cloud, many input poses.  Does the frame follow the object
               when the object turns?  With PCA_FIRST this is ~0 by
               construction, so anything else means the rule is throwing and
               falling back, or the PCA pre-frame itself is wobbling.

  consistency  many clouds of one class, each from a random pose.  Do different
               instances land in the SAME frame?  This is the quality number,
               and the hard one.

Run:
    python class_report.py                      # all classes, figures for the new five
    python class_report.py --instances 20 --rotations 8
"""

import argparse
import time
from pathlib import Path

import numpy as np

import geo_canon as g

BUILTIN = ("airplane", "car", "chair", "table", "bowl")
ADDED = ("laptop", "bed", "bottle", "guitar", "sofa")

# Which datasets store their instances in a common pose.  Consistency is
# measured in each object's own stored coordinates, so where storage is not
# azimuth-aligned the number includes the spread of the inputs and overstates
# the error.  Checked by eye, per class.
ALIGNED = {"airplane": True, "car": True, "chair": True, "table": True,
           "bowl": True,            # a surface of revolution: azimuth is moot
           "bed": True, "bottle": True, "guitar": True, "sofa": True,
           "laptop": False}         # ModelNet laptops vary in stored azimuth


def write_simple(path, results):
    """Just the two headline numbers, one row per class, nothing else."""
    L = ["class       stability  consistency", "-" * 36]
    for cls, r in results.items():
        L.append(f"{cls:10s} {r['stability_sym_deg']:10.3f} {r['consistency_sym_deg']:12.3f}")
    L.append("-" * 36)
    L.append(f"{'MEAN':10s} "
             f"{np.mean([r['stability_sym_deg'] for r in results.values()]):10.3f} "
             f"{np.mean([r['consistency_sym_deg'] for r in results.values()]):12.3f}")
    L.append("")
    L.append("degrees; lower is better")
    Path(path).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {path}")


def write_report(path, results, meta):
    L = []
    L.append("stability and consistency by class")
    L.append("=" * 78)
    L.append("")
    L.append(f"generated   {time.strftime('%Y-%m-%d %H:%M')}")
    L.append(f"data        {meta['data']}")
    L.append(f"instances   up to {meta['instances']} per class")
    L.append(f"rotations   {meta['rotations']} random poses per instance (stability)")
    L.append(f"references  {meta['refs']}")
    L.append(f"pca_first   {g.PCA_FIRST}")
    L.append(f"rules       {g.RULE_DATABASE.fingerprint}")
    L.append("")
    L.append("stability   one cloud, many input poses -- does the frame follow the")
    L.append("            object when the object turns?  ~0 is the pass mark.")
    L.append("consistency many clouds of one class, each from a random pose -- do")
    L.append("            different instances land in the same frame?  The hard one.")
    L.append("sym         measured modulo the detected symmetry group.")
    L.append("")
    L.append("-" * 78)
    L.append(f"{'class':10s} {'n':>3s} {'stab sym':>9s} {'cons sym':>9s} "
             f"{'cons med':>9s} {'<10deg':>7s} {'group':>7s} {'rule':>9s}")
    L.append("-" * 78)

    for cls, r in results.items():
        grp = max(set(r["symmetry_groups"]), key=r["symmetry_groups"].count)
        kind = "built-in" if cls in BUILTIN else "added"
        L.append(f"{cls:10s} {r['n_instances']:3d} {r['stability_sym_deg']:9.3f} "
                 f"{r['consistency_sym_deg']:9.3f} "
                 f"{r['consistency_sym_median_deg']:9.3f} "
                 f"{100 * r['consistency_within10']:6.0f}% {grp:>7s} {kind:>9s}")
    L.append("-" * 78)
    n = len(results)
    if n:
        L.append(f"{'MEAN':10s} {'':3s} "
                 f"{np.mean([r['stability_sym_deg'] for r in results.values()]):9.3f} "
                 f"{np.mean([r['consistency_sym_deg'] for r in results.values()]):9.3f} "
                 f"{np.mean([r['consistency_sym_median_deg'] for r in results.values()]):9.3f} "
                 f"{100 * np.mean([r['consistency_within10'] for r in results.values()]):6.0f}%")
    L.append("")
    L.append("Read the median as well as the RMS column: RMS squares the errors, so a")
    L.append("single inverted instance in twenty shows up as roughly 40 degrees and")
    L.append("hides the fact that the rest agree closely.")
    L.append("")

    unaligned = [c for c in results if not ALIGNED.get(c, True)]
    if unaligned:
        L.append("CAVEAT -- consistency assumes aligned storage")
        L.append("-" * 78)
        L.append("Consistency compares frames in each object's OWN stored coordinates,")
        L.append("which only means what it should if the dataset stores every instance")
        L.append("in a common pose.  ShapeNet does.  ModelNet is upright-aligned but")
        L.append("not azimuth-aligned, and these classes are affected:")
        L.append("")
        for c in unaligned:
            L.append(f"    {c}: stored azimuth varies between instances, so the")
            L.append(f"    {'':{len(c)}s}  consistency figure below includes that spread and")
            L.append(f"    {'':{len(c)}s}  overstates the rule's error.")
        L.append("")

    L.append("per-instance consistency, degrees from the consensus frame")
    L.append("-" * 78)
    for cls, r in results.items():
        per = np.array(r["per_instance_consistency_deg"])
        bad = int((per > 45).sum())
        L.append(f"{cls:10s} median {np.median(per):6.2f}   worst {per.max():6.2f}   "
                 f"instances over 45 deg: {bad}/{len(per)}")
        if bad and bad < len(per):
            keep = per[per <= 45]
            L.append(f"{'':10s} excluding those: median {np.median(keep):6.2f}, "
                     f"max {keep.max():6.2f}")
    L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="processed_data")
    ap.add_argument("--rules", type=Path, default=g.DEFAULT_RULES_PATH,
                    help="JSON rule database")
    ap.add_argument("--instances", type=int, default=16)
    ap.add_argument("--points", type=int, default=1024)
    ap.add_argument("--rotations", type=int, default=6)
    ap.add_argument("--refs", default="refs.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="class_report.txt")
    ap.add_argument("--simple-out", default="stability_consistency.txt",
                    help="the same two headline numbers with nothing around them")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    g.configure_rules(args.rules)

    data_path = Path(args.data)
    if not data_path.exists() and not data_path.is_absolute():
        script_relative = Path(__file__).resolve().parent / data_path
        if script_relative.exists():
            data_path = script_relative
    if not data_path.is_dir():
        ap.error(f"data directory not found: {data_path}")

    ds = g.load_real(data_path, args.instances, args.points, args.seed)
    print("loaded: " + ", ".join(f"{c}={len(v[0])}" for c, v in ds.items()))
    if not ds:
        ap.error(f"no usable class data found in {data_path.resolve()}")

    refs = {}
    if args.refs and Path(args.refs).is_file():
        refs, rmeta = g.load_references(args.refs)
        g.PCA_FIRST = bool(rmeta.get("pca_first", g.PCA_FIRST))
        print(f"references: {sorted(refs)}  pca_first={g.PCA_FIRST}")
    canon = g.make_canonicaliser(refs, refine_classes=g.REFINE_CLASSES)

    print("\nevaluating")
    results, _, _ = g.evaluate(ds, canon, args.rotations, args.seed,
                               want_features=False)

    write_report(args.out, results, {"data": data_path.resolve(),
                                     "instances": args.instances,
                                     "rotations": args.rotations,
                                     "refs": args.refs if refs else "none"})
    write_simple(args.simple_out, results)

    if not args.no_figures:
        want = {c: ds[c] for c in ADDED if c in ds}
        if want:
            print(f"\nfigures for {', '.join(want)}")
            g.make_figures(want, canon, args.figures, args.seed,
                           n_show=min(16, args.instances), n_rot=5)


if __name__ == "__main__":
    main()
