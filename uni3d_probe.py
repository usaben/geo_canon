"""
uni3d_probe.py -- turn Uni3D-S into a 5-way classifier without CLIP.

Uni3D's own zero-shot route embeds class *names* with EVA02-E-14-plus and
compares them to the point embedding.  That CLIP checkpoint is 10.1 GB, and for
a fixed set of classes it buys nothing we cannot get from the labels already on
disk: embed a few clouds per class, average, and the class centroid plays the
part the text embedding would have.  Cosine-nearest centroid over L2-normalised
features is then the classifier.

    python uni3d_probe.py                 # build centroids, report accuracy
    python uni3d_probe.py --rotate-test   # does canonicalisation buy accuracy?

The centroids land in uni3d_centroids.npz, which demo_app.py --classify reads.

Centroids are built from the FIRST --ref-instances clouds of each class, the
same ones geo_canon's references are built from, so `demo_app.py --skip N`
keeps showing only clouds that neither the references nor the centroids saw.
"""

import argparse
import time
from pathlib import Path

import numpy as np

import geo_canon as g
import uni3d_cpu

OUT = "uni3d_centroids.npz"


def centroids_from(feats, labels, n_classes, drop=None):
    """Mean feature per class, renormalised.  `drop` excludes one row (for LOO)."""
    out = []
    for c in range(n_classes):
        sel = labels == c
        if drop is not None:
            sel = sel.copy()
            sel[drop] = False
        v = feats[sel].mean(0)
        out.append(v / max(np.linalg.norm(v), 1e-9))
    return np.stack(out)


def embed_all(model, ds, classes, limit, swap_yz, tag="", framed=False):
    feats, labels = [], []
    for ci, c in enumerate(classes):
        clouds = ds[c][0][:limit]
        t0 = time.time()
        feats.append(uni3d_cpu.encode(model, clouds, swap_yz=swap_yz, framed=framed))
        labels += [ci] * len(clouds)
        print(f"  {tag}{c:9s} {len(clouds):3d} clouds  ({time.time() - t0:.0f}s)")
    return np.concatenate(feats), np.array(labels)


def loo_accuracy(feats, labels, n_classes):
    """Leave-one-out: each cloud is scored against centroids built without it."""
    hits = sum(int(np.argmax(centroids_from(feats, labels, n_classes, drop=i) @ feats[i])
                   == labels[i]) for i in range(len(feats)))
    return hits / len(feats)


# ---------------------------------------------------------------------------

def build(args, model, ds, classes):
    print(f"\nembedding {args.ref_instances} cloud(s) per class for the centroids")
    feats, labels = embed_all(model, ds, classes, args.ref_instances, args.swap_yz,
                              framed=args.framed)
    cents = centroids_from(feats, labels, len(classes))

    acc = loo_accuracy(feats, labels, len(classes))
    print(f"\nleave-one-out nearest-centroid accuracy: {acc:.1%} "
          f"({len(feats)} clouds, {len(classes)} classes)")

    # How far apart the classes sit -- a sanity number worth having in the report.
    sim = cents @ cents.T
    off = sim[~np.eye(len(classes), dtype=bool)]
    print(f"centroid cosine similarity: mean off-diagonal {off.mean():.3f}, "
          f"max {off.max():.3f}")

    np.savez(OUT, centroids=cents, classes=np.array(classes), framed=args.framed,
             swap_yz=args.swap_yz, n_ref=args.ref_instances, loo_accuracy=acc)
    print(f"\nwrote {OUT}  ({len(classes)} centroids, dim {cents.shape[1]})")
    return cents


def rotate_test(args, model, ds, classes, cents):
    """The thesis claim, measured.

    Uni3D has no rotation invariance of its own -- nothing in the architecture
    provides it and nothing in the training data asks for it.  So a cloud in a
    pose it was never shown should classify worse, and putting that cloud back
    into a canonical frame first should recover the loss.  Three conditions,
    same clouds, same centroids.
    """
    refs, rmeta = g.load_references(args.refs)
    g.PCA_FIRST = bool(rmeta.get("pca_first", g.PCA_FIRST))
    canon = g.make_canonicaliser(refs, refine_classes=g.REFINE_CLASSES)
    print(f"\nreferences from {args.refs}: "
          f"{ {c: len(r['clusters']) for c, r in refs.items()} }, pca_first={g.PCA_FIRST}")

    rng = np.random.default_rng(args.seed)
    rows = {"as stored": [], "randomly rotated": [], "rotated, then canonicalised": []}

    for ci, c in enumerate(classes):
        # clouds the centroids never saw
        clouds = ds[c][0][args.ref_instances:args.ref_instances + args.test_instances]
        if not len(clouds):
            continue
        rots = g.random_rotations(len(clouds), rng)
        plain = list(clouds)
        turned = [X @ R.T for X, R in zip(clouds, rots)]
        fixed = []
        for X in turned:
            R, _ = canon(X, c)
            fixed.append(X @ R.T)
        for name, batch in (("as stored", plain), ("randomly rotated", turned),
                            ("rotated, then canonicalised", fixed)):
            F = uni3d_cpu.encode(model, batch, swap_yz=args.swap_yz,
                                 framed=args.framed)
            rows[name] += (np.argmax(F @ cents.T, axis=1) == ci).tolist()
        print(f"  {c:9s} done ({len(clouds)} clouds)")

    print("\n  condition                      accuracy")
    print("  " + "-" * 42)
    for name, hits in rows.items():
        print(f"  {name:30s} {np.mean(hits):6.1%}  ({sum(hits)}/{len(hits)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="processed_data")
    ap.add_argument("--points", type=int, default=1024)
    ap.add_argument("--ref-instances", type=int, default=12,
                    help="clouds per class averaged into each centroid")
    ap.add_argument("--test-instances", type=int, default=10,
                    help="held-out clouds per class for --rotate-test")
    ap.add_argument("--swap-yz", action="store_true",
                    help="Uni3D's OpenShape axis convention; measured worse on "
                         "ShapeNet-frame data, so off by default")
    ap.add_argument("--no-framed", dest="framed", action="store_false",
                    help="embed clouds as they arrive instead of in their PCA "
                         "frame; Uni3D is not rotation invariant, so framing "
                         "first measurably improves naming of turned clouds")
    ap.set_defaults(framed=True)
    ap.add_argument("--rotate-test", action="store_true")
    ap.add_argument("--refs", default="refs.npz")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    need = args.ref_instances + (args.test_instances if args.rotate_test else 0)
    ds = g.load_real(Path(args.data), need, args.points, args.seed)
    classes = sorted(ds)
    print(f"loaded {args.data}: " + ", ".join(f"{c}={len(ds[c][0])}" for c in classes))

    print("loading Uni3D-S")
    model = uni3d_cpu.load(args.ckpt)

    cents = build(args, model, ds, classes)
    if args.rotate_test:
        rotate_test(args, model, ds, classes, cents)


if __name__ == "__main__":
    main()
