"""
demo_app.py -- a browser demo for the canonicalisation pipeline.
"""

import argparse
import json
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import geo_canon as g

STATE = {"data": {}, "canon": None, "refs": {}, "lock": threading.Lock()}


# ---------------------------------------------------------------------------
# Zero-ML Multi-Variant PCA Overlay Predictor
# ---------------------------------------------------------------------------

CLASS_TREES = {}

def extract_points_from_proto(proto, cls_name):
    """Safely extracts an (N, 3) coordinate array from cluster metadata."""
    if isinstance(proto, np.ndarray) and proto.ndim == 2 and proto.shape[1] == 3:
        return proto
    if isinstance(proto, dict):
        for key in ("points", "X", "cloud", "template", "canon", "exemplar", "proto"):
            val = proto.get(key)
            if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] == 3:
                return val
        for idx_key in ("idx", "exemplar_idx", "cloud_idx", "id"):
            val = proto.get(idx_key)
            if isinstance(val, (int, np.integer)) and cls_name in STATE.get("data", {}):
                clouds, _ = STATE["data"][cls_name]
                if 0 <= val < len(clouds):
                    return clouds[val]
    return None


CLOUD_EXT = {".pt", ".npy", ".npz"} | g.MESH_EXT


def build_class_signatures():
    """Precomputes PCA-aligned KDTrees for every class prototype."""
    global CLASS_TREES
    CLASS_TREES = {}
    for cls_name, ref_data in STATE["refs"].items():
        if not ref_data: 
            continue
            
        proto_pts = None
        # clusters is keyed by kmeans label, and build_reference drops a label
        # whose cluster held a single instance, so 0 need not be one of them.
        # Each value is a list of aligned clouds; the first is the exemplar.
        clusters = ref_data.get("clusters") or {}
        for key in sorted(clusters):
            members = clusters[key]
            if isinstance(members, (list, tuple)):
                members = members[0] if members else None
            proto_pts = extract_points_from_proto(members, cls_name)
            if proto_pts is not None:
                break

        if proto_pts is None and cls_name in STATE["data"]:
            proto_pts = STATE["data"][cls_name][0][0]
            
        if proto_pts is not None:
            # 1. Align the reference prototype to its PCA axes
            pts = proto_pts - np.mean(proto_pts, axis=0)
            _, _, Vt = np.linalg.svd(pts, full_matrices=False)
            aligned_proto = pts @ Vt.T
            
            # 2. Store a KDTree of the aligned prototype for ultra-fast overlay checking
            CLASS_TREES[cls_name] = cKDTree(aligned_proto)
            
    print(f"built PCA-overlay trees for {len(CLASS_TREES)} class(es)")


def predict_category(cloud):
    """Name the class of one cloud -> (class, confidence, how).

    This is the hinge of the demo: whatever comes back here is the class the
    canonicaliser is then asked for, so it decides which reference the cloud is
    snapped to.  Nothing downstream is told the true label.

    Uni3D-S does the naming when --classify loaded it, and it can name every
    class in the centroid file.  The PCA-overlay fallback below can only name
    classes that have a reference prototype, so with the default
    REFERENCE_CLASSES it will never answer bowl or table.
    """
    if STATE.get("uni3d") is not None:
        f = STATE["encode"](STATE["uni3d"], np.asarray(cloud, dtype=np.float32),
                            swap_yz=STATE["swap_yz"], framed=STATE.get("framed", False))
        logits = STATE["scale"] * (STATE["cent"] @ f)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        i = int(np.argmax(p))
        return STATE["cent_classes"][i], float(p[i]), "uni3d"

    cls = _predict_by_overlay(cloud)
    return cls, None, "pca-overlay"


def _predict_by_overlay(cloud):
    """
    Predicts category by checking all 16 possible PCA axis orientations
    against the reference prototypes and picking the tightest physical fit.
    """
    if not CLASS_TREES:
        build_class_signatures()
        if not CLASS_TREES:
            return "unknown"

    # 1. Align target cloud to its PCA axes
    pts = cloud - np.mean(cloud, axis=0)
    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    base_aligned = pts @ Vt.T
    
    # 2. Subsample to 256 points for <2ms execution time
    if len(base_aligned) > 256:
        idx = np.random.choice(len(base_aligned), 256, replace=False)
        base_aligned = base_aligned[idx]
        
    # 3. Generate the 16 possible SVD axis flips and swaps
    variants = []
    for fx in (1, -1):
        for fy in (1, -1):
            for fz in (1, -1):
                v = base_aligned * np.array([fx, fy, fz])
                variants.append(v)
                variants.append(v[:, [1, 0, 2]]) # Swap X and Y for symmetric objects (like square tables)
    
    best_cls = None
    best_dist = float('inf')
    
    # 4. Find the class prototype with the smallest physical distance to any variant
    for cls_name, tree in CLASS_TREES.items():
        for variant in variants:
            dists, _ = tree.query(variant, k=1)
            dist = float(np.mean(dists))
            if dist < best_dist:
                best_dist = dist
                best_cls = cls_name
                
    return best_cls or list(STATE["data"].keys())[0]


# ---------------------------------------------------------------------------
# startup: load clouds and build the class references once
# ---------------------------------------------------------------------------

def prepare(args):
    print(f"class rules: {', '.join(g.CLASS_ORDER)}")

    root = Path(args.data)
    total = args.instances + args.skip
    if args.synthetic or not root.is_dir():
        print("using procedural shapes")
        ds = g.load_synthetic(total, args.points, args.seed)
    else:
        print(f"loading from {root.resolve()}")
        ds = g.load_real(root, total, args.points, args.seed, args.mesh_points)
    if not ds:
        print("no data found; falling back to procedural shapes")
        ds = g.load_synthetic(total, args.points, args.seed)

    if args.skip:
        ds = {c: (clouds[args.skip:], ids[args.skip:]) for c, (clouds, ids) in ds.items()}

    STATE["data"] = {c: ([g.normalise_cloud(X)[0] for X in clouds], list(ids))
                     for c, (clouds, ids) in ds.items()}

    # Clouds added by hand or kept from an earlier upload.  They join the class
    # their filename names -- bottle_kitchen.npy is a bottle -- so they get that
    # class's rule and sit beside the ShapeNet instances for comparison.  Their
    # ids are prefixed, because the point of having them here is to see whether
    # the added one lands in the same attitude as the rest, and that is
    # impossible if you cannot tell them apart.
    if args.clouds:
        pdir = Path(args.clouds)
        added = 0
        for f in sorted(pdir.glob("*.npy")) if pdir.is_dir() else []:
            cls = f.stem.split("_")[0]
            if cls not in STATE["data"]:
                print(f"  [warn] {f.name}: no rule for class '{cls}', skipped")
                continue
            try:
                X = np.load(f).astype(float)[:, :3]
            except Exception as exc:
                print(f"  [warn] {f.name}: {exc}")
                continue
            clouds, ids = STATE["data"][cls]
            clouds.insert(0, g.normalise_cloud(X)[0])
            ids.insert(0, f"added:{f.stem}")
            added += 1
        if added:
            print(f"loaded {added} added cloud(s) from {pdir.resolve()}")
        elif pdir.is_dir():
            print(f"no .npy clouds under {pdir.resolve()}")
        else:
            print(f"[warn] --clouds {pdir} is not a directory")

    for c, (clouds, _) in STATE["data"].items():
        print(f"  {c:9s} {len(clouds)} clouds")

    refs = {}
    if args.refs:
        refs, rmeta = g.load_references(args.refs)
        g.PCA_FIRST = bool(rmeta.get("pca_first", g.PCA_FIRST))
    elif not args.no_reference:
        for cls, (clouds, _) in STATE["data"].items():
            if cls not in g.REFERENCE_CLASSES:
                continue
            refs[cls] = g.build_reference(clouds, cls, g.canonicalise,
                                          max_ref=min(args.ref_instances, len(clouds)),
                                          seed=args.seed)
    STATE["clouds_dir"] = args.clouds or "photos"
    STATE["points"] = args.points
    STATE["mesh_points"] = args.mesh_points
    STATE["refs"] = refs
    STATE["canon"] = g.make_canonicaliser(refs, refine_classes=g.REFINE_CLASSES)

    if args.classify:
        import uni3d_cpu
        z = np.load(args.centroids, allow_pickle=True)
        STATE["cent"] = z["centroids"]
        STATE["cent_classes"] = [str(c) for c in z["classes"]]
        STATE["swap_yz"] = bool(z["swap_yz"])
        # Query clouds must be embedded exactly the way the centroids were, or
        # the comparison is between two different representations.
        STATE["framed"] = bool(z["framed"]) if "framed" in z else False
        print(f"loading Uni3D-S (centroids from {args.centroids}: "
              f"{', '.join(STATE['cent_classes'])})")
        STATE["uni3d"] = uni3d_cpu.load(args.uni3d_ckpt)
        STATE["encode"] = uni3d_cpu.encode
        # the model's own trained temperature, so the confidences on screen are
        # on the same scale Uni3D itself uses to compare against text embeddings
        STATE["scale"] = float(STATE["uni3d"].logit_scale.exp())

    build_class_signatures()


# ---------------------------------------------------------------------------
# http
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)

        if url.path in ("/", "/index.html"):
            return self._send(PAGE.encode(), ctype="text/html; charset=utf-8")

        if url.path == "/api/index":
            return self._send({c: ids for c, (_, ids) in STATE["data"].items()})

        if url.path == "/api/predictable":
            # the classes 'auto' can actually name: the centroid file's when Uni3D
            # is loaded, otherwise every loaded class
            return self._send({"classes": STATE.get("cent_classes") or list(STATE["data"])})

        if url.path == "/api/cloud":
            cls = q.get("cls", [""])[0]
            idx = int(q.get("idx", ["0"])[0])
            clouds, ids = STATE["data"].get(cls, ([], []))
            if not clouds:
                return self._send({"error": "unknown class"}, 404)
            idx = max(0, min(idx, len(clouds) - 1))
            X = clouds[idx]
            return self._send({"points": np.round(X, 4).ravel().tolist(),
                               "n": len(X), "id": str(ids[idx])})

        return self._send({"error": "not found"}, 404)

    def _upload(self):
        """Take a point cloud or a mesh posted from the browser, add it as a
        new object, and answer with where it landed.

        Accepts .pt / .npy / .npz clouds and .obj / .ply geometry; a mesh is
        sampled to --mesh-points over its surface, a .ply of bare vertices is
        read as it stands.

        The file arrives as the raw request body rather than as multipart, since
        a Blob can be posted directly by fetch and BaseHTTPRequestHandler has no
        multipart parser worth the trouble.

        The class may be left to the pipeline.  With cls=auto the cloud is named
        by predict_category -- the same Uni3D step the demo uses everywhere else
        -- and filed under whatever comes back, so an upload is not required to
        know what it is.  The confidence travels back with the answer so the
        page can say when it is shaky, which matters because the name is what
        picks the rule: a bottle called a bowl is signed 'up' towards the wide
        end by rule_bowl where rule_bottle signs it towards the narrow one, and
        it lands upside down.

        The cloud is APPENDED, never inserted.  The page holds an index per
        visible object and sends those back with /api/canon, so putting a new
        cloud at the front would silently shift every one of them by one and
        canonicalise the wrong objects.
        """
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        want = q.get("cls", ["auto"])[0]
        raw_name = q.get("name", ["upload"])[0]
        suffix = Path(raw_name).suffix.lower()
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(raw_name).stem)[:40] or "upload"
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > 200 * 1024 * 1024:
            return self._send({"error": "empty or oversized upload (200 MB max)"}, 400)
        if suffix not in CLOUD_EXT:
            return self._send({"error": f"unsupported file type {suffix or '?'}; "
                                        f"give me one of "
                                        f"{', '.join(sorted(CLOUD_EXT))}"}, 400)
        blob = self.rfile.read(n)

        updir = Path(STATE.get("clouds_dir") or "photos")
        updir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        src = updir / f"_incoming_{stamp}{suffix}"
        try:
            src.write_bytes(blob)
            is_geom = suffix in g.MESH_EXT
            # A mesh has no point count of its own, so it is sampled at mesh
            # density and kept there; a stored cloud is thinned to --points,
            # which is the size the rest of the demo works at.
            X = g.load_xyz(src, STATE.get("mesh_points", g.MESH_POINTS), 0)
            if X.ndim != 2 or X.shape[1] < 3 or len(X) < 24:
                raise ValueError(f"expected at least 24 points of (N,>=3), "
                                 f"got {tuple(X.shape)}")
            pts = X if is_geom else g.subsample(X, STATE.get("points", 1024), 0)
            how = f"{'mesh' if is_geom else 'cloud'} {suffix}"
        except Exception as exc:
            src.unlink(missing_ok=True)
            return self._send({"error": f"{type(exc).__name__}: {exc}"}, 400)

        norm = g.normalise_cloud(pts)[0]
        cls, conf, by = want, None, "given"
        if want in ("", "auto", "all") or want not in STATE["data"]:
            try:
                pred, conf, by = predict_category(norm)
            except Exception as exc:
                pred, by = None, f"failed: {type(exc).__name__}"
            if pred in STATE["data"]:
                cls = pred
            elif want not in STATE["data"]:
                src.unlink(missing_ok=True)
                return self._send({"error": f"could not name this cloud "
                                            f"(got {pred!r}); pick a class"}, 400)

        dst = updir / f"{cls}_{name}_{stamp}{suffix}"
        src.replace(dst)
        np.save(updir / f"{cls}_{name}_{stamp}.npy", np.asarray(pts, np.float32))

        with STATE["lock"]:
            clouds, ids = STATE["data"][cls]
            clouds.append(norm)
            ids.append(f"added:{name}")
            idx = len(clouds) - 1
        print(f"  uploaded {dst.name} -> {cls}[{idx}], {len(pts)} pts, {how}"
              + (f", named by {by}" + (f" at {conf:.2f}" if conf else "") if by != "given" else ""))
        return self._send({"cls": cls, "idx": idx, "id": f"added:{name}",
                           "mask": how, "n": len(pts),
                           "predicted_by": None if by == "given" else by,
                           "predicted_conf": None if conf is None else round(conf, 3)})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/upload":
            return self._upload()
        if path != "/api/canon":
            return self._send({"error": "not found"}, 404)
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")

        fallback_cls = req.get("cls", "")
        items = req.get("items") or [{"idx": req.get("idx", 0), "R": req.get("R"), "cls": fallback_cls}]

        out, t0 = [], time.time()
        for it in items:
            # `cls` addresses the cloud and nothing else: idx alone is meaningless,
            # since every class numbers its clouds from zero.  Whether that name is
            # allowed to reach the canonicaliser is a separate question, answered by
            # `auto` below -- in auto mode the name indexes the data and is then
            # thrown away, so the pipeline is still working from a predicted label.
            item_cls = it.get("cls") or fallback_cls
            auto = bool(it.get("auto"))
            idx = max(0, int(it.get("idx", 0)))

            cloud = None
            if item_cls in STATE["data"]:
                clouds, _ = STATE["data"][item_cls]
                cloud = clouds[min(idx, len(clouds) - 1)]
            else:
                for c, (clouds, _) in STATE["data"].items():
                    if idx < len(clouds):
                        cloud = clouds[idx]
                        break

            if cloud is None:
                out.append({"error": "cloud not found"})
                continue

            A = np.array(it.get("R") or [1, 0, 0, 0, 1, 0, 0, 0, 1], float).reshape(3, 3)
            U, _, Vt = np.linalg.svd(A)
            A = U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))]) @ Vt

            rotated_cloud = cloud @ A.T

            # Predict geometric category robustly
            predicted_cls, predicted_conf, predicted_by = None, None, None
            target_cls = item_cls
            if auto or not item_cls or item_cls in ("auto", "all") \
                    or item_cls not in g.RULES:   # only when asked, or when
                # the label names no rule.  This used to fire whenever the
                # class had no REFERENCE, which with 21 classes and refs for
                # three meant the class picked in the dropdown was discarded
                # for eighteen of them: a door you selected was canonicalised
                # as whatever Uni3D guessed.
                predicted_cls, predicted_conf, predicted_by = predict_category(rotated_cloud)
                target_cls = predicted_cls

            try:
                R, info = STATE["canon"](rotated_cloud, target_cls)
            except Exception as exc:
                out.append({"error": f"{type(exc).__name__}: {exc}"})
                continue

            out.append({
                "M": np.round(R @ A, 6).ravel().tolist(),
                "symmetry": info.get("symmetry", "I"),
                "cluster": info.get("cluster", None),
                "predicted_cls": predicted_cls,
                "predicted_conf": None if predicted_conf is None else round(predicted_conf, 3),
                "predicted_by": predicted_by,
                "snapped": bool(info.get("snapped", False)),
                "refined_deg": round(float(info.get("refined_deg", 0.0)), 2),
            })
        ms = 1000.0 * (time.time() - t0)
        return self._send({"items": out, "ms": round(ms, 1),
                           "per": round(ms / max(len(out), 1), 1)})


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>geo_canon demo</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
 :root{--ink:#e8ecff;--mut:#9aa4d4;--faint:#6f7ab0;
       --line:rgba(150,170,255,.15);--line-2:rgba(150,170,255,.28);
       --violet:#8b7bff;--cyan:#6fe3ff;--rose:#ff8ad0;
       --glass:rgba(14,16,40,.52)}
 *{box-sizing:border-box}
 body{margin:0;background:#04050e;color:var(--ink);overflow-x:hidden;min-height:100vh;
      font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
      -webkit-font-smoothing:antialiased}

 /* --- the heavens: nebula wash, then two parallax star layers ------------ */
 #neb,#sky{position:fixed;pointer-events:none;z-index:0}
 #neb{inset:0;background:
   radial-gradient(62vw 46vw at 12% -10%, rgba(104,71,255,.20), transparent 62%),
   radial-gradient(56vw 42vw at 88% 4%,  rgba(233,74,182,.14), transparent 62%),
   radial-gradient(74vw 58vw at 50% 114%,rgba(38,128,235,.15), transparent 60%),
   radial-gradient(40vw 30vw at 70% 55%, rgba(120,255,235,.06), transparent 65%)}
 #sky{inset:-80px;opacity:.95;animation:drift 300s linear infinite}
 @keyframes drift{from{background-position:0 0,0 0}
                  to{background-position:600px 1200px,-900px 900px}}
 header,.wrap{position:relative;z-index:1}

 header{position:sticky;top:0;z-index:6;padding:13px 18px;
        background:linear-gradient(180deg,rgba(6,8,24,.86),rgba(6,8,24,.45));
        backdrop-filter:blur(14px) saturate(1.3);
        border-bottom:1px solid var(--line);
        display:flex;align-items:center;gap:11px;flex-wrap:wrap}
 .brand{margin-right:auto;min-width:0}
 .brand h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.04em;
           background:linear-gradient(92deg,#cfd6ff,var(--cyan) 38%,var(--rose) 72%,#ffd9a8);
           -webkit-background-clip:text;background-clip:text;color:transparent}
 .brand p{margin:1px 0 0;font-size:11.5px;color:var(--faint);letter-spacing:.02em}
 .ctl{display:flex;align-items:center;gap:7px;flex-wrap:wrap}
 .field{display:flex;align-items:center;gap:7px;padding:3px 5px 3px 10px;
        background:var(--glass);border:1px solid var(--line);border-radius:10px;
        backdrop-filter:blur(8px)}
 .field span{font-size:10px;color:var(--faint);text-transform:uppercase;letter-spacing:.11em}
 select,button{font:inherit;color:var(--ink);outline:0}
 select{border:0;background:transparent;padding:3px 4px;border-radius:7px;cursor:pointer;
        max-width:11.5em}
 select option{background:#0b0d22;color:var(--ink)}
 button{background:var(--glass);border:1px solid var(--line);border-radius:10px;
        padding:6px 12px;cursor:pointer;backdrop-filter:blur(8px);
        transition:border-color .2s,box-shadow .2s,color .2s,transform .1s}
 button:hover{border-color:var(--line-2);color:#fff;
              box-shadow:0 0 22px -6px rgba(139,123,255,.75),
                         inset 0 0 18px -10px rgba(139,123,255,.9)}
 button:active{transform:translateY(1px)}
 #status{display:inline-flex;align-items:center;gap:7px;font-size:12px;color:var(--mut);
         opacity:0;transition:opacity .2s;letter-spacing:.03em}
 #status.on{opacity:1}
 #status i{width:7px;height:7px;border-radius:50%;background:var(--cyan);
           box-shadow:0 0 12px 2px rgba(111,227,255,.85);
           animation:pulse 1.15s ease-in-out infinite}
 @keyframes pulse{0%,100%{opacity:.3;transform:scale(.7)}50%{opacity:1;transform:scale(1.25)}}

 .wrap{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px}
 @media(max-width:880px){.wrap{grid-template-columns:1fr}}
 .panel{display:flex;flex-direction:column;min-width:0;border-radius:16px;overflow:hidden;
        background:var(--glass);border:1px solid var(--line);
        backdrop-filter:blur(16px) saturate(1.25);
        box-shadow:inset 0 1px 0 rgba(255,255,255,.06),
                   0 24px 70px -28px rgba(0,0,0,.95),
                   0 0 80px -40px rgba(120,100,255,.55)}
 .phead{display:flex;align-items:center;gap:9px;padding:11px 15px;
        border-bottom:1px solid var(--line)}
 .phead h2{font-size:12px;margin:0;font-weight:600;letter-spacing:.14em;text-transform:uppercase}
 .phead em{font-style:normal;font-size:11.5px;color:var(--faint);
           white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 .dot{flex:none;width:7px;height:7px;border-radius:50%;background:var(--faint)}
 .dot.live{background:var(--cyan);box-shadow:0 0 12px 2px rgba(111,227,255,.8)}
 .stage{position:relative;height:clamp(320px,50vh,520px);
        background:radial-gradient(72% 72% at 50% 44%,
                   rgba(46,50,116,.34),rgba(5,7,20,.62))}
 .stage canvas{display:block;width:100%;height:100%;background:transparent}
 .cells{position:absolute;inset:0;display:grid;pointer-events:none}
 .cell{margin:4px;border:1px solid transparent;border-radius:12px;position:relative;
       transition:background .2s,border-color .2s,box-shadow .2s}
 .cell.hot{border-color:var(--line-2);background:rgba(139,123,255,.07);
           box-shadow:inset 0 0 40px -18px rgba(139,123,255,.9)}
 .tags{position:absolute;left:9px;top:8px;display:flex;gap:5px;flex-wrap:wrap;
       max-width:calc(100% - 18px)}
 .tag{font:10.5px/1.7 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
      color:var(--mut);background:rgba(8,10,28,.62);border:1px solid var(--line);
      border-radius:999px;padding:0 8px;letter-spacing:.03em;white-space:nowrap;
      backdrop-filter:blur(6px)}
 .tag.sym{color:#dfe6ff;border-color:var(--line-2);
          background:linear-gradient(100deg,rgba(139,123,255,.26),rgba(111,227,255,.18));
          box-shadow:0 0 20px -8px rgba(139,123,255,.9)}
 .bar{display:flex;align-items:center;gap:9px;height:42px;padding:0 15px;
      border-top:1px solid var(--line);color:var(--mut);font-size:12px;
      white-space:nowrap;overflow:hidden;letter-spacing:.02em}
 .sep{color:rgba(150,170,255,.3)}
 .k{color:#fff;font-weight:600;font-variant-numeric:tabular-nums;
    text-shadow:0 0 14px rgba(139,123,255,.6)}
 #cA{cursor:grab;touch-action:none} #cA.dragging{cursor:grabbing}

</style></head><body>
<div id="neb"></div><div id="sky"></div>
<header>
  <div class="brand"><h1>geometric canonicalisation</h1>
    <p>turn the input any way you like &mdash; the canonical frame stays put</p></div>
  <div class="ctl">
    <div class="field"><span>class</span><select id="cls"></select></div>
    <div class="field"><span>show</span><select id="count"></select></div>
    <button id="pick">new objects</button>
    <button id="rand">random poses</button>
    <button id="reset">reset</button>
    <button id="addcloud">add cloud</button>
    <input id="cloudfile" type="file" accept=".pt,.npy,.npz,.obj,.ply" multiple hidden>
  </div>
  <span id="status"><i></i>canonicalising…</span>
</header>
<div class="wrap">
  <section class="panel">
    <div class="phead"><span class="dot"></span><h2>input</h2>
      <em>drag inside a cell to turn that object</em></div>
    <div class="stage"><canvas id="cA"></canvas><div class="cells" id="gA"></div></div>
    <div class="bar" id="barA"></div></section>
  <section class="panel">
    <div class="phead"><span class="dot live"></span><h2>canonical</h2>
      <em>pipeline output</em></div>
    <div class="stage"><canvas id="cB"></canvas><div class="cells" id="gB"></div></div>
    <div class="bar" id="barB"></div></section>
</div>
<script>
// --- starfield: drawn once, tiled, two layers at different scales ---------
(function(){
  function layer(size,count,maxR,alpha){
    const c=document.createElement('canvas'); c.width=c.height=size;
    const x=c.getContext('2d');
    for(let i=0;i<count;i++){
      const px=Math.random()*size, py=Math.random()*size;
      const r=(0.3+Math.random()*maxR)*3;
      const a=alpha*(0.25+Math.random()*0.75);
      const g=x.createRadialGradient(px,py,0,px,py,r);
      g.addColorStop(0,`rgba(255,255,255,${a})`);
      g.addColorStop(.32,`rgba(205,218,255,${a*0.45})`);
      g.addColorStop(1,'rgba(150,180,255,0)');
      x.fillStyle=g; x.beginPath(); x.arc(px,py,r,0,Math.PI*2); x.fill();
    }
    return c.toDataURL();
  }
  const sky=document.getElementById('sky');
  sky.style.backgroundImage=`url(${layer(600,230,1.1,.95)}),url(${layer(900,170,.75,.55)})`;
  sky.style.backgroundSize='600px 600px,900px 900px';
})();

const S={cls:"all",n:4,items:[],busy:false,pending:false};
const TIP=new THREE.Matrix4().makeRotationX(-Math.PI/2);

// Chosen so cells stay near square in a panel roughly 1.5x wider than tall:
// 10 objects go 4x3 rather than 5x2, which would make every cell a tall slot.
const TILES={1:[1,1],2:[2,1],3:[3,1],4:[2,2],5:[3,2],6:[3,2],
             7:[4,2],8:[4,2],9:[3,3],10:[4,3]};
const MAXN=10;
function grid(n){ const [cols,rows]=TILES[Math.min(Math.max(n,1),MAXN)]; return {cols,rows}; }

function makeRenderer(canvas){
  // alpha:true so the nebula and stars behind the panel show through the cells
  const r=new THREE.WebGLRenderer({canvas,antialias:true,alpha:true});
  r.setPixelRatio(Math.min(devicePixelRatio,2));
  r.setClearColor(0x000000,0); r.setScissorTest(true);
  return r;
}
const rA=makeRenderer(document.getElementById("cA"));
const rB=makeRenderer(document.getElementById("cB"));
const cam=new THREE.PerspectiveCamera(38,1,0.05,50);
const VIEW_R=1.5;
const EYE=new THREE.Vector3(0.62,0.5,0.95).normalize();

function fitCamera(aspect){
  const f=0.5*cam.fov*Math.PI/180;
  const d=Math.max(VIEW_R/Math.sin(f),VIEW_R/Math.sin(Math.atan(Math.tan(f)*aspect)));
  cam.aspect=aspect;
  cam.position.copy(EYE).multiplyScalar(d);
  cam.lookAt(0,-0.08,0);
  cam.near=Math.max(0.05,d-3); cam.far=d+4;
  cam.updateProjectionMatrix();
}

// A tight white core with a short halo.  The core is what keeps the points
// reading as sharp stars rather than blur; the halo is what makes them glow
// once additive blending stacks them up.
// Two sprites, because one cannot be both bold and glowing.  CORE is a hard
// disc that keeps every point an opaque dot with a crisp edge; HALO is the soft
// bloom stacked behind it additively for the starlight.  Drawing only the halo
// is what made the cloud read as haze.
function disc(soft){
  const c=document.createElement('canvas'); c.width=c.height=64;
  const x=c.getContext('2d'), g=x.createRadialGradient(32,32,0,32,32,32);
  if(soft){
    g.addColorStop(0,'rgba(255,255,255,.80)');
    g.addColorStop(.26,'rgba(255,255,255,.38)');
    g.addColorStop(.58,'rgba(255,255,255,.11)');
    g.addColorStop(1,'rgba(255,255,255,0)');
  }else{
    g.addColorStop(0,'rgba(255,255,255,1)');
    g.addColorStop(.72,'rgba(255,255,255,1)');   // flat to here: a solid disc
    g.addColorStop(.86,'rgba(255,255,255,.5)');  // one short antialiased edge
    g.addColorStop(.95,'rgba(255,255,255,0)');
  }
  x.fillStyle=g; x.beginPath(); x.arc(32,32,32,0,Math.PI*2); x.fill();
  const t=new THREE.CanvasTexture(c);
  // no mipmaps: at a few pixels across, mip selection is itself a blur
  t.minFilter=THREE.LinearFilter; t.magFilter=THREE.LinearFilter;
  t.generateMipmaps=false;
  return t;
}
const CORE=disc(false), HALO=disc(true);

// Height ramp through a nebula: deep blue at the foot, violet and magenta
// through the body, warm starlight at the crown.  The foot is kept bright
// enough not to sink into the background.
const RAMP=[[0.34,0.45,0.88],[0.52,0.40,0.90],[0.78,0.40,0.80],
            [0.91,0.53,0.58],[0.92,0.73,0.50],[0.93,0.84,0.64]];
function rampAt(t,out,o){
  t=Math.min(Math.max(t,0),1)*(RAMP.length-1);
  const i=Math.min(Math.floor(t),RAMP.length-2), f=t-i;
  for(let k=0;k<3;k++) out[o+k]=RAMP[i][k]+f*(RAMP[i+1][k]-RAMP[i][k]);
}

function points(flat){
  const n=flat.length/3,pos=new Float32Array(flat),col=new Float32Array(n*3);
  let lo=1e9,hi=-1e9;
  for(let i=0;i<n;i++){const z=pos[i*3+2]; if(z<lo)lo=z; if(z>hi)hi=z;}
  for(let i=0;i<n;i++) rampAt((pos[i*3+2]-lo)/Math.max(hi-lo,1e-6),col,i*3);
  const gm=new THREE.BufferGeometry();
  gm.setAttribute('position',new THREE.BufferAttribute(pos,3));
  gm.setAttribute('color',new THREE.BufferAttribute(col,3));
  // one geometry, two passes: bloom behind, solid dot in front
  const halo=new THREE.Points(gm,new THREE.PointsMaterial(
    {size:0.115,vertexColors:true,sizeAttenuation:true,map:HALO,
     transparent:true,opacity:.20,depthWrite:false,
     blending:THREE.AdditiveBlending}));
  halo.renderOrder=0;
  const core=new THREE.Points(gm,new THREE.PointsMaterial(
    {size:0.075,vertexColors:true,sizeAttenuation:true,map:CORE,
     transparent:true,alphaTest:.45,depthWrite:true}));
  core.renderOrder=1;
  const grp=new THREE.Group(); grp.add(halo,core);
  return grp;
}
function axes(len){
  const p=[],c=[],cols=[[1,.45,.62],[.52,.95,.72],[.45,.72,1]];
  [[1,0,0],[0,1,0],[0,0,1]].forEach((d,i)=>{
    p.push(0,0,0,d[0]*len,d[1]*len,d[2]*len); c.push(...cols[i],...cols[i]);});
  const gm=new THREE.BufferGeometry();
  gm.setAttribute('position',new THREE.Float32BufferAttribute(p,3));
  gm.setAttribute('color',new THREE.Float32BufferAttribute(c,3));
  return new THREE.LineSegments(gm,new THREE.LineBasicMaterial(
    {vertexColors:true,transparent:true,opacity:.85,
     blending:THREE.AdditiveBlending,depthWrite:false}));
}
function scene(flat,withAxes){
  const sc=new THREE.Scene();
  const gh=new THREE.GridHelper(2.8,8,0x3b4490,0x232a5e); gh.position.y=-1.18;
  gh.material.transparent=true; gh.material.opacity=.42; gh.material.depthWrite=false;
  sc.add(gh);
  if(withAxes) sc.add(axes(0.85));
  const grp=new THREE.Group(); grp.matrixAutoUpdate=false; grp.add(points(flat));
  sc.add(grp); return {sc,grp};
}
function m4(m9){ return new THREE.Matrix4().set(
  m9[0],m9[1],m9[2],0, m9[3],m9[4],m9[5],0, m9[6],m9[7],m9[8],0, 0,0,0,1); }
function setRot(grp,rot){ grp.matrix.copy(TIP).multiply(rot); }

const PAD=4;
function cellRect(canvas,i,n){
  const {cols,rows}=grid(n);
  const w=canvas.clientWidth/cols, h=canvas.clientHeight/rows;
  const col=i%cols, row=Math.floor(i/cols);
  return {x:col*w, w, h,
          y:canvas.clientHeight-(row*h+h)};
}
function drawPanel(rend,canvas,which){
  const w=canvas.clientWidth,h=canvas.clientHeight;
  if(canvas.width!==Math.floor(w*rend.getPixelRatio())) rend.setSize(w,h,false);
  rend.setScissor(0,0,w,h); rend.setViewport(0,0,w,h); rend.clear();
  const n=S.items.length;
  S.items.forEach((it,i)=>{
    const r=cellRect(canvas,i,n);
    const vw=Math.max(r.w-2*PAD,1), vh=Math.max(r.h-2*PAD,1);
    rend.setViewport(r.x+PAD,r.y+PAD,vw,vh);
    rend.setScissor(r.x+PAD,r.y+PAD,vw,vh);
    fitCamera(vw/vh);
    rend.render(which==='A'?it.sA:it.sB,cam);
  });
}
function render(){ drawPanel(rA,document.getElementById("cA"),'A');
                   drawPanel(rB,document.getElementById("cB"),'B'); }

function chip(text,cls,full){
  return text?`<b class="tag${cls}" title="${full||text}">${text}</b>`:"";
}
function shortId(s){ s=String(s); return s.length>10?s.slice(0,8)+"…":s; }
function overlay(){
  const n=S.items.length,{cols,rows}=grid(n);
  [["gA",i=>{
      const it=S.items[i];
      const desc=(S.cls==="all"||S.cls==="auto")?`${it.cls} · ${shortId(it.id)}`:shortId(it.id);
      return chip(desc,"",`${it.cls} / ${it.id}`);
    }],
   ["gB",i=>chip(S.items[i].tag," sym")]].forEach(([id,lab])=>{
    const el=document.getElementById(id);
    el.style.gridTemplateColumns=`repeat(${cols},1fr)`;
    el.style.gridTemplateRows=`repeat(${rows},1fr)`;
    el.innerHTML="";
    for(let i=0;i<n;i++){ const d=document.createElement("div");
      d.className="cell"; d.innerHTML=`<div class="tags">${lab(i)}</div>`;
      el.appendChild(d); }
  });
}
function hot(i){
  const cs=document.getElementById("gA").children;
  for(let k=0;k<cs.length;k++) cs[k].classList.toggle("hot",k===i);
}

function randRot(){
  const u1=Math.random(),u2=Math.random(),u3=Math.random();
  const q=new THREE.Quaternion(
    Math.sqrt(1-u1)*Math.sin(2*Math.PI*u2),Math.sqrt(1-u1)*Math.cos(2*Math.PI*u2),
    Math.sqrt(u1)*Math.sin(2*Math.PI*u3),  Math.sqrt(u1)*Math.cos(2*Math.PI*u3));
  return new THREE.Matrix4().makeRotationFromQuaternion(q);
}

let INDEX={}, PREDICTABLE=[];
function shuffled(a){
  a=a.slice();
  for(let i=a.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [a[i],a[j]]=[a[j],a[i]]; }
  return a;
}
async function loadObjects(){
  S.items=[];
  const classes=Object.keys(INDEX);
  if(!classes.length) return;

  const targetClasses=[];
  if(S.cls==="all" || S.cls==="auto"){
    // a random draw of distinct classes, reshuffled once the pool runs out, so
    // "show 10" is not always the first ten alphabetically.  auto only draws
    // from classes Uni3D can name, or it would be set up to fail.
    let pool=classes;
    if(S.cls==="auto" && PREDICTABLE.length) pool=classes.filter(c=>PREDICTABLE.includes(c));
    let bag=[];
    for(let i=0;i<S.n;i++){ if(!bag.length) bag=shuffled(pool); targetClasses.push(bag.pop()); }
  }else{
    for(let i=0;i<S.n;i++) targetClasses.push(S.cls);
  }

  const usedPerClass={};
  classes.forEach(c=>usedPerClass[c]=new Set());

  for(const c of targetClasses){
    const ids=INDEX[c]||[];
    if(!ids.length) continue;
    let idx=Math.floor(Math.random()*ids.length);
    if(usedPerClass[c].size<ids.length){
      while(usedPerClass[c].has(idx)) idx=Math.floor(Math.random()*ids.length);
      usedPerClass[c].add(idx);
    }
    const d=await (await fetch(`/api/cloud?cls=${c}&idx=${idx}`)).json();
    const a=scene(d.points,false), b=scene(d.points,true);
    const it={cls:c,idx,id:d.id,W:randRot(),sA:a.sc,gA:a.grp,sB:b.sc,gB:b.grp,M:null,tag:""};
    setRot(it.gA,it.W); setRot(it.gB,new THREE.Matrix4());
    S.items.push(it);
  }

  overlay();
  let clsLabel=S.cls;
  if(S.cls==="all") clsLabel="mixed classes";
  if(S.cls==="auto") clsLabel="auto-predicted categories";
  document.getElementById("barA").innerHTML=
    `<span class="k">${S.items.length}</span> object(s) of <span class="k">${clsLabel}</span>`+
    `<span class="sep">·</span>drag a cell to turn that one`;
  render(); go();
}

function busy(on){ document.getElementById("status").classList.toggle("on",on); }

async function go(){
  if(S.busy){S.pending=true;return;}
  S.busy=true; busy(true);
  try{
    const body={
      items:S.items.map(it=>{
        const e=it.W.elements;
        return {
          cls:it.cls,                     // addresses the cloud
          auto:(S.cls==="auto"),          // ...but keep the name from the pipeline
          idx:it.idx,
          R:[e[0],e[4],e[8],e[1],e[5],e[9],e[2],e[6],e[10]]
        };
      })
    };
    const d=await (await fetch("/api/canon",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})).json();
    if(d.error){document.getElementById("barB").textContent=d.error;}
    else{
      d.items.forEach((r,i)=>{
        if(r.M){
          S.items[i].M=r.M;
          setRot(S.items[i].gB,m4(r.M));
          // the output chip carries the category Uni3D named, which is what
          // selected the reference -- the symmetry group and cluster id are
          // internals of that choice and stay off the screen
          S.items[i].pred=r.predicted_cls||null;
          S.items[i].predConf=r.predicted_conf;
          // with a class picked by hand there is nothing predicted, so no chip
          S.items[i].tag=r.predicted_cls||"";
        }
      });
      overlay(); render();
      document.getElementById("barB").innerHTML=
        `<span class="k">${d.per} ms</span> per cloud<span class="sep">·</span>`+
        `<span class="k">${d.ms} ms</span> total`+
        (S.cls==="auto"?`<span class="sep">·</span>chips show the predicted category`:``);
    }
  }catch(e){document.getElementById("barB").textContent=String(e);}
  busy(false); S.busy=false;
  if(S.pending){S.pending=false;go();}
}

const cA=document.getElementById("cA");
let drag=-1,px=0,py=0;
function cellAt(e){
  const b=cA.getBoundingClientRect(), n=S.items.length, {cols,rows}=grid(n);
  const col=Math.floor((e.clientX-b.left)/(b.width/cols));
  const row=Math.floor((e.clientY-b.top)/(b.height/rows));
  const i=row*cols+col;
  return (col>=0&&col<cols&&row>=0&&row<rows&&i<n)?i:-1;
}
function stopDrag(){ if(drag>=0){drag=-1;cA.classList.remove("dragging");go();} }
cA.addEventListener('pointerdown',e=>{ drag=cellAt(e); px=e.clientX; py=e.clientY;
  if(drag>=0){cA.setPointerCapture(e.pointerId);cA.classList.add("dragging");} });
cA.addEventListener('pointerup',stopDrag);
cA.addEventListener('pointerleave',()=>{ stopDrag(); hot(-1); });
cA.addEventListener('pointermove',e=>{ if(drag<0){ hot(cellAt(e)); return; }
  const dx=(e.clientX-px)*0.012, dy=(e.clientY-py)*0.012; px=e.clientX; py=e.clientY;
  const it=S.items[drag];
  it.W.premultiply(new THREE.Matrix4().makeRotationY(dx)
    .multiply(new THREE.Matrix4().makeRotationX(dy)));
  setRot(it.gA,it.W); render(); });

// --- upload a point cloud or a mesh --------------------------------------
// The class comes from the dropdown when one is chosen and from predict_category
// when it is on 'auto', so an upload never has to know what it is.  The server
// sends back what it decided and how sure it was; a shaky name is worth showing
// rather than hiding, because the name picks the rule -- see _upload.
const addBtn=document.getElementById("addcloud");
const addFile=document.getElementById("cloudfile");
addBtn.onclick=()=>addFile.click();
addFile.onchange=()=>{
  const fs=[...addFile.files];
  addFile.value="";
  uploadMany(fs);
};
// several files at once, from the picker or dropped anywhere on the page; they
// go up one by one and the canonicaliser runs once at the end
const OK_EXT=/\.(pt|npy|npz|obj|ply)$/i;
addEventListener('dragover',e=>e.preventDefault());
addEventListener('drop',e=>{
  e.preventDefault();
  uploadMany([...(e.dataTransfer?.files||[])].filter(f=>OK_EXT.test(f.name)));
});
async function uploadMany(fs){
  if(!fs.length) return;
  if(fs.length===1){ await doUpload(fs[0]); return; }
  const bar=document.getElementById("barB");
  let ok=0;
  for(let i=0;i<fs.length;i++){
    bar.textContent=`reading ${fs[i].name} (${i+1} of ${fs.length})`;
    if(await doUpload(fs[i],false)) ok++;
  }
  overlay(); render(); go();
  bar.textContent=`added ${ok} of ${fs.length} file(s)`+
    (fs.length>MAXN?` - only the newest ${MAXN} are shown`:``)+` - drag one to check the frame holds`;
}
async function doUpload(f,finish=true){
  const bar=document.getElementById("barB");
  busy(true); if(finish) bar.textContent=`reading ${f.name}`;
  try{
    const r=await fetch(`/api/upload?cls=${encodeURIComponent(S.cls)}`+
                        `&name=${encodeURIComponent(f.name)}`,
                        {method:"POST",body:f});
    const d=await r.json();
    if(d.error){ bar.textContent=`upload failed (${f.name}): `+d.error; busy(false); return false; }
    INDEX=await (await fetch("/api/index")).json();
    const c=await (await fetch(`/api/cloud?cls=${d.cls}&idx=${d.idx}`)).json();
    const a=scene(c.points,false), b=scene(c.points,true);
    const it={cls:d.cls,idx:d.idx,id:d.id,W:new THREE.Matrix4(),
              sA:a.sc,gA:a.grp,sB:b.sc,gB:b.grp,M:null,tag:""};
    setRot(it.gA,it.W); setRot(it.gB,new THREE.Matrix4());
    S.items.unshift(it);                      // newest first, so it is visible
    if(S.items.length>MAXN) S.items.length=MAXN;
    if(!finish){ busy(false); return true; }
    overlay(); render(); go();
    let msg=`${d.id}: ${d.n} points, ${d.mask}`;
    if(d.predicted_by){
      msg+=` - named ${d.cls}`+(d.predicted_conf==null?``:` (${d.predicted_conf})`)+
           ` by ${d.predicted_by}`;
      // The name picks the rule, so a shaky one is worth saying out loud.
      if(d.predicted_conf!=null&&d.predicted_conf<0.9)
        msg+=` - low confidence; set the class if it looks wrong`;
    }
    bar.textContent=msg+` - drag it to check the frame holds`;
  }catch(e){ bar.textContent=`upload failed (${f.name}): `+e; busy(false); return false; }
  busy(false);
  return true;
}

document.getElementById("pick").onclick=loadObjects;
document.getElementById("rand").onclick=()=>{ S.items.forEach(it=>{
  it.W=randRot(); setRot(it.gA,it.W); }); render(); go(); };
document.getElementById("reset").onclick=()=>{ S.items.forEach(it=>{
  it.W=new THREE.Matrix4(); setRot(it.gA,it.W); }); render(); go(); };

(async function(){
  INDEX=await (await fetch("/api/index")).json();
  PREDICTABLE=(await (await fetch("/api/predictable")).json()).classes||[];
  const cs=document.getElementById("cls");

  const autoOpt=document.createElement("option");
  autoOpt.value="auto";
  autoOpt.textContent="auto (predict)";
  cs.appendChild(autoOpt);

  const mixedOpt=document.createElement("option");
  mixedOpt.value="all";
  mixedOpt.textContent="all (mixed)";
  cs.appendChild(mixedOpt);

  Object.keys(INDEX).forEach(c=>{
    const o=document.createElement("option");
    o.value=c;o.textContent=c;cs.appendChild(o);
  });

  const cnt=document.getElementById("count");
  for(let i=1;i<=MAXN;i++){
    const o=document.createElement("option");
    // canonicalising and naming are both per cloud, so the wait scales with
    // this number -- worth saying rather than letting a drag seem to hang
    o.value=i;o.textContent=i>6?`${i} (slower)`:`${i}`;cnt.appendChild(o);
  }

  S.cls="auto"; cs.value="auto"; cnt.value=4; S.n=4;
  cs.onchange=()=>{S.cls=cs.value;loadObjects();};
  cnt.onchange=()=>{S.n=+cnt.value;loadObjects();};
  addEventListener('resize',()=>{overlay();render();});
  if(window.ResizeObserver){ const ro=new ResizeObserver(()=>render());
    document.querySelectorAll('.stage').forEach(s=>ro.observe(s)); }
  await loadObjects();
})();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="processed_data")
    ap.add_argument("--rules", type=Path, default=g.DEFAULT_RULES_PATH,
                    help="JSON rule database")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--instances", type=int, default=12)
    ap.add_argument("--points", type=int, default=1024)
    ap.add_argument("--mesh-points", type=int, default=g.MESH_POINTS,
                    help="points sampled off each .obj mesh, loaded or uploaded")
    ap.add_argument("--ref-instances", type=int, default=10)
    ap.add_argument("--refs", default=None, help=".npz from geo_canon.py --save-reference")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--clouds", default=None,
                    help="directory of extra .npy clouds, and where uploads are "
                         "kept; each joins the class its filename starts with "
                         "and is listed as added:<name>")
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--classify", action="store_true",
                    help="name categories with Uni3D-S instead of the PCA-overlay "
                         "fallback; the predicted name is what selects the reference")
    ap.add_argument("--centroids", default="uni3d_centroids.npz",
                    help="class centroids from uni3d_probe.py")
    ap.add_argument("--uni3d-ckpt", default=None,
                    help="uni3d-s model.pt; downloaded from the hub if omitted")
    args = ap.parse_args()
    g.configure_rules(args.rules)

    prepare(args)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\nopen http://localhost:{args.port}   (ctrl-c to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
