"""Part-layout evidence from zero-shot part labels (PatchAlign3D).

The rules decide a frame from geometry alone.  When they fail they fail by a
whole axis (upside down, turned round, lying on a side), and the object's parts
say which way is right: chair legs sit under the seat, the back stands behind
it, an aircraft's tail trails its wings.  `part_check` labels the points of the
rule-canonical cloud with the class's part names, then scores the 24
axis-aligned re-orientations of the rule frame by how well the part centroids
obey a handful of written relations, and moves only when one of them beats the
rule's own frame by a clear margin.

Nothing here is learned from our data.  PatchAlign3D (CVPR 2026, MIT licence)
was trained on Objaverse; its text side is OpenCLIP ViT-bigG-14, cached once as
`part_text_bank.npz` for the part names used in the rules file.  The input it
sees is the rule-canonical cloud, which is pose-invariant, so every input pose
of one object gets the same labels and the same decision.  The symmetry group
the rules reported is passed through untouched.
"""
import hashlib
import importlib.util
import itertools
import math
import os
import sys
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "vendor" / "PatchAlign3D" / "src"
_CKPT = _ROOT / "vendor" / "patchalign3d_ckpt" / "patchalign3d.pt"
_BANK = _ROOT / "part_text_bank.npz"
_TAU = 0.07                      # CLIP temperature used by PatchAlign3D
_STATE = {}
_CACHE = OrderedDict()
_CACHE_MAX = 4096


# ---------------------------------------------------------------------------
# CPU stand-ins for the two CUDA extensions the encoder imports
# ---------------------------------------------------------------------------

def _install_shims():
    import torch

    if "pointnet2_ops" not in sys.modules:
        def furthest_point_sample(xyz, npoint):
            # seeded with point 0, like the CUDA kernel
            xyz = xyz[..., :3].contiguous()
            B, N, _ = xyz.shape
            idx = torch.zeros(B, npoint, dtype=torch.long)
            dist = torch.full((B, N), float("inf"), dtype=xyz.dtype)
            far = torch.zeros(B, dtype=torch.long)
            rows = torch.arange(B)
            for i in range(npoint):
                idx[:, i] = far
                d = ((xyz - xyz[rows, far].unsqueeze(1)) ** 2).sum(-1)
                dist = torch.minimum(dist, d)
                far = dist.argmax(-1)
            return idx.int()

        def gather_operation(features, idx):
            idx = idx.long()
            B, C, _ = features.shape
            return torch.gather(features, 2, idx.unsqueeze(1).expand(B, C, idx.shape[1])).contiguous()

        utils = types.ModuleType("pointnet2_ops.pointnet2_utils")
        utils.furthest_point_sample = furthest_point_sample
        utils.gather_operation = gather_operation
        pkg = types.ModuleType("pointnet2_ops")
        pkg.pointnet2_utils = utils
        sys.modules["pointnet2_ops"] = pkg
        sys.modules["pointnet2_ops.pointnet2_utils"] = utils

    if "knn_cuda" not in sys.modules:
        class KNN:
            def __init__(self, k, transpose_mode=True):
                self.k = k

            def __call__(self, ref, query):
                d = torch.cdist(query, ref)
                dist, idx = d.topk(self.k, dim=-1, largest=False)
                return dist, idx

        mod = types.ModuleType("knn_cuda")
        mod.KNN = KNN
        sys.modules["knn_cuda"] = mod

    if "patchalign3d" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "patchalign3d", _SRC / "__init__.py", submodule_search_locations=[str(_SRC)])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["patchalign3d"] = pkg
        spec.loader.exec_module(pkg)


def _load():
    if "model" in _STATE:
        return _STATE
    import torch
    import torch.nn.functional as F
    _install_shims()
    from patchalign3d.models import point_transformer

    try:
        from easydict import EasyDict
    except Exception:                                   # plain namespace is enough
        EasyDict = types.SimpleNamespace
    cfg = EasyDict(trans_dim=384, depth=12, drop_path_rate=0.1, cls_dim=50, num_heads=6,
                   group_size=32, num_group=128, encoder_dims=256, color=False, num_classes=16)
    model = point_transformer.get_model(cfg).eval()
    ckpt_path = _CKPT
    if not ckpt_path.is_file():                        # fetched once, like the Uni3D weights
        from huggingface_hub import hf_hub_download
        ckpt_path = Path(hf_hub_download("patchalign3d/patchalign3d-encoder", "patchalign3d.pt",
                                         local_dir=str(_CKPT.parent)))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if [k for k in missing if not k.startswith("cls_")]:
        raise RuntimeError(f"PatchAlign3D checkpoint is missing weights: {missing[:5]}")
    proj = ckpt["proj"]
    W = proj["proj.weight"].float()
    b = proj["proj.bias"].float()
    bank = np.load(_BANK)
    _STATE.update(model=model, W=W, b=b, F=F, torch=torch,
                  bank={k: bank[k].astype(np.float32) for k in bank.files})
    threads = int(os.environ.get("PART_CUES_THREADS", "4"))
    torch.set_num_threads(max(1, threads))
    return _STATE


def part_labels(P, names, category=""):
    """Per-point part index into `names` (and its softmax probability) for the
    cloud P, which should be the rule-canonical cloud (z up).  With a category
    the prompts name it too ("the leg of a chair"), PatchAlign3D's
    part-plus-category text setting."""
    P = np.asarray(P, np.float64)
    keys = [f"{category}|{n}" if category else n for n in names]
    key = hashlib.sha1(np.round(P, 3).tobytes() + "|".join(keys).encode()).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    st = _load()
    torch, F = st["torch"], st["F"]
    missing = [k for k in keys if k not in st["bank"]]
    if missing:
        raise KeyError(f"part names not in part_text_bank.npz: {missing}")
    X = P - P.mean(0)
    X = X / max(float(np.sqrt((X ** 2).sum(1)).max()), 1e-9)       # unit sphere, as pc_normalize
    with torch.no_grad():
        pts = torch.from_numpy(X.T.copy()).float().unsqueeze(0)       # (1, 3, N)
        emb, centres, _ = st["model"].forward_patches(pts)            # (1,384,G), (1,3,G)
        feat = F.normalize(emb.transpose(1, 2) @ st["W"].T + st["b"], dim=-1)[0]    # (G, D)
        text = torch.from_numpy(np.stack([st["bank"][k] for k in keys]))
        text = F.normalize(text, dim=-1)
        logits = feat @ text.T / _TAU                                  # (G, K)
        nearest = torch.cdist(pts[0].T, centres[0].T).argmin(-1)     # (N,)
        prob = logits.softmax(-1)[nearest]                             # (N, K)
        conf, lab = prob.max(-1)
    out = (lab.numpy().astype(np.int16), conf.numpy().astype(np.float32))
    _CACHE[key] = out
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return out


def _octahedral():
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if abs(np.linalg.det(M) - 1.0) < 1e-9:
                mats.append(M)
    mats.sort(key=lambda M: -np.trace(M))          # identity first, then small turns
    return mats


_OCT = _octahedral()


def _pairs(spec):
    out = []
    for s in spec:
        a, _, b = str(s).partition(":")
        if a and b:
            out.append((a.strip(), b.strip()))
    return out


def part_check(shape, R, info, parts=("part",), below=(), above=(), behind=(), front=(),
               category="", margin=1.0, delta=0.3, best_frac=0.6, min_frac=0.03, lock_up=False):
    """Re-orient by part layout.  Relations are "a:b" pairs in the canonical
    frame of the result (x forward, z up):
        below  a:b   a sits lower than b         above  a:b   a sits higher than b
        behind a:b   a lies behind b (smaller x) front  a:b   a lies in front of b
    Each relation scores clip(separation / delta, -1, 1), so tiny centroid
    offsets from noisy labels count for little.  A candidate replaces the rule's
    frame only if it gains more than `margin` AND satisfies the relations
    decisively (score >= best_frac * number of relations).  `lock_up` keeps the
    rule's up axis."""
    trace = {"switched": False}
    try:
        P = shape.X @ R.T
        lab, conf = part_labels(P, list(parts), category)
    except Exception as exc:                          # never lose a cloud to this step
        return R, dict(info, part_check={"error": f"{type(exc).__name__}: {exc}"})

    cents = {}
    for k, name in enumerate(parts):
        m = lab == k
        if m.mean() >= min_frac:
            cents[name] = P[m].mean(0)
    rels = ([(a, b, 2, +1) for a, b in _pairs(below)] + [(a, b, 2, -1) for a, b in _pairs(above)] +
            [(a, b, 0, +1) for a, b in _pairs(behind)] + [(a, b, 0, -1) for a, b in _pairs(front)])
    usable = [r for r in rels if r[0] in cents and r[1] in cents]
    trace.update(parts={n: round(float((lab == k).mean()), 3) for k, n in enumerate(parts)},
                 relations=len(usable))
    if not usable:
        return R, dict(info, part_check=trace)

    def score(O):
        s = 0.0
        for a, b, axis, sgn in usable:
            d = (O @ cents[b])[axis] - (O @ cents[a])[axis]
            s += min(1.0, max(-1.0, sgn * d / delta))
        return s

    cands = [O for O in _OCT if not lock_up or O[2, 2] > 0.99]
    base = score(np.eye(3))
    best_s, best_O = base, np.eye(3)
    for O in cands:                                   # identity first: ties keep the rule
        s = score(O)
        if s > best_s + 1e-9:
            best_s, best_O = s, O
    gain = best_s - base
    trace.update(base=round(base, 3), best=round(best_s, 3), gain=round(gain, 3))
    if (gain > margin and best_s >= best_frac * len(usable)
            and not np.allclose(best_O, np.eye(3))):
        trace["switched"] = True
        trace["turn"] = best_O.astype(int).tolist()
        return best_O @ R, dict(info, part_check=trace)
    return R, dict(info, part_check=trace)
