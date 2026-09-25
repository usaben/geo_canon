#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class-conditional geometric canonicalisation of arbitrarily rotated point clouds.

No learning, no weights, no training data.  Given a point cloud and its class
label and a recipe loaded from rules.json the pipeline computes a canonical
rotation R in SO(3) by a deterministic sequence of geometric constructions:

    principal axes -> symmetry detection -> stored geometric recipe -> frame

and then measures

    STABILITY   : equivariance of the estimator under random input rotations
    CONSISTENCY : agreement of the canonical frame across instances of a class

Both metrics are reported raw and quotiented by the object's rotational
symmetry group, so that a bowl (C_inf about its axis) or a square table (C_4)
is not penalised for a rotation that maps the object onto itself.

Usage
-----
    python geo_canon.py                       # auto: real data if found, else synthetic
    python geo_canon.py --data processed_data # ShapeNet-style folders per synset id
    python geo_canon.py --data shapenet_subset# raw .obj/.ply geometry, sampled
    python geo_canon.py --data shapenet_subset --mesh-points 16384
    python geo_canon.py --synthetic           # force procedural shapes
    python geo_canon.py --instances 30 --rotations 8 --no-figures
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

if __package__:
    from .rule_database import Operation, RuleDatabase
else:  # direct script execution, as used by the demo and CLI
    from rule_database import Operation, RuleDatabase

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------


# Symmetry group of each class in CANONICAL coordinates (z = up, x = forward).
# "I"     : trivial, the frame is fully determined
# "C2z"   : 180 deg about canonical z is a self-map
# "C4z"   : 90 deg steps about canonical z
# "Cinfz" : any rotation about canonical z (surface of revolution)

SIGMA_MATCH = 0.05      # tolerance of the symmetry matching kernel (cloud has rms radius 1)
N_PROBE = 320           # points used to score a candidate symmetry
MIRROR_TRIM = 0.80      # fraction of probes kept when scoring a mirror plane,
                        # so that an occluded or cropped region cannot veto the
                        # true plane of symmetry
PARTIAL_MIRROR = False  # let the mirror plane leave the centroid.  Correct for
                        # heavily occluded scans, but on complete clouds the
                        # extra freedom invents planes -- an offset plane can
                        # map one wing of an aircraft onto the other -- so it is
                        # off by default and exposed as --partial-mirror
EPS = 1e-12
RULESET_VERSION = 2


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

MESH_EXT = {".obj", ".ply"}   # read by trimesh
CLOUD_EXT = {".pt", ".npy", ".npz"}
MESH_POINTS = 8192          # how densely a mesh is sampled when nobody says


def sample_mesh(path: Path, n_points: int, seed: int) -> np.ndarray:
    """n_points off a geometry file (.obj / .ply), as XYZ.

    A mesh is sampled uniformly over its surface -- the same sampling as
    obj_to_pt.py, so a cloud read straight off the mesh and one read from the
    .pt that script writes are drawn from the same law.

    A .ply carrying vertices and no faces is already a point cloud and is read
    rather than sampled.  That includes a Gaussian-splat .ply: the xyz are the
    splat centres, and everything that makes it a splat rather than a point --
    opacity, scale, rotation, spherical harmonics -- is dropped, since the
    pipeline only ever looks at positions.
    """
    import trimesh  # imported lazily so the script runs without trimesh
    kind = path.suffix.lower().lstrip(".")
    if kind == "ply":                    # may be either, so let the file say
        geom = trimesh.load(path, file_type=kind, process=False)
    else:
        geom = trimesh.load(path, file_type=kind, process=False, force="mesh")
    if isinstance(geom, trimesh.Scene):
        geom = trimesh.util.concatenate(list(geom.geometry.values()))
    if len(getattr(geom, "faces", ())) == 0:
        return subsample(np.asarray(geom.vertices, dtype=np.float64), n_points, seed)
    np.random.seed(seed)
    points, _ = trimesh.sample.sample_surface(geom, n_points)
    return np.asarray(points, dtype=np.float64)


def cloud_id(path: Path) -> str:
    """Name a cloud by its file, except a mesh, which is named by its model
    folder -- every ShapeNet mesh is called model_normalized.obj."""
    if path.suffix.lower() in MESH_EXT:
        folder = path.parent
        if folder.name == "models":       # <model_id>/models/model_normalized.obj
            folder = folder.parent
        return folder.name
    return path.stem


def load_xyz(path: Path, n_points: int = MESH_POINTS, seed: int = 0) -> np.ndarray:
    """Read one point cloud from .pt / .npy / .npz, or take one off a .obj or
    .ply geometry, keep XYZ only.  n_points and seed apply to geometry only."""
    path = Path(path)
    if path.suffix.lower() in MESH_EXT:
        arr = sample_mesh(path, n_points, seed)
    elif path.suffix == ".pt":
        import torch  # imported lazily so the script runs without torch
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            for key in ("points", "pos", "xyz", "point_cloud"):
                if key in obj:
                    obj = obj[key]
                    break
        if hasattr(obj, "detach"):
            obj = obj.detach().cpu().numpy()
        arr = np.asarray(obj)
    elif path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        z = np.load(path)
        key = next((k for k in ("points", "pos", "xyz") if k in z), z.files[0])
        arr = z[key]
    else:
        raise ValueError(f"unsupported file type: {path}")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"expected (N,>=3) array, got {arr.shape} from {path}")
    return np.ascontiguousarray(arr[:, :3])


def subsample(X: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(X) <= n:
        return X.copy()
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), n, replace=False)]


def natural_key(path):
    """Sort car_2 before car_10.

    The files are named <class>_<n>, so plain lexicographic order would take
    1, 10, 100, 1000 as the first four instances of a class and make "the first
    twelve" mean something nobody expects.
    """
    parts = re.split(r"(\d+)", Path(path).name)
    return [int(t) if t.isdigit() else t for t in parts]


def load_folder(root, n_points: int = 1024, seed: int = 0):
    """Every cloud in one flat folder, as (clouds, ids).

    load_real walks a tree of synset folders; this is for a folder that holds
    one class already, such as results/<class>_aug0, whose files are dicts
    ({points, knn_idx, variant}) rather than bare tensors.  load_xyz unwraps
    those, so they need no converting.
    """
    root = Path(root)
    clouds, ids = [], []
    for i, p in enumerate(sorted(root.glob("*"), key=natural_key)):
        if p.suffix.lower() not in CLOUD_EXT | MESH_EXT:
            continue
        try:
            clouds.append(subsample(load_xyz(p, n_points, seed + i), n_points, seed + i))
            ids.append(cloud_id(p))
        except Exception as exc:
            print(f"  [warn] skipped {p.name}: {exc}")
    return clouds, ids


def load_real(data_root: Path, n_instances: int, n_points: int, seed: int,
              mesh_points: int | None = None):
    """Return {class_name: (list_of_clouds, list_of_ids)} for whatever is present.

    n_points thins a stored cloud, which has whatever size it was written with.
    A mesh has no size of its own -- it is sampled -- so mesh_points (default
    MESH_POINTS) says how densely, and that cloud is kept at that density
    instead of being thinned to n_points.
    """
    mesh_points = int(mesh_points or MESH_POINTS)
    out = {}
    synset_of = {cls: syn for syn, cls in SYNSET_TO_CLASS.items()}
    for cls in CLASS_ORDER:
        # by synset folder if the class has one and the tree is raw ShapeNet,
        # otherwise by a folder named after the class
        folder = data_root / synset_of.get(cls, cls)
        if not folder.is_dir():
            folder = data_root / cls
        if not folder.is_dir():
            continue
        files = sorted((p for p in folder.rglob("*")
                        if p.suffix.lower() in CLOUD_EXT),
                       key=natural_key)[:n_instances]
        if not files:
            # no sampled clouds here: fall back to the geometry itself, i.e.
            # a raw ShapeNet tree of <synset>/<model_id>/[models/]*.obj
            files = sorted((p for p in folder.rglob("*")
                            if p.suffix.lower() in MESH_EXT),
                           key=natural_key)[:n_instances]
        if not files:
            continue
        clouds, ids = [], []
        for i, f in enumerate(files):
            try:
                if f.suffix.lower() in MESH_EXT:
                    clouds.append(load_xyz(f, mesh_points, seed + i))
                else:
                    clouds.append(subsample(load_xyz(f), n_points, seed + i))
                ids.append(cloud_id(f))
            except Exception as exc:                       # keep going on a bad file
                print(f"  [warn] skipped {f.name}: {exc}")
        if clouds:
            out[cls] = (clouds, ids)
    return out


# ----------------------------------------------------------------------------
# Procedural shapes (fallback / self-test).
# Generated directly in canonical pose: z = up, x = forward.  The evaluation
# rotates them randomly, so the generator never leaks its frame to the solver.
# ----------------------------------------------------------------------------

def _sample_box(rng, n, cx, cy, cz, sx, sy, sz):
    """n points on the surface of an axis-aligned box."""
    areas = np.array([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy], float)
    counts = rng.multinomial(n, areas / areas.sum())
    pts = []
    for face, k in enumerate(counts):
        if k == 0:
            continue
        u, v = rng.random(k) - 0.5, rng.random(k) - 0.5
        if face < 2:
            p = np.stack([np.full(k, 0.5 if face == 0 else -0.5), u, v], 1)
        elif face < 4:
            p = np.stack([u, np.full(k, 0.5 if face == 2 else -0.5), v], 1)
        else:
            p = np.stack([u, v, np.full(k, 0.5 if face == 4 else -0.5)], 1)
        pts.append(p * np.array([sx, sy, sz]) + np.array([cx, cy, cz]))
    return np.concatenate(pts, 0) if pts else np.zeros((0, 3))


def _sample_disc(rng, n, center, radius, axis=2, thickness=0.0):
    r = radius * np.sqrt(rng.random(n))
    a = rng.random(n) * 2 * np.pi
    p = np.zeros((n, 3))
    idx = [i for i in range(3) if i != axis]
    p[:, idx[0]] = r * np.cos(a)
    p[:, idx[1]] = r * np.sin(a)
    p[:, axis] = (rng.random(n) - 0.5) * thickness
    return p + np.asarray(center, float)


def make_airplane(rng, n):
    span = rng.uniform(0.8, 1.1)
    length = rng.uniform(0.9, 1.2)
    parts = []
    # fuselage: tapered tube along x, nose at +x
    k = int(0.30 * n)
    t = rng.random(k)
    radius = 0.055 * (1 - t ** 2.2) + 0.012
    a = rng.random(k) * 2 * np.pi
    parts.append(np.stack([length * (t - 0.35),
                           radius * np.cos(a),
                           radius * np.sin(a) * 0.9], 1))
    k = int(0.14 * n)                                    # rear fuselage
    t = rng.random(k)
    radius = 0.055 * (1 - 0.7 * t)
    a = rng.random(k) * 2 * np.pi
    parts.append(np.stack([-0.35 * length - 0.32 * length * t,
                           radius * np.cos(a), radius * np.sin(a)], 1))
    k = int(0.30 * n)                                    # main wings, swept back
    u = rng.random(k)
    side = rng.choice([-1.0, 1.0], k)
    chord = 0.26 * (1 - 0.6 * u)
    parts.append(np.stack([-0.05 * length - 0.28 * u + (rng.random(k) - 0.5) * chord,
                           side * u * span / 2,
                           np.full(k, 0.0) + (rng.random(k) - 0.5) * 0.012], 1))
    k = int(0.12 * n)                                    # horizontal stabiliser
    u = rng.random(k)
    side = rng.choice([-1.0, 1.0], k)
    parts.append(np.stack([-0.58 * length - 0.12 * u + (rng.random(k) - 0.5) * 0.10,
                           side * u * span * 0.30,
                           np.full(k, 0.02)], 1))
    k = n - sum(len(p) for p in parts)                   # vertical fin (breaks up/down)
    u = rng.random(max(k, 1))
    parts.append(np.stack([-0.58 * length - 0.14 * u + (rng.random(len(u)) - 0.5) * 0.10,
                           (rng.random(len(u)) - 0.5) * 0.012,
                           0.02 + u * 0.26], 1))
    return np.concatenate(parts, 0)


def make_car(rng, n):
    length, width = rng.uniform(1.0, 1.25), rng.uniform(0.42, 0.52)
    body_h, cabin_h = rng.uniform(0.22, 0.28), rng.uniform(0.16, 0.22)
    parts = [_sample_box(rng, int(0.42 * n), 0.0, 0.0, body_h / 2 + 0.10,
                         length, width, body_h)]
    # cabin: narrower and set back from centre -> gives the up and forward signs
    parts.append(_sample_box(rng, int(0.26 * n), -0.13 * length, 0.0,
                             body_h + 0.10 + cabin_h / 2,
                             length * 0.44, width * 0.80, cabin_h))
    k = int(0.07 * n)
    for sx in (0.30, -0.30):
        for sy in (0.5, -0.5):
            parts.append(_sample_disc(rng, k, (sx * length, sy * width * 1.02, 0.11),
                                      0.105, axis=1, thickness=0.05))
    return np.concatenate(parts, 0)


def make_chair(rng, n):
    w, d = rng.uniform(0.42, 0.55), rng.uniform(0.42, 0.55)
    seat_h, back_h = rng.uniform(0.42, 0.50), rng.uniform(0.38, 0.55)
    leg = 0.035
    parts = [_sample_box(rng, int(0.38 * n), 0.0, 0.0, seat_h, d, w, 0.05)]
    parts.append(_sample_box(rng, int(0.36 * n), -d / 2 + 0.03, 0.0,
                             seat_h + back_h / 2, 0.05, w, back_h))
    k = int(0.065 * n)
    for sx in (0.42, -0.42):
        for sy in (0.42, -0.42):
            parts.append(_sample_box(rng, k, sx * d, sy * w, seat_h / 2,
                                     leg, leg, seat_h))
    return np.concatenate(parts, 0)


def make_table(rng, n):
    """Rectangular, square or round tops, in roughly the mix ShapeNet has.  The
    shape of the top is the whole point of the symmetry stage: a rectangle is
    C2, a square C4 and a round top C_inf, and each admits a different set of
    equally valid canonical frames."""
    kind = rng.choice(["rect", "square", "round"], p=[0.5, 0.2, 0.3])
    h = rng.uniform(0.55, 0.75)
    leg = 0.05
    if kind == "round":
        radius = rng.uniform(0.35, 0.6)
        k = int(0.56 * n)
        a = rng.random(k) * 2 * np.pi
        rr = radius * np.sqrt(rng.random(k))
        z = h + 0.05 * (rng.random(k) - 0.5)
        parts = [np.stack([rr * np.cos(a), rr * np.sin(a), z], 1)]
        if rng.random() < 0.5:                        # central pedestal
            m = n - k
            az = rng.random(m) * 2 * np.pi
            pr = 0.07 * radius / 0.45
            parts.append(np.stack([pr * np.cos(az), pr * np.sin(az),
                                   h * rng.random(m)], 1))
        else:                                         # four legs on a circle
            m = (n - k) // 4
            for ang in (0.25, 0.75, 1.25, 1.75):
                parts.append(_sample_box(rng, m, 0.72 * radius * math.cos(ang * np.pi),
                                         0.72 * radius * math.sin(ang * np.pi),
                                         h / 2, leg, leg, h))
        return np.concatenate(parts, 0)

    a, b = rng.uniform(0.55, 1.1), rng.uniform(0.55, 1.1)
    d, w = max(a, b), min(a, b)       # long side along x, matching the canonical rule
    if kind == "square":
        w = d
    parts = [_sample_box(rng, int(0.56 * n), 0.0, 0.0, h, d, w, 0.05)]
    k = int(0.11 * n)
    for sx in (0.42, -0.42):
        for sy in (0.42, -0.42):
            parts.append(_sample_box(rng, k, sx * d, sy * w, h / 2, leg, leg, h))
    return np.concatenate(parts, 0)


def make_bowl(rng, n):
    radius = rng.uniform(0.40, 0.55)
    depth = radius * rng.uniform(0.55, 0.85)
    k = int(0.78 * n)                                     # inner+outer surface of revolution
    u = rng.random(k) ** 0.75
    a = rng.random(k) * 2 * np.pi
    rr = radius * np.sin(np.pi / 2 * u)
    zz = depth * (1 - np.cos(np.pi / 2 * u))
    parts = [np.stack([rr * np.cos(a), rr * np.sin(a), zz], 1)]
    k2 = int(0.12 * n)                                    # rim ring, open side is +z
    a = rng.random(k2) * 2 * np.pi
    parts.append(np.stack([radius * np.cos(a), radius * np.sin(a),
                           depth + rng.random(k2) * 0.012], 1))
    k3 = n - k - k2                                       # flat base
    parts.append(_sample_disc(rng, max(k3, 1), (0, 0, 0.004), radius * 0.30, axis=2))
    return np.concatenate(parts, 0)


SYNTHETIC = {"airplane": make_airplane, "car": make_car, "chair": make_chair,
             "table": make_table, "bowl": make_bowl}


def load_synthetic(n_instances: int, n_points: int, seed: int):
    """Procedural stand-ins, for the classes that have a generator.

    CLASS_ORDER has grown past the five shapes generated here, so it is the
    generators that decide what --synthetic can offer, not the rule table.
    """
    out = {}
    for ci, cls in enumerate(c for c in CLASS_ORDER if c in SYNTHETIC):
        clouds, ids = [], []
        for i in range(n_instances):
            rng = np.random.default_rng(seed + 1000 * ci + i)
            clouds.append(SYNTHETIC[cls](rng, n_points))
            ids.append(f"{cls}_{i:03d}")
        out[cls] = (clouds, ids)
    return out


# ----------------------------------------------------------------------------
# Geometry primitives
# ----------------------------------------------------------------------------

def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > EPS else np.array([0.0, 0.0, 1.0])


def normalise_cloud(X):
    """Remove translation and scale.  Rotation is untouched, so the frame we
    estimate afterwards is unaffected by this step."""
    X = np.asarray(X, float)
    c = X.mean(0)
    Y = X - c
    s = float(np.sqrt((Y ** 2).sum(1).mean()))
    return Y / max(s, EPS), c, s


def principal_axes(X):
    """Eigen-decomposition of the covariance.  Columns of V are axes, sorted by
    decreasing eigenvalue.  Equivariant: cov(X A^T) = A cov(X) A^T."""
    C = (X.T @ X) / len(X)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def candidate_axes(w, V, tol=0.30, n_extra=16):
    """The 3 principal axes, plus a fan inside any near-degenerate eigenplane.

    For a shape with an exact mirror or rotational symmetry the symmetry axis is
    an eigenvector of the covariance, so the principal axes are the right
    candidate set.  When two eigenvalues nearly coincide the eigenvectors inside
    that plane are numerically arbitrary, so the plane is sampled instead."""
    cands = [V[:, 0], V[:, 1], V[:, 2]]
    for i, j in ((0, 1), (1, 2), (0, 2)):
        if abs(w[i] - w[j]) <= tol * max(abs(w[i]), abs(w[j]), EPS):
            for t in np.linspace(0, np.pi, n_extra, endpoint=False)[1:]:
                cands.append(math.cos(t) * V[:, i] + math.sin(t) * V[:, j])
    return np.array([unit(c) for c in cands])


def fib_hemisphere(n):
    """n roughly equidistant directions on a hemisphere (canonical coordinates)."""
    k = np.arange(n) + 0.5
    z = k / n                                     # upper hemisphere only: d ~ -d
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * k
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


_FIB_CACHE = {}


def direction_grid(V, n):
    """The hemisphere grid rotated into the object's own principal frame, so the
    search set follows the object instead of the world axes.  A rotated copy of
    the cloud gets the rotated grid, which is what keeps the search equivariant;
    any residual error from a noisy principal frame is removed by the refit that
    follows the search."""
    if n not in _FIB_CACHE:
        _FIB_CACHE[n] = fib_hemisphere(n)
    return _FIB_CACHE[n] @ V.T


def complement(n):
    """Two unit vectors spanning the plane orthogonal to n."""
    n = unit(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = unit(a - (a @ n) * n)
    return u, np.cross(n, u)


def frame_from(forward, up):
    """Right-handed rotation with rows [x=forward, y=left, z=up].
    Canonical coordinates of a world point p are R @ p, i.e. P = X @ R.T."""
    z = unit(up)
    x = forward - (forward @ z) * z
    if np.linalg.norm(x) < 1e-8:                     # degenerate input, pick anything
        x = complement(z)[0]
    x = unit(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    if np.linalg.det(R) < 0:                         # cannot happen, kept as a guard
        R[1] *= -1
    return R


def is_rotation(R, atol=1e-7):
    return (np.allclose(R.T @ R, np.eye(3), atol=atol)
            and abs(np.linalg.det(R) - 1.0) < atol)


def random_rotations(k, rng):
    """Uniform on SO(3) via QR of a Gaussian matrix (Haar measure)."""
    out = []
    for _ in range(k):
        Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
        Q = Q * np.sign(np.diag(R))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        out.append(Q)
    return out


# ----------------------------------------------------------------------------
# Symmetry detection
# ----------------------------------------------------------------------------

class Shape:
    """A normalised cloud plus the structures every rule needs."""

    def __init__(self, X, n_probe=N_PROBE, sigma=SIGMA_MATCH):
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"expected (N,3) points, got {X.shape}")
        if len(X) < 24:
            raise ValueError(f"need at least 24 points, got {len(X)}")
        self.X, self.centre, self.scale = normalise_cloud(X)
        self.n = len(self.X)
        self.tree = cKDTree(self.X)
        step = max(1, self.n // n_probe)
        self.probe = self.X[::step][:n_probe]
        # The matching kernel must track the sampling density: a fixed width
        # either saturates on dense clouds or registers nothing on sparse ones.
        nn, _ = self.tree.query(self.X, k=2, workers=-1)
        self.nn_spacing = float(np.median(nn[:, 1]))
        self.sigma = max(sigma, 1.5 * self.nn_spacing)
        self.w, self.V = principal_axes(self.X)
        self.cands = candidate_axes(self.w, self.V)
        self._grids = {}

    def grid(self, n):
        if n not in self._grids:
            self._grids[n] = np.vstack([self.V.T, direction_grid(self.V, n)])
        return self._grids[n]

    def match_score(self, Y, trim=None):
        """Mean Gaussian agreement between transformed probes and the cloud.
        Equivariant by construction: distances are preserved by rotation.

        `trim` keeps only that fraction of the best-matched probes.  A cloud
        with a missing region has no partner for the reflections of what
        survives next to the hole, and an untrimmed mean lets such a hole veto
        the true plane; the trimmed mean scores the plane on the part of the
        object that is actually present."""
        d, _ = self.tree.query(Y, workers=-1)
        s = np.exp(-(d / self.sigma) ** 2)
        if trim is not None and trim < 1.0:
            k = max(8, int(round(trim * len(s))))
            s = np.partition(s, len(s) - k)[len(s) - k:]
        return float(s.mean())

    def mirror_score(self, normals, trim=None):
        """Reflection symmetry score for one or many plane normals."""
        trim = MIRROR_TRIM if trim is None else trim
        normals = np.atleast_2d(normals)
        P = self.probe
        out = np.empty(len(normals))
        for i, n in enumerate(normals):
            n = unit(n)
            out[i] = self.match_score(P - 2.0 * np.outer(P @ n, n), trim=trim)
        return out

    def rot_score(self, axis, angles=(np.pi / 2, 2 * np.pi / 3, 0.83)):
        """Rotational symmetry about an axis, averaged over several angles.
        Irrational-looking angles test for a full surface of revolution."""
        a = unit(axis)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        s = 0.0
        for th in angles:
            R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
            s += self.match_score(self.probe @ R.T)
        return s / len(angles)


def mirror_score_at(shape, n, offset=0.0, trim=None):
    """Reflection score for the plane {x : n.x = offset}."""
    n = unit(n)
    P = shape.probe
    Pr = P - 2.0 * np.outer(P @ n - offset, n)
    return shape.match_score(Pr, trim=MIRROR_TRIM if trim is None else trim)


def best_offset(shape, n, span=0.18, n_steps=13, trim=None):
    """Best displacement of the plane along its own normal.

    The centroid lies on the symmetry plane only when the whole object is
    present.  Crop a cloud, or occlude it, and the centroid slides off the
    plane, so a search restricted to planes through the centroid rejects the
    true symmetry.  One cheap 1-D scan along the normal restores it."""
    offs = np.linspace(-span, span, n_steps)
    sc = [mirror_score_at(shape, n, o, trim) for o in offs]
    i = int(np.argmax(sc))
    return float(offs[i]), float(sc[i])


def refine_mirror(shape, n, iters=6, trim=0.8, offset=0.0, estimate_offset=True):
    """Sharpen a mirror plane by constrained ICP.

    Reflect the cloud, take nearest-neighbour correspondences, then solve for the
    improper motion x -> M x + t (det M = -1) that best explains them.  The plane
    normal is the eigenvector of M with eigenvalue -1 and the plane offset is
    t.n/2.  Estimating t as well as M is what makes this work on partial clouds.
    The result is a continuous function of the cloud instead of a grid pick."""
    X = shape.X
    n = unit(n)
    for _ in range(iters):
        Xr = X - 2.0 * np.outer(X @ n - offset, n)
        d, j = shape.tree.query(Xr, workers=-1)
        keep = d <= max(np.quantile(d, trim), 1e-6)      # trimmed, robust to holes
        P, Q = X[keep], X[j[keep]]
        if len(P) < 12:
            break
        if estimate_offset:
            pc, qc = P.mean(axis=0), Q.mean(axis=0)
            H = (P - pc).T @ (Q - qc)
        else:
            pc = qc = np.zeros(3)
            H = P.T @ Q                                   # maximise tr(M H) over O(3)^-
        U, S, Vt = np.linalg.svd(H)
        D = np.eye(3)
        if np.linalg.det(Vt.T @ U.T) > 0:
            D[2, 2] = -1.0
        M = Vt.T @ D @ U.T
        Msym = 0.5 * (M + M.T)
        ew, ev = np.linalg.eigh(Msym)
        n_new = unit(ev[:, 0])                            # eigenvalue closest to -1
        if n_new @ n < 0:
            n_new = -n_new
        off_new = float((qc - M @ pc) @ n_new) / 2.0 if estimate_offset else 0.0
        shift = np.linalg.norm(n_new - n) + abs(off_new - offset)
        n, offset = n_new, off_new
        if shift < 1e-6:
            break
    return n, offset


def best_mirror(shape, n_grid=48, n_refine=3, partial=None):
    """Strongest reflection plane.

    Candidates are the principal axes, which are exactly the mirror normals of a
    perfectly symmetric shape, together with a coarse grid that covers the case
    where two eigenvalues are close and the eigenvectors are therefore poorly
    determined.  The best few are polished by ICP -- with a plane offset when
    `partial`, so that a cropped or occluded cloud is still matched -- and the
    winner is kept only if the polish did not make the score worse."""
    partial = PARTIAL_MIRROR if partial is None else partial
    cands = np.vstack([shape.cands, shape.grid(n_grid)])
    scores = shape.mirror_score(cands)
    order = np.argsort(scores)[::-1][:n_refine]
    best_n, best_s = cands[order[0]], float(scores[order[0]])
    for idx in order:
        n0 = cands[idx]
        o0 = best_offset(shape, n0)[0] if partial else 0.0
        n, o = refine_mirror(shape, n0, offset=o0, estimate_offset=partial)
        sc = mirror_score_at(shape, n, o)
        if sc > best_s:
            best_n, best_s = n, sc
    return unit(best_n), best_s


# ----------------------------------------------------------------------------
# Semantic geometric features along a direction
# ----------------------------------------------------------------------------

def slab_peak(X, d, width=0.07, ends_only=True, end_frac=0.38):
    """Largest fraction of points inside a thin slab perpendicular to d.

    A table top, a chair back or any flat plate produces a sharp peak when d is
    its normal.  Returns (fraction, centre, at_high_end)."""
    t = np.sort(X @ unit(d))
    lo, hi = t[0], t[-1]
    span = max(hi - lo, EPS)
    w = width * span
    j = np.searchsorted(t, t + w, side="right")
    count = j - np.arange(len(t))
    centre = t + w / 2
    if ends_only:
        ok = (centre <= lo + end_frac * span) | (centre >= hi - end_frac * span)
        if not ok.any():
            ok = np.ones(len(t), bool)
        count = np.where(ok, count, -1)
    k = int(np.argmax(count))
    frac = count[k] / len(t)
    c = float(centre[k])
    return float(frac), c, bool(c > (lo + hi) / 2)


def support_score(X, d, bottom_frac=0.30):
    """How leg-like the low end of direction d is.

    Legs and wheels are sparse in mass but wide in footprint, which is what
    separates 'down' from every other direction for chairs, tables and cars.
    Returns a score; larger means d is more plausibly 'up'."""
    d = unit(d)
    t = X @ d
    lo, hi = t.min(), t.max()
    span = max(hi - lo, EPS)
    m = t <= lo + bottom_frac * span
    f = float(m.mean())
    if f < 1e-3:
        return 0.0
    u, v = complement(d)
    P = np.stack([X @ u, X @ v], 1)
    r_all = float(np.sqrt((P ** 2).sum(1)).mean()) + EPS
    r_bot = float(np.sqrt((P[m] ** 2).sum(1)).mean())
    return (r_bot / r_all) ** 2 / (f + 0.08)


def taper_sign(X, d, frac=0.33):
    """+1 if the cloud is wider (perpendicular to d) at the low end than at the
    high end.  A car body is wide at the wheels and narrow at the roof."""
    d = unit(d)
    t = X @ d
    lo, hi = t.min(), t.max()
    span = max(hi - lo, EPS)
    u, v = complement(d)
    P = np.stack([X @ u, X @ v], 1)
    r = np.sqrt((P ** 2).sum(1))
    bot = r[t <= lo + frac * span]
    top = r[t >= hi - frac * span]
    if len(bot) < 5 or len(top) < 5:
        return 1.0
    return 1.0 if bot.mean() >= top.mean() else -1.0


def top_is_shorter(X, d, frac=0.20):
    """True when the high end of d spans less of the object's long horizontal
    axis than the low end does.

    The cue that separates a roof from an underbody and a backrest from a floor
    pan: the thing you sit on or drive on runs the whole length of the object,
    while what sits on top of it -- cabin, greenhouse, backrest -- covers only
    part of it.  Unlike support_score this asks about extent, not about mass,
    so a sparse-but-wide top cannot pass itself off as a set of legs.

    Measured over 50 ShapeNet instances per class, on the instances whose up
    AXIS the rule already gets right (so this scores the sign alone), against
    ShapeNet's own +y:

        cue                  chair    car   table
        support_score        52.9%  43.9%   93.3%
        taper_sign           67.6%  92.7%   76.7%
        top_is_shorter       82.4%  87.8%   63.3%

    So chair takes this one, car takes taper_sign, and table keeps support:
    a table is legs and a flat top with nothing above it, which is the case
    support_score was written for.
    """
    d = unit(d)
    t = X @ d
    w = X - np.outer(t, d)                        # the horizontal part
    wc = w - w.mean(0)
    long_axis = unit(np.linalg.svd(wc, full_matrices=False)[2][0])
    L = w @ long_axis
    lo, hi = np.quantile(t, frac), np.quantile(t, 1.0 - frac)
    top, bot = L[t > hi], L[t < lo]
    if len(top) < 5 or len(bot) < 5:
        return True
    return float(np.ptp(top)) <= float(np.ptp(bot))


def floor_score(X, d, frac=0.10):
    """How much the low end of d looks like a foot standing on a floor: wide
    across and thin along d.  support_score asks about mass, this asks about
    planarity, which is what separates a base from an open rim."""
    d = unit(d)
    t = X @ d
    span = max(float(np.ptp(t)), EPS)
    m = t <= t.min() + frac * span
    if m.sum() < 8:
        return 0.0
    u, v = complement(d)
    P = np.stack([X[m] @ u, X[m] @ v], 1)
    return float(np.sqrt((P ** 2).sum(1)).mean()) / (float(np.std(t[m])) / span + 0.02)


def base_coverage(X, d, frac=0.08, cells=10):
    """How much of the footprint the lowest slab along d actually fills.

    The cue support_score and floor_score both miss on a cabinet: they ask how
    wide the low end is and how thin, and the top of a bookshelf answers both
    as well as the bottom does.  What separates them is that the base is a
    closed panel and covers the footprint, while the other end is a rim, a
    back edge or an open top and covers only part of it.  So the footprint is
    gridded and the occupied fraction returned.

    Measured on bookshelves: 50% of instances land up within 15 deg against
    43% for support_score, and the frame is stable where the tempting
    variants -- a thicker slab, a finer grid -- are not."""
    d = unit(d)
    t = X @ d
    span = max(float(np.ptp(t)), EPS)
    m = t <= t.min() + frac * span
    if m.sum() < 8:
        return 0.0
    u, v = complement(d)
    P = np.stack([X @ u, X @ v], 1)           # footprint of the whole object
    lo, hi = P.min(0), P.max(0)
    q = np.floor((P[m] - lo) / np.maximum(hi - lo, EPS) * (cells - 1e-9)).astype(int)
    return len(set(map(tuple, q.tolist()))) / float(cells * cells)


def end_spread(X, d, frac=0.16):
    """Perpendicular spread of the two extreme slices along d, as (low, high).
    An aircraft nose is a point, its tail carries the stabilisers."""
    d = unit(d)
    t = X @ d
    lo, hi = t.min(), t.max()
    span = max(hi - lo, EPS)
    u, v = complement(d)
    P = np.stack([X @ u, X @ v], 1)
    r = np.sqrt((P ** 2).sum(1))
    a = r[t <= lo + frac * span]
    b = r[t >= hi - frac * span]
    return (float(a.mean()) if len(a) else 0.0,
            float(b.mean()) if len(b) else 0.0)


def end_profile_vote(X, axis, bands=(0.16, 0.24, 0.32)):
    """Vote for the narrower end (+axis positive), with a dimensionless margin.

    Robust extents and several spatial bands tolerate stabilisers/handles that
    sit inward of the tip. Median aggregation prevents one band dominating.
    The returned confidence measures agreement/strength, not a probability.
    """
    axis = unit(axis)
    t = X @ axis
    lo, hi = np.quantile(t, [0.01, 0.99])
    radius = np.linalg.norm(X - np.outer(t, axis), axis=1)
    votes = []
    for band in bands:
        a = radius[t <= lo + band * (hi - lo)]
        b = radius[t >= hi - band * (hi - lo)]
        if min(len(a), len(b)) >= 8:
            ra, rb = np.quantile(a, 0.8), np.quantile(b, 0.8)
            votes.append(float((ra - rb) / max(ra + rb, EPS)))
    vote = float(np.median(votes)) if votes else 0.0
    return vote, abs(vote)


def crown_direction(X, up, forward, plate_height=None):
    """Point away from a back/headboard using several height bands.

    Offsets are relative to the footprint centre, not the sampling centroid.
    When available, the seat/mattress height excludes the platform itself.
    """
    h, t = X @ up, X @ forward
    lo, hi = np.quantile(t, [0.02, 0.98])
    middle, extent = (lo + hi) / 2, max(hi - lo, EPS)
    offsets = []
    for q in (0.72, 0.82, 0.90):
        cutoff = float(np.quantile(h, q))
        if plate_height is not None:
            cutoff = max(cutoff, plate_height + 0.08 * np.ptp(h))
        crown = t[h > cutoff]
        if len(crown) >= 8:
            offsets.append(float((np.median(crown) - middle) / extent))
    offset = float(np.median(offsets)) if offsets else 0.0
    return (forward if offset <= 0 else -forward), min(1.0, 2 * abs(offset))


def upper_structure_vote(X, up, forward=None):
    """A headboard/cabin occupies less length than the platform below it.

    Spatial bands avoid a dense mattress consuming an entire quantile band.
    Measure spans about each band, so an off-centre headboard is still narrow.
    """
    long_axis = forward
    if long_axis is None:
        long_axis, _, _, _ = _plane_pca_axes(X, up)
    h, t = X @ up, X @ long_axis
    lo, hi = np.quantile(h, [0.01, 0.99])
    votes = []
    for band in (0.12, 0.20, 0.28):
        bottom, top = t[h <= lo + band * (hi - lo)], t[h >= hi - band * (hi - lo)]
        if min(len(bottom), len(top)) >= 8:
            a = float(np.diff(np.quantile(bottom, [0.05, 0.95]))[0])
            b = float(np.diff(np.quantile(top, [0.05, 0.95]))[0])
            votes.append((a - b) / max(a + b, EPS))
    return float(np.median(votes)) if votes else 0.0


def stabiliser_vote(X, forward, lateral):
    """Locate a smaller span lobe separated from the main wings.

    End widths alone fail on short-nosed propeller planes: the main wing can
    enter the nose band. Exclude the main-wing neighbourhood before looking
    for a secondary horizontal surface. A strong fin cue takes precedence
    for canards, whose smaller horizontal surface is at the front.
    """
    t, radius = X @ forward, np.abs(X @ lateral)
    lo, hi = np.quantile(t, [0.01, 0.99])
    edges = np.linspace(lo, hi, 21)
    centres = (edges[1:] + edges[:-1]) / 2
    widths = np.zeros(20)
    for i in range(20):
        # Symmetric inclusion of boundary points matters for meshes with many
        # coplanar samples: reversing an unsigned axis must reverse the vote.
        values = radius[np.abs(t - centres[i]) <= (edges[i + 1] - edges[i]) / 2 + 1e-10]
        if len(values) >= 8:
            widths[i] = np.quantile(values, 0.90)
    wings = float(widths.max())
    if wings <= EPS:
        return 0.0, 0.0
    main = widths >= 0.9 * wings
    wing_centre = float(np.average(centres[main], weights=widths[main]))
    separated = np.abs(centres - wing_centre) > 0.25 * (hi - lo)
    ends = []
    for low in (True, False):
        mask = separated & ((centres < (lo + hi) / 2) if low else
                            (centres > (lo + hi) / 2))
        ends.append(float(np.max(widths[mask])) if mask.any() else 0.0)
    margin = (ends[0] - ends[1]) / max(sum(ends), EPS)
    if max(ends) < 0.20 * wings:
        return 0.0, 0.0
    return margin, abs(margin)


def plane_normal_of_slab(X, d, centre, width=0.09):
    """Refit a plate's normal from the points inside its slab.  Continuous in
    the data, which is what keeps the estimator equivariant."""
    d = unit(d)
    t = X @ d
    span = max(t.max() - t.min(), EPS)
    m = np.abs(t - centre) <= width * span
    if m.sum() < 20:
        return d, 0.0
    Y = X[m] - X[m].mean(0)
    w, V = np.linalg.eigh((Y.T @ Y) / len(Y))
    n = unit(V[:, 0])
    flat = 1.0 - w[0] / max(w[2], EPS)
    if n @ d < 0:
        n = -n
    return n, float(flat)


# ----------------------------------------------------------------------------
# Class-conditional canonicalisation rules
#
# Every rule returns (R, info).  R maps world coordinates to canonical
# coordinates: P = X @ R.T, with z = up, x = forward, y = left.
# ----------------------------------------------------------------------------

def _plane_pca_axes(X, n):
    """Principal directions of the cloud projected onto the plane normal to n,
    returned as (major, minor) 3-vectors.  Closed form, so it is exact and
    continuous rather than a search."""
    u, v = complement(n)
    P = np.stack([X @ u, X @ v], 1)
    C = (P.T @ P) / len(P)
    w, E = np.linalg.eigh(C)
    major = unit(E[0, 1] * u + E[1, 1] * v)
    minor = unit(E[0, 0] * u + E[1, 0] * v)
    return major, minor, float(w[1]), float(w[0])


def _circle_argmax(X, n, fn, n_samples=180, refine=3):
    """Maximise fn(direction) over the full circle of directions orthogonal to n.
    Coarse sweep followed by parabolic refinement of the peak."""
    u, v = complement(n)
    phis = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    vals = np.array([fn(math.cos(p) * u + math.sin(p) * v) for p in phis])
    k = int(np.argmax(vals))
    step = phis[1] - phis[0]
    best_phi, best_val = phis[k], vals[k]
    for _ in range(refine):                       # local trisection around the peak
        step /= 3.0
        for p in (best_phi - step, best_phi + step):
            d = math.cos(p) * u + math.sin(p) * v
            val = fn(d)
            if val > best_val:
                best_phi, best_val = p, val
    return unit(math.cos(best_phi) * u + math.sin(best_phi) * v), float(best_val)


def winged_frame(shape):
    """Use bilateral wings, their plane, then dorsal fin and end profiles.

    The fin is read before choosing the nose: choosing a tail from one narrow
    end slice first makes a missed stabiliser invert BOTH semantic signs.
    Outer wings provide a fallback plane and a datum for fin protrusions.
    Flying wings without a fin fall back to multi-scale end profiles; weak
    evidence is exposed in metadata, never promoted to rotational symmetry.
    """
    X = shape.X
    span, mscore = best_mirror(shape)
    # A nearly flat aircraft also reflects across its wing plane. Trimming
    # away the fin can make that the winning mirror. The lateral direction
    # must carry appreciably more variance than the thin direction.
    cands = np.vstack([span, shape.cands])
    cands = [d for d in cands if np.mean((X @ d) ** 2) > 2 * shape.w[-1]]
    if cands:
        scores = shape.mirror_score(cands, trim=1.0)
        span = unit(cands[int(np.argmax(scores))])
        mscore = float(np.max(scores))
    got = plate_axis(X, span, min_frac=0.12)      # the wing sheet fixes the roll
    if got is not None:
        up = got[0]
        fwd = unit(np.cross(span, up))
        cue = "wing_plate"
    else:
        fwd, up, _, _ = _plane_pca_axes(X, span)
        cue = "plane_pca"

    lateral = np.abs(X @ span)
    wings = X[lateral >= np.quantile(lateral, 0.65)]
    W = wings - wings.mean(0)
    w, V = np.linalg.eigh(W.T @ W / len(W))
    if got is None and w[0] < 0.12 * max(w[1], EPS):
        normal = V[:, 0] - (V[:, 0] @ span) * span
        up = unit(normal)
        fwd = unit(np.cross(span, up))
        cue = "outer_wing_plane"
    h = X @ up
    t = X @ fwd
    lo, hi = np.quantile(t, [0.01, 0.99])
    length = max(hi - lo, EPS)
    centreline = lateral < 0.25 * np.quantile(lateral, 0.98)
    fin_heights = []
    for at_high in (False, True):
        heights = []
        for band in (0.22, 0.32, 0.42):
            end = t >= hi - band * length if at_high else t <= lo + band * length
            values = h[end & centreline]
            if len(values) >= 12:
                q = np.quantile(values, [0.05, 0.5, 0.95])
                # Asymmetric vertical extension rejects a thick, round nose.
                heights.append(float(abs(q[2] + q[0] - 2 * q[1])))
        fin_heights.append(float(np.median(heights)) if heights else 0.0)
    fin_margin = (fin_heights[0] - fin_heights[1]) / max(sum(fin_heights), EPS)
    stabiliser, stabiliser_conf = stabiliser_vote(X, fwd, span)
    # Also search away from the tips: twin fins need not be on the centreline,
    # and a long aft boom can put all stabilisers inside the end bands.
    wing_h = float(np.median(wings @ up))
    residual = h - wing_h
    low, high = np.quantile(residual, [0.02, 0.98])
    dorsal_sign = 1.0 if high >= -low else -1.0
    dorsal_margin = abs(high + low) / max(high - low, EPS)
    dorsal = residual * dorsal_sign
    fin = dorsal > max(0.55 * np.quantile(dorsal, 0.98), 0.05 * length)
    wing_t = float(np.median(wings @ fwd))
    fin_offset = float(np.median(t[fin]) - wing_t) if fin.sum() >= 8 else 0.0
    fin_is_local = (fin.sum() >= 8 and float(np.quantile(lateral[fin], 0.9)) <
                    0.75 * float(np.quantile(lateral, 0.98)))
    wing_chord = float(np.diff(np.quantile(wings @ fwd, [0.1, 0.9]))[0])
    dorsal_found = (dorsal_margin > 0.55 and fin_is_local and
                    abs(fin_offset) > max(0.03 * length, 0.5 * wing_chord))
    if dorsal_found:
        fwd = fwd if fin_offset < 0 else -fwd
        forward_conf = dorsal_margin
        forward_cue = "dorsal_protrusion"
    elif max(fin_heights) > 0.035 * length and abs(fin_margin) > 0.55:
        fwd = fwd if fin_margin > 0 else -fwd
        forward_conf = abs(fin_margin)
        forward_cue = "end_fin_profile"
    elif stabiliser_conf > 0.35:
        fwd = fwd if stabiliser >= 0 else -fwd
        forward_conf = stabiliser_conf
        forward_cue = "secondary_wing_lobe"
    else:
        vote, forward_conf = end_profile_vote(X, fwd)
        fwd = fwd if vote >= 0 else -fwd
        forward_cue = "multi_scale_end_profile"
    # Sign up only after identifying the tail, relative to its own fuselage
    # median. This tolerates high/low wings and pitched wing sheets.
    t = X @ fwd
    tail = h[(t <= np.quantile(t, 0.28)) & centreline]
    if len(tail) < 12:
        tail = h[t <= np.quantile(t, 0.28)]
    q = np.quantile(tail, [0.03, 0.5, 0.97])
    up_vote = float(q[2] + q[0] - 2 * q[1])
    up_conf = abs(up_vote) / max(q[2] - q[0], EPS)
    if up_conf < 0.18:
        # With no clearly resolved fin skew, use the tail's robust extent
        # about the cloud centre, retaining a low confidence.
        up_vote = float(q[2] + q[0])
    if dorsal_found:
        up_vote, up_conf = dorsal_sign, dorsal_margin
    if up_vote < 0:
        up = -up
    return frame_from(fwd, up), {"mirror_score": mscore, "up_cue": cue,
                                 "forward_cue": forward_cue,
                                 "up_confidence": float(up_conf),
                                 "forward_confidence": float(forward_conf),
                                 "symmetry": "I"}


def bilateral_frame(shape, lateral="mirror", up_axis="plate", interior_only=True,
                    min_frac=0.08, plate_cue="plate", project_up=True,
                    up_sign="support", forward="crown", crown_pct=50):
    """Compose lateral, vertical and sign cues selected by a stored recipe."""
    X = shape.X
    info = {}
    if lateral == "mirror":
        lat, info["mirror_score"] = best_mirror(shape)
    else:
        lat = unit(shape.V[:, 0])
    if up_axis == "support":
        up, _ = _circle_argmax(X, lat, lambda d: support_score(X, d))
        cue = "support"
    else:
        got = plate_axis(X, lat, interior_only=interior_only, min_frac=min_frac)
        if got is not None:
            up, cue = got[0], plate_cue
        elif up_axis == "plate_or_pca":
            fwd, up, _, _ = _plane_pca_axes(X, lat)
            cue = "plane_pca"
        else:
            up, _ = _circle_argmax(X, lat, lambda d: support_score(X, d))
            cue = "support"
    if forward == "roof_offset":
        # Historically chosen before signing up; preserve that ordering.
        if up_axis != "plate_or_pca" or got is not None:
            fwd = unit(np.cross(lat, up))
    if project_up:
        up = unit(up - (up @ lat) * lat)
    if up_sign == "upper_structure":
        vote = upper_structure_vote(X, up)
        info["up_confidence"] = abs(vote)
        positive = vote >= 0
    elif up_sign == "taper":
        positive = taper_sign(X, up) >= 0
    else:
        positive = UP_SIGNS[up_sign](X, up)
    if not positive:
        up = -up
    if forward != "roof_offset":
        fwd = unit(np.cross(lat, up))
    h = X @ up
    if forward == "crown_vote":
        fwd, info["forward_confidence"] = crown_direction(X, up, fwd)
    elif forward == "roof_offset":
        hi, span = h.max(), max(h.max() - h.min(), EPS)
        offset = 0.0
        for band in (0.15, 0.20, 0.25):
            mask = h >= hi - band * span
            if mask.sum() >= 10:
                value = float((X[mask] @ fwd).mean())
                if abs(value) > abs(offset):
                    offset = value
        if offset > 0:
            fwd = -fwd
        info["roof_offset"] = abs(offset)
    else:
        crown = X[h >= float(np.percentile(h, crown_pct))]
        if len(crown) > 10 and float((crown @ fwd).mean()) > 0:
            fwd = -fwd
    return frame_from(fwd, up), dict(info, up_cue=cue, symmetry="I")


def principal_frame(shape, up_axis=0, fwd_axis=2, end_frac=0.25):
    """An elongated planar shape: narrow end up, positive face skew forward."""
    X = shape.X
    up, fwd = unit(shape.V[:, up_axis]), unit(shape.V[:, fwd_axis])
    lo, hi = end_spread(X, up, frac=end_frac)
    if hi > lo:
        up = -up
    if float(((X @ fwd) ** 3).mean()) < 0:
        fwd = -fwd
    return frame_from(fwd, up), {"symmetry": "I"}


def crown_forward(shape, R, info):
    forward, confidence = crown_direction(shape.X, R[2], R[0])
    return frame_from(forward, R[2]), dict(info, forward_confidence=confidence)


def axial_symmetry(shape, R, info, allow=("C2z", "C4z", "Cinfz"), min_score=0.0):
    symmetry, scores = detect_axial_symmetry(shape, R[2], allow=allow, min_score=min_score)
    return R, dict(info, symmetry=symmetry, sym_scores=scores)


def plate_sweep(X, n, n_samples=180, width=0.06):
    """Sweep the circle of directions orthogonal to n and record, for each, the
    strongest flat plate and where it sits along that direction.

    The distinction that matters is interior versus terminal: a chair's seat is
    a plate in the middle of the vertical extent, its back is a plate at the end
    of the horizontal extent.  That is what separates up from forward without
    any appeal to which dimension happens to be larger."""
    u, v = complement(n)
    phis = np.linspace(0, np.pi, n_samples, endpoint=False)
    rec = []
    for p in phis:
        d = math.cos(p) * u + math.sin(p) * v
        frac, centre, _ = slab_peak(X, d, width=width, ends_only=False)
        t = X @ d
        lo, hi = t.min(), t.max()
        rel = (centre - lo) / max(hi - lo, EPS)
        rec.append((frac, rel, d, centre))
    return rec


def plate_axis(X, lat, width=0.06, interior_only=False, min_frac=0.08,
               n_samples=180, iters=3):
    """Strongest flat plate whose normal is orthogonal to lat, refined by
    alternately refitting the plate and re-locating its slab.

    Plates are what carry the semantics here: the wing sheet of an aircraft, the
    roof and floor pan of a car, the seat of a chair, the top of a table.  Their
    normals are far better conditioned than in-plane principal axes, which get
    dragged around by any front-to-back asymmetry of the mass."""
    rec = plate_sweep(X, lat, n_samples=n_samples, width=width)
    pool = [r for r in rec if 0.28 <= r[1] <= 0.72] if interior_only else rec
    if not pool:
        return None
    frac, rel, d, centre = max(pool, key=lambda r: r[0])
    if frac < min_frac:
        return None
    for _ in range(iters):
        n, flat = plane_normal_of_slab(X, d, centre, width=width + 0.03)
        n = n - (n @ lat) * lat
        if np.linalg.norm(n) < 1e-6 or flat < 0.85:
            break
        n = unit(n)
        if n @ d < 0:
            n = -n
        if math.degrees(math.acos(float(np.clip(n @ d, -1, 1)))) > 20.0:
            break
        f2, c2, _ = slab_peak(X, n, width=width, ends_only=False)
        if f2 < frac - 0.005:                      # keep only a genuine improvement
            break
        d, centre, frac = n, c2, f2
    return unit(d), float(frac), float(centre), float(rel)


def detect_axial_symmetry(shape, up, allow=("C2z", "C4z", "Cinfz"), min_score=0.0):
    """Measure, rather than assume, the rotational symmetry about the up axis.

    A generic angle gives the chance level of the matching score for this cloud,
    which is what the candidate symmetries are compared against.  Only the
    symmetries the class can actually have are tested, so a car is never granted
    a 180 degree ambiguity just because its body is roughly a box. min_score
    optionally requires an absolute match as well as the relative margin."""
    generic = float(np.mean([shape.rot_score(up, angles=(a,))
                             for a in (0.77, 1.31, 2.21)]))
    s180 = shape.rot_score(up, angles=(np.pi,))
    s90 = shape.rot_score(up, angles=(np.pi / 2,))
    sc = {"generic": generic, "s180": s180, "s90": s90}
    if "Cinfz" in allow and generic >= max(min_score, 0.85 * max(s180, s90)) and s180 > 1e-6:
        return "Cinfz", sc
    if "C4z" in allow and s90 >= max(min_score, 0.90 * s180) and s180 >= 1.25 * generic:
        return "C4z", sc
    if "C2z" in allow and s180 >= max(min_score, 1.30 * generic):
        return "C2z", sc
    return "I", sc


def detect_flip_symmetry(shape, fwd, factor=1.30):
    """Does turning this object upside down leave it unchanged?

    Asked of a shelf unit, because the answer is often yes: strip a bookcase of
    its back and its shelves and what is left is an open rectangular tube with
    a flat panel at each end, and no cue can say which panel is the floor
    because the object itself does not distinguish them.  Measured per instance
    against a generic-angle baseline, exactly as detect_axial_symmetry does it,
    so a cabinet with a plinth or a closed top is not granted the ambiguity."""
    generic = float(np.mean([shape.rot_score(fwd, angles=(a,))
                             for a in (0.77, 1.31, 2.21)]))
    s180 = float(shape.rot_score(fwd, angles=(np.pi,)))
    return ("C2x" if s180 >= factor * generic else "I"), {"s180": s180,
                                                          "generic": generic}


def platform_frame(shape):
    """Up is the normal of the dominant flat plate, which is the top, refined by
    fitting that plate.  In plane, the long side of the top's minimum-area
    rectangle becomes x.  The top's outline sets the symmetry group."""
    from scipy.spatial import ConvexHull
    X = shape.X

    best = (-1.0, None, None, None)
    for d in shape.grid(128):
        frac, centre, at_high = slab_peak(X, d, width=0.07, ends_only=True)
        if frac > best[0]:
            best = (frac, unit(d), centre, at_high)
    frac, up, centre, at_high = best

    for _ in range(3):                            # alternate: fit plate, re-slab
        n, flat = plane_normal_of_slab(X, up, centre)
        if flat < 0.85:
            break
        up = n
        frac, centre, at_high = slab_peak(X, up, width=0.07, ends_only=True)
    if not at_high:                               # put the top on the +z side
        up, centre = -up, -centre

    t = X @ up                                    # in-plane orientation from the top
    span = max(t.max() - t.min(), EPS)
    m = np.abs(t - centre) <= 0.10 * span
    u, v = complement(up)
    P2 = np.stack([X[m] @ u, X[m] @ v], 1) if m.sum() >= 16 else np.stack([X @ u, X @ v], 1)
    P2 = P2 - P2.mean(0)

    angle, area_best, dims = 0.0, np.inf, (1.0, 1.0)
    try:
        hull = P2[ConvexHull(P2).vertices]
        edges = np.roll(hull, -1, 0) - hull
        for e in edges:                           # rotating calipers
            L = np.linalg.norm(e)
            if L < 1e-9:
                continue
            c, s_ = e[0] / L, e[1] / L
            Rw = np.array([[c, s_], [-s_, c]])
            Z = hull @ Rw.T
            ext = Z.max(0) - Z.min(0)
            if ext[0] * ext[1] < area_best:
                area_best, angle, dims = ext[0] * ext[1], math.atan2(s_, c), tuple(ext)
    except Exception:
        w2, E2 = np.linalg.eigh((P2.T @ P2) / max(len(P2), 1))
        angle = math.atan2(E2[1, 1], E2[0, 1])
        dims = (float(np.sqrt(w2[1])), float(np.sqrt(w2[0])))

    long_dir = (math.cos(angle) * u + math.sin(angle) * v)
    if dims[1] > dims[0]:
        long_dir = -math.sin(angle) * u + math.cos(angle) * v
    aspect = max(dims) / max(min(dims), EPS)
    sym, sd = detect_axial_symmetry(shape, up)
    return frame_from(long_dir, up), {"top_fraction": float(frac),
                                      "aspect": float(aspect),
                                      "symmetry": sym, "sym_scores": sd}


# ----------------------------------------------------------------------------
# Five more classes, on the same skeleton: a lateral axis, an up axis, and the
# signs that separate front from back.  What differs between them is only which
# cue is trusted for each, and the docstrings say what was measured to pick it.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Reusable sign cues and frame constructors. Class-specific choices live in
# rules.json; see RULE_DATABASE.md for the operation contract and extension guide.
# ----------------------------------------------------------------------------


def _sign_support(X, u):
    return support_score(X, u) >= support_score(X, -u)


def _sign_floor(X, u):
    return floor_score(X, u) >= floor_score(X, -u)


def _sign_top_short(X, u):
    return top_is_shorter(X, u)


def _sign_wide_up(X, u):
    """Hollow things -- a tub, a screen on a stand -- carry their width at the
    open end, which is the end that points up."""
    return taper_sign(X, u) < 0


def _sign_base(X, u):
    """Down is the end whose slab covers the footprint: a closed base."""
    return base_coverage(X, u) >= base_coverage(X, -u)


def _sign_third(X, u):
    """Heavy end low, by the third moment along u."""
    return float(((X @ u) ** 3).mean()) <= 0


UP_SIGNS = {"support": _sign_support, "floor": _sign_floor,
            "top_short": _sign_top_short, "wide_up": _sign_wide_up,
            "third": _sign_third, "base": _sign_base}


def upright_frame(shape, up_axis="plate", up_sign="support", crown_pct=72,
                  flip_symmetry=False):
    """Bilaterally symmetric, stands on a floor, tall part at one end.

    The mirror normal is the lateral axis; up is found inside the plane it
    spans, either as the strongest interior plate (a seat, a lid, a mattress)
    or as the direction the object reaches furthest along; and the crown -- the
    top slice -- sits behind, which fixes forward.
    """
    X = shape.X
    lat, mscore = best_mirror(shape)

    if up_axis == "principal":
        # The object's OWN axes, not a swept direction.  Sweeping the circle
        # for maximum extent finds the diagonal of the height-by-depth
        # rectangle rather than the height: for a wardrobe 2.0 tall and 0.6
        # deep that diagonal sits 17 deg off vertical, which is the whole of
        # the error.  Measured on bookshelves, sweeping puts up 24.4 deg out
        # (45% within 15 deg) where the principal axis puts it 1.2 deg out
        # (85%); on wardrobes 25.2 deg / 35% against 0.9 deg / 80%.
        cands = [unit(shape.V[:, i]) for i in range(3)]
        cands = [c for c in cands if abs(c @ lat) < 0.5] or cands
        up = max(cands, key=lambda c: float(np.ptp(X @ c)))
        cue = "principal"
    elif up_axis == "sharpest":
        # The densest flat plate in any direction, wherever it sits along its
        # own extent.  plate_axis asks for an INTERIOR plate, which is right
        # for a seat with a back above it and wrong for a bench: strip the back
        # off and the seat is the top surface, with nothing above it to make it
        # interior, so the test rejects the one plate that matters.  Measured
        # on twenty benches, the interior test puts up 7.3 deg out (65% within
        # 15 deg) and this puts it 1.2 deg out (80%).
        up, _ = _circle_argmax(X, lat, lambda d: slab_peak(X, d, ends_only=False)[0])
        cue = "sharpest_plate"
    elif up_axis == "tallest":
        up, _ = _circle_argmax(X, lat, lambda d: float(np.ptp(X @ unit(d))))
        cue = "tallest"
    else:
        got = plate_axis(X, lat, interior_only=True)
        if got is None:
            up, _ = _circle_argmax(X, lat, lambda d: support_score(X, d))
            cue = "support"
        else:
            up, cue = got[0], "plate"
    up = unit(up - (up @ lat) * lat)
    if not UP_SIGNS[up_sign](X, up):
        up = -up

    fwd = unit(np.cross(lat, up))
    h = X @ up
    crown = X[h >= float(np.percentile(h, crown_pct))]
    if len(crown) > 10 and float((crown @ fwd).mean()) > 0:
        fwd = -fwd
    sym, sc = "I", None
    if flip_symmetry:
        sym, sc = detect_flip_symmetry(shape, fwd)
    return frame_from(fwd, up), {"mirror_score": mscore, "up_cue": cue,
                                 "symmetry": sym, "flip_scores": sc}


def revolution_frame(shape, wide_end_up=True, sign_forward=True,
                     narrow_tie_up=False, end_frac=0.22):
    """A surface of revolution: the axis carries everything, the azimuth is
    free, and the only question is which end of the axis is up. Stored
    recipes select the end sign and whether to sign the in-plane seed."""
    X = shape.X
    scores = np.array([shape.rot_score(d) for d in shape.cands])
    axis = unit(shape.cands[int(np.argmax(scores))])
    k = int(np.argmax([abs(shape.V[:, i] @ axis) for i in range(3)]))
    snapped = unit(shape.V[:, k])                  # snap to the exact eigenvector
    axis = snapped if snapped @ axis > 0 else -snapped

    lo, hi = end_spread(X, axis, frac=end_frac)
    up = axis if ((hi >= lo) == wide_end_up) else -axis
    if narrow_tie_up and hi == lo:
        up = axis
    sym, sd = detect_axial_symmetry(shape, up)
    j = 0 if k != 0 else 1                         # any equivariant in-plane seed
    seed = shape.V[:, j] - (shape.V[:, j] @ up) * up
    fwd = unit(seed) if np.linalg.norm(seed) > 1e-6 else complement(up)[0]
    if sign_forward and float(np.mean((X @ fwd) ** 3)) < 0:
        fwd = -fwd
    return frame_from(fwd, up), {"rot_score": float(scores.max()),
                                 "symmetry": sym, "sym_scores": sd}


def slab_frame(shape, up_axis=0, fwd_axis=2, up_sign="third"):
    """Thin and rectangular, so the inertia tensor hands over both axes and
    only the signs are left.  Declared C2z, which is honest for a slab and has
    the useful consequence that the face sign comes free: flipping forward
    flips lateral with it, and that pair is exactly the C2z image."""
    X = shape.X
    up = unit(shape.V[:, up_axis])
    fwd = unit(shape.V[:, fwd_axis])
    if not UP_SIGNS[up_sign](X, up):
        up = -up
    fwd = unit(fwd - (fwd @ up) * up)
    # An eigensolver's eigenvector sign is not a geometric face cue. Fix it
    # explicitly so an asymmetric panel remains stable when C2z is rejected.
    if float(np.mean((X @ fwd) ** 3)) < 0:
        fwd = -fwd
    return frame_from(fwd, up), {"symmetry": "C2z"}


def open_basin_frame(shape):
    """An elongated open basin: closed floor below, open rim above.

    Evaluate both signs of candidate normals using central coverage of the
    end slabs. A side wall cannot win merely because it is a large plate:
    the opposite end must actually have an opening. Works for sinks too.
    """
    X = shape.X
    candidates = list(shape.V.T)
    for d in shape.V.T:
        _, centre, _ = slab_peak(X, d, width=0.10)
        normal, flat = plane_normal_of_slab(X, d, centre, width=0.13)
        if flat > 0.85:
            candidates.append(normal)
    best = (-np.inf, None)
    for axis in candidates:
        for up in (axis, -axis):
            t = X @ up
            lo, hi = np.quantile(t, [0.01, 0.99])
            a, b, _, _ = _plane_pca_axes(X, up)
            P = np.column_stack([X @ a, X @ b])
            bounds = np.quantile(P, [0.02, 0.98], axis=0)
            Q = (P - bounds.mean(0)) / np.maximum(np.diff(bounds, axis=0)[0] / 2, EPS)
            central = np.max(np.abs(Q), axis=1) < 0.55
            votes = []
            for band in (0.12, 0.22, 0.32):
                bottom = t <= lo + band * (hi - lo)
                top = t >= hi - band * (hi - lo)
                if min(bottom.sum(), top.sum()) >= 12:
                    votes.append(float(central[bottom].mean() - central[top].mean()))
            score = float(np.median(votes)) if votes else -1.0
            if score > best[0]:
                best = (score, up)
    confidence, up = best
    fwd, _, _, _ = _plane_pca_axes(X, up)
    fwd, front_conf = crown_direction(X, up, fwd)
    sym, scores = detect_axial_symmetry(shape, up, allow=("C2z",), min_score=0.65)
    return frame_from(fwd, up), {"up_cue": "closed_floor_open_rim",
                                "up_confidence": max(0.0, float(confidence)),
                                "forward_confidence": front_conf,
                                "symmetry": sym, "sym_scores": scores}


def panel_symmetry(shape, R, info, min_score=0.75):
    # Featureless slabs have three proper half-turns, not just yaw. A strong
    # self-match is required on every generator and its product; this is a
    # measured ambiguity, not permission to ignore a bad semantic sign.
    scores = [float(shape.rot_score(axis, angles=(np.pi,))) for axis in R]
    valid = [score >= min_score for score in scores]
    if all(valid):
        symmetry = "D2"
    elif valid[2]:
        symmetry = "C2z"
    elif valid[0]:
        symmetry = "C2x"
    elif valid[1]:
        symmetry = "C2y"
    else:
        symmetry = "I"
    return R, dict(info, symmetry=symmetry, half_turn_scores=scores)


def supported_case_frame(shape):
    """Two shapes share this class, so the rule first tells them apart.

    A grand stands on three legs under a flat case, so the low end of its
    thinnest principal axis is sparse and wide: support_score along PC3 is
    2.5-7.7 times its value along the other two axes on every grand measured,
    and about 1 on uprights.  A grand therefore takes up from PC3, signed by
    support, and forward along PC1 towards the wide keyboard end; the tail
    tapers.  best_mirror is not used for it: a grand's curved side leaves it
    without a left-right mirror, and the plane found instead scattered
    forward across the class.

    An upright is a box standing on end: PC1 runs across the keyboard, up is
    PC2, signed by support (the keyboard brackets), and forward is the thin
    PC3.  Taken from the inertia tensor rather than best_mirror for the same
    reason as a wide-backed seat: through the mirror, two near-cubic instances wandered
    60-80 deg between poses of the same cloud.

    Measured on 25 instances against the stored z-up, the old rule (lid plate,
    support) put up within 15 deg on 10 and lost nearly every upright, which
    it laid on its back.  This puts 17 within 15 deg; stability 6.1 -> 0.0
    deg, consistency median 94 -> 40 deg, RMS 98 -> 82.  Forward is still the
    weak part: the grands agree with each other, the uprights less so.

    Support alone mistook two uprights for grands: an upright lying on its
    back also has a sparse, wide low end (the keyboard and its brackets).
    What legs add is empty space: under a grand the lowest 30% of the height
    holds under 5.5% of the points (0-5% on every grand but one), an
    upright's keyboard end holds 6-18%.  Requiring both sorts 28/30 correctly
    and takes consistency RMS 81 -> 76 deg on all 30, with the even half
    improving (82 -> 73) and the odd half unchanged.

    Measured and rejected, so nobody repeats them: reference snapping (RMS
    84-97 with 0-3 clusters); forward from the top-view bounding rectangle
    (more often semantically right -- the dataset stores grands 6, 12, 21 and
    26 turned 90 deg -- but it scores worse against the stored poses); a
    majority vote of support/floor/third for the upright sign (fixes 7,
    breaks 16); choosing the upright's up between PC1 and PC2 by support,
    floor or the mirror (11/14 each, the same as PC2 alone)."""
    X, V = shape.X, shape.V
    sup = [max(support_score(X, V[:, i]), support_score(X, -V[:, i])) for i in range(3)]
    up = unit(V[:, 2])
    if support_score(X, up) < support_score(X, -up):
        up = -up
    h = X @ up                                     # legs leave the low 30% nearly empty
    legs = float(np.mean(h <= h.min() + 0.30 * np.ptp(h))) < 0.055
    if sup[2] < 2.2 * max(sup[:2]) or not legs:    # no legs under the case: upright
        R, info = slab_frame(shape, up_axis=1, fwd_axis=2, up_sign="support")
        return R, dict(info, symmetry="I")        # a piano has no half-turn

    fwd = unit(V[:, 0])
    lo, hi = end_spread(X, fwd)                    # keyboard end is wide, tail tapers
    if lo > hi:
        fwd = -fwd
    return frame_from(fwd, up), {"up_cue": "grand_legs", "symmetry": "I"}


def profile_box_frame(shape):
    """Reported as a failure: up lands within 15 deg on 36% of instances.

    The shape is right there -- a bowl at mid height, a cistern standing at the
    back, a pedestal on the floor -- and none of it is readable from 1024
    points.  What was measured, all of it on the twenty instances:

        up axis, sweeping the plane the mirror leaves free
            interior plate (the seat)                  40%
            terminal plate allowed                     40%
            sharpest plate anywhere                    45%
            flat foot                                  10%
            base coverage                              40%
            perpendicular to the most terminal plate   35%
            tallest direction in the plane              5%
            widest mid-height cut                       0%
            ring-with-a-hole at the seat               40%
        up sign, over the two best axes
            support, floor, base, top_is_shorter, crown at 72 and 88
            -- best whole-frame result 35%
        consensus instead of a cue
            one reference ensemble                     12%
            three clusters                              0%
            snapping to a template built from the
            dataset's own stored poses                 33%

    The cues disagree with each other rather than with the truth, which is what
    it looks like when a class has no single geometry: a toilet's height and
    its depth are within 15% of each other, so nothing separates up from
    forward by size, and the cistern is present on some models and absent on
    others.  None of them was better than wrong more often than right.

    What does work is the side profile's bounding rectangle (below): the axis
    that best reads as a support is up, the wide end (bowl rim, not the
    pedestal) signs it, and the empty top-front corner gives forward.  Up
    within 15 deg on 13/20, up from 6/20; consistency median 64 -> 5 deg,
    within 10 deg 20% -> 55%.  The seven still wrong are the boxy models with
    the bowl hidden inside a skirt, whose profile has no empty corner to read.

    Measured and rejected for the choice of up edge and its sign: tallest,
    shortest, plate, end plate and floor against support, with every UP_SIGN
    -- the lowest-RMS of those put up right on 0-2 of 20, agreeing only by
    being uniformly wrong.  The same empty-corner forward on upright pianos
    made them worse (RMS 76 -> 96)."""
    X = shape.X
    lat, mscore = best_mirror(shape)

    # The mirror plane is right on all twenty; what failed was up inside it,
    # which sits diagonally between the principal axes of the L-shaped side
    # profile.  The tightest rectangle around that profile has its edges on
    # the floor and the back wall: one edge is within 15 deg of up on 18/20.
    a, _ = _circle_argmax(X, lat, lambda d: -float(np.ptp(X @ d) * np.ptp(X @ np.cross(lat, d))))
    b = unit(np.cross(lat, a))
    up = max((a, b), key=lambda d: max(support_score(X, d), support_score(X, -d)))
    if not _sign_wide_up(X, up):                   # bowl rim wider than the pedestal
        up = -up

    # Forward from the empty corner of the profile: above the bowl and in
    # front of the cistern.  The crown test (cistern mean behind) flipped the
    # boxy models, whose top slice is the lid as much as the cistern; counting
    # which half of the top is emptier takes consistency RMS 101 -> 85 deg and
    # the median 13 -> 5, better on both odd and even halves.
    fwd = unit(np.cross(lat, up))
    h = X @ up
    f = X @ fwd
    top = h > np.median(h)
    if np.sum(top & (f > np.median(f))) > np.sum(top & (f < np.median(f))):
        fwd = -fwd
    return frame_from(fwd, up), {"mirror_score": mscore, "up_cue": "profile_box",
                                 "symmetry": "I"}


def end_closure(shape, R, info, min_score=0.65):
    """Open vessel; the closed base fixes up even when the neck is narrow.

    Width alone cannot distinguish a flared pot from a narrow-necked vase.
    First find the rotational axis and refine against a terminal plate. Then
    compare central coverage at each end using that end's OWN radius, so a
    narrow neck is not mistaken for a filled base. Low confidence remains
    explicit when neither end has enough interior samples.
    """
    X, up = shape.X, R[2]
    _, centre, _ = slab_peak(X, up, width=0.06)
    normal, flat = plane_normal_of_slab(X, up, centre, width=0.08)
    if flat > 0.95 and normal @ up > np.cos(np.deg2rad(20)):
        up = normal
    h = X @ up
    lo, hi = np.quantile(h, [0.01, 0.99])
    radial = X - np.outer(h, up)
    votes = []
    for band in (0.06, 0.10, 0.16):
        coverage = []
        for mask in (h <= lo + band * (hi - lo), h >= hi - band * (hi - lo)):
            points = radial[mask]
            if len(points) < 12:
                coverage.append(0.0)
                continue
            radius = np.linalg.norm(points - np.median(points, axis=0), axis=1)
            coverage.append(float(np.mean(radius < 0.55 * np.quantile(radius, 0.90))))
        votes.append(coverage[0] - coverage[1])
    vote = float(np.median(votes))
    if vote < 0:
        up = -up
    symmetry, scores = detect_axial_symmetry(shape, up, min_score=min_score)
    return frame_from(R[0], up), dict(info, up_cue="local_end_closure",
                                     up_confidence=abs(vote), symmetry=symmetry,
                                     sym_scores=scores)


def octahedral_rotations():
    """The 24 rotations that map the coordinate axes onto themselves.

    These are exactly the mistakes the rules make.  Every cue that fixes a sign
    or picks which axis is up is a discrete choice on a continuous statistic,
    so when it fails it does not fail by a few degrees -- it swaps an axis or
    turns the object end for end, which is one of these 24."""
    import itertools
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if abs(np.linalg.det(M) - 1.0) < 1e-9:
                mats.append(M)
    return mats


OCTAHEDRAL = octahedral_rotations()

# Only the classes whose rules fail by swapping an axis get a reference.  A
# table or a bowl is symmetric about its own vertical, so several attitudes are
# genuinely equivalent, the nearest-reference choice among them is arbitrary,
# and forcing one costs accuracy instead of adding any.

# How many shape clusters each class's reference is split into.  One means a
# single ensemble, which is right for classes whose members look alike.
#
# Chairs need more.  Measured on real ShapeNet with held-out instances, a single
# chair ensemble scored a median of 94 degrees with nothing inside 10; split
# into four clusters the same machinery gives a median of 9.2 degrees with 54%
# inside 10.  The synset holds dining chairs, stools, benches and armchairs, so
# matching an instance against the class as a whole is decided by shape mismatch
# rather than by attitude.  Aligning within a cluster, then aligning the handful
# of cluster references to each other, matches like with like.
#
# Measured on 50-60 real instances with held-out scoring: chairs go from nothing
# inside 10 degrees to 60% with four clusters, and cars from a median of 3.3 to
# 1.6.  Airplanes get worse -- 84% down to 63% -- and their stability rises from
# 0.1 to 3.2 degrees, which gives the reason away: stability cannot move at all
# under --pca-first unless a discrete choice is wavering, and the only new one is
# the cluster assignment.  Airplane clusters are not cleanly separated, so a
# borderline plane lands in different clusters for different input poses.  They
# keep a single ensemble.
# Bed and bathtub references are supported by disjoint validation: they repair
# ambiguous head/foot signs and occasional plate-axis swaps. The same trial
# regressed monitor/piano and did not fix sofa/toilet, so those remain rules
# only. Every ICP correction remains bounded to 25 degrees.


def _one_way_chamfer(P, tree):
    d, _ = tree.query(P, workers=-1)
    return float(d.mean())


def _trimmed_cost(P, trees, keep=0.5):
    """Distance from a cloud to an ensemble, using only the nearest half of the
    members.  A class holds several shapes -- a stool, an armchair, a bench --
    and a cloud only has to match the members it resembles.  Averaging over all
    of them lets the unlike ones drown out the signal."""
    d = sorted(_one_way_chamfer(P, t) for t in trees)
    k = max(1, int(round(keep * len(d))))
    return float(np.mean(d[:k]))


def snap_to_reference(P, trees, n_pts=384, coarse_pts=128, coarse_trees=4,
                      shortlist=5, rng=None, rotations=None, min_improvement=0.0):
    """Pick the axis assignment that puts this cloud in the same attitude as the
    class reference.  Returned rotation is applied on the left of the rule's.

    Scoring all 24 assignments against the whole ensemble costs more than the
    rule itself, so a cheap pass on a sparse subsample and a few members picks a
    shortlist and only those are scored properly.  The coarse pass only has to
    rank the right answer in the top few, which is easy: a wrong axis assignment
    is wrong by tens of degrees, not by a hair."""
    if not isinstance(trees, (list, tuple)):
        trees = [trees]
    rng = rng or np.random.default_rng(0)
    if len(P) > n_pts:
        P = P[rng.choice(len(P), n_pts, replace=False)]
    Pc = P[::max(1, len(P) // coarse_pts)]
    tc = trees[:coarse_trees]
    rotations = OCTAHEDRAL if rotations is None else rotations
    coarse = sorted((_trimmed_cost(Pc @ F.T, tc), i) for i, F in enumerate(rotations))
    identity_cost = _trimmed_cost(P, trees)
    best, bestF = identity_cost, np.eye(3)
    for _, i in coarse[:shortlist]:
        F = rotations[i]
        c = _trimmed_cost(P @ F.T, trees)
        if c < best:
            best, bestF = c, F
    if best >= identity_cost * (1.0 - min_improvement):
        return np.eye(3)
    return bestF


def _kmeans(F, k, seed=0, iters=60):
    """Plain k-means with k-means++ seeding, on standardised features."""
    rng = np.random.default_rng(seed)
    n = len(F)
    k = max(1, min(k, n))
    centres = [F[rng.integers(n)]]
    for _ in range(k - 1):
        d = np.min([((F - c) ** 2).sum(1) for c in centres], axis=0)
        tot = float(d.sum())
        centres.append(F[rng.choice(n, p=d / tot)] if tot > 1e-12 else F[rng.integers(n)])
    C = np.array(centres)
    lab = np.full(n, -1)
    for _ in range(iters):
        new_lab = np.argmin(((F[:, None, :] - C[None, :, :]) ** 2).sum(2), axis=1)
        if np.array_equal(new_lab, lab):
            break
        lab = new_lab
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = F[m].mean(0)
    return lab, C


def _joint_align(Ps, iters=5):
    """Turn a set of canonical clouds into agreement with each other, by letting
    each repeatedly re-pick its axis assignment to fit the others.  No instance
    has to be right on its own; the correct attitude is simply the one they can
    all agree on."""
    Ps = [P.copy() for P in Ps]
    trees = [cKDTree(P) for P in Ps]
    for _ in range(iters):
        moved = False
        for i in range(len(Ps)):
            others = [t for j, t in enumerate(trees) if j != i]
            if not others:
                continue
            F = min(OCTAHEDRAL, key=lambda M: _trimmed_cost(Ps[i] @ M.T, others))
            if not np.allclose(F, np.eye(3)):
                Ps[i] = Ps[i] @ F.T
                trees[i] = cKDTree(Ps[i])
                moved = True
        if not moved:
            break
    return Ps


def _align_clusters(groups):
    """Bring separately aligned clusters into one common attitude.  Only a few
    coherent groups take part, which is the easy half of the problem and the
    reason for clustering first."""
    keys = list(groups)
    if len(keys) < 2:
        return {k: np.eye(3) for k in keys}
    anchor = max(keys, key=lambda k: len(groups[k]))
    anchor_trees = [cKDTree(P) for P in groups[anchor]]
    out = {anchor: np.eye(3)}
    for k in keys:
        if k == anchor:
            continue
        best, bestF = np.inf, np.eye(3)
        for F in OCTAHEDRAL:
            c = float(np.mean([_trimmed_cost(P @ F.T, anchor_trees) for P in groups[k]]))
            if c < best:
                best, bestF = c, F
        out[k] = bestF
    return out


def build_reference(clouds, cls, rule_canon, max_ref=10, n_pts=384, seed=0,
                    n_clusters=None, *, rule_database=None):
    """Reference attitude for a class: ensembles of its own instances, turned
    into agreement with each other.

    With `n_clusters` above one the instances are first split by rotation-
    invariant shape features, aligned within each group, and the groups then
    aligned to each other, so a stool is never matched against an armchair.

    The rules still have to get the frame right up to an axis swap, because that
    is all this repairs.  The ensembles come from the first `max_ref` instances,
    so their own numbers are optimistic; judge the method on the rest."""
    database = RULE_DATABASE if rule_database is None else rule_database
    cls = database.canonical_name(cls)
    policy = database.reference_policy(cls)
    n_clusters = policy.get("clusters", 1) if n_clusters is None else n_clusters
    rng = np.random.default_rng(seed)
    Ps, feats = [], []
    for X in clouds[:max_ref]:
        try:
            R, _ = rule_canon(X, cls)
            f = invariant_features(Shape(X))
        except Exception:
            continue
        P = normalise_cloud(X)[0] @ R.T
        if len(P) > n_pts:
            P = P[rng.choice(len(P), n_pts, replace=False)]
        Ps.append(P)
        feats.append(f)
    if len(Ps) < 3:
        return None

    F = np.array(feats)
    mu, sd = F.mean(0), F.std(0) + EPS
    Fz = (F - mu) / sd

    if policy.get("preserve_semantics", False):
        # Keep the semantic attitude of the rules. Free joint alignment can
        # turn an entire ensemble tail-first, then undo a correct fin cue on
        # every unseen aircraft. Shape matching only resolves weak axes.
        return {"clusters": {0: Ps}, "centres": np.zeros((1, F.shape[1])),
                "mu": mu, "sd": sd, "flat": True}

    if n_clusters <= 1:
        return {"clusters": {0: _joint_align(Ps)}, "centres": np.zeros((1, F.shape[1])),
                "mu": mu, "sd": sd, "flat": True}

    lab, C = _kmeans(Fz, n_clusters, seed=seed)
    groups = {}
    for j in sorted(set(int(v) for v in lab)):
        idx = [i for i, l in enumerate(lab) if int(l) == j]
        if len(idx) < 2:                       # a singleton cannot self-align
            continue
        groups[j] = _joint_align([Ps[i] for i in idx])
    if not groups:
        return {"clusters": {0: _joint_align(Ps)}, "centres": np.zeros((1, F.shape[1])),
                "mu": mu, "sd": sd, "flat": True}
    turns = _align_clusters(groups)
    return {"clusters": {j: [P @ turns[j].T for P in groups[j]] for j in groups},
            "centres": C, "mu": mu, "sd": sd, "flat": False}


def refine_to_reference(P, tree, Q, max_deg=25.0, iters=6, trim=0.7):
    """Small rotation-only ICP onto one reference member.

    The axis snap repairs a frame that is wrong by a whole axis.  It cannot
    touch a frame that is wrong by twenty degrees, which is what a mirror plane
    landing slightly off produces, and that shows up as objects sitting visibly
    tilted.  This closes that gap, and is capped: a correction larger than
    `max_deg` means the reference is the wrong shape to be matching against, and
    is thrown away rather than allowed to drag the frame somewhere worse."""
    R = np.eye(3)
    for _ in range(iters):
        Pr = P @ R.T
        d, j = tree.query(Pr, workers=-1)
        keep = d <= max(np.quantile(d, trim), 1e-9)
        A, B = Pr[keep], Q[j[keep]]
        if len(A) < 12:
            break
        U, _, Vt = np.linalg.svd(A.T @ B)
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
        M = Vt.T @ D @ U.T
        R = M @ R
        if math.degrees(math.acos(float(np.clip((np.trace(M) - 1) / 2, -1, 1)))) < 0.05:
            break
    ang = math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return R if ang <= max_deg else np.eye(3)


def save_references(path, refs, extra=None, *, rule_database=None):
    """Write the built references to a .npz so they never have to be rebuilt.

    The file carries the clouds of every cluster plus the feature statistics
    used to assign a new instance to one, and it records whether the references
    were built with the principal-axis pre-canonicalisation on.  Aligning a new
    cloud under a different setting from the one the references were built with
    would silently mis-assign, so the setting travels with the file."""
    database = RULE_DATABASE if rule_database is None else rule_database
    arrays, meta = {}, {"pca_first": bool(PCA_FIRST), "classes": {},
                        "ruleset_version": RULESET_VERSION,
                        "rules_fingerprint": database.fingerprint}
    for cls, ref in (refs or {}).items():
        if not ref or not ref.get("clusters"):
            continue
        meta["classes"][cls] = {"flat": bool(ref.get("flat", False)),
                                "clusters": {str(j): len(Ps)
                                             for j, Ps in ref["clusters"].items()}}
        arrays[f"{cls}|centres"] = np.asarray(ref["centres"], np.float32)
        arrays[f"{cls}|mu"] = np.asarray(ref["mu"], np.float32)
        arrays[f"{cls}|sd"] = np.asarray(ref["sd"], np.float32)
        for j, Ps in ref["clusters"].items():
            for i, P in enumerate(Ps):
                arrays[f"{cls}|c{j}|{i}"] = np.asarray(P, np.float32)
    if extra:
        meta.update(extra)
    arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(path, **arrays)
    return meta


def load_references(path, *, rule_database=None):
    """Read references written by save_references.  Returns (refs, meta)."""
    database = RULE_DATABASE if rule_database is None else rule_database
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["__meta__"]).decode())
    if (meta.get("ruleset_version") != RULESET_VERSION or
            meta.get("rules_fingerprint") != database.fingerprint):
        warnings.warn("Reference cache has different or unrecorded orientation rules; "
                      "rebuild it with --save-reference before comparing results.",
                      UserWarning, stacklevel=2)
    refs = {}
    for cls, m in meta["classes"].items():
        clusters = {}
        for j, n in m["clusters"].items():
            clusters[int(j)] = [np.asarray(z[f"{cls}|c{j}|{i}"], float) for i in range(n)]
        refs[cls] = {"clusters": clusters,
                     "centres": np.asarray(z[f"{cls}|centres"], float),
                     "mu": np.asarray(z[f"{cls}|mu"], float),
                     "sd": np.asarray(z[f"{cls}|sd"], float),
                     "flat": bool(m["flat"])}
    return refs, meta


def make_canonicaliser(references=None, refine=True, refine_classes=None, *, rule_database=None):
    """Bind class references to the pipeline.  Without them this is the plain
    rule-based canonicaliser."""
    database = RULE_DATABASE if rule_database is None else rule_database
    store = {}
    for c, ref in (references or {}).items():
        if not ref or not ref.get("clusters"):
            continue
        store[database.canonical_name(c)] = {"trees": {j: [cKDTree(P) for P in Ps]
                              for j, Ps in ref["clusters"].items()},
                    "pts": ref["clusters"],
                    "centres": ref["centres"], "mu": ref["mu"], "sd": ref["sd"],
                    "flat": ref.get("flat", False)}

    def canon(X, cls):
        cls = database.canonical_name(cls)
        R, info = canonicalise(X, cls, rule_database=database)
        policy = database.reference_policy(cls)
        ref = store.get(cls)
        if not ref:
            return R, info

        keys = sorted(ref["trees"])
        if ref["flat"] or len(keys) == 1:
            j = keys[0]
        else:                                  # match against this shape's own kind
            try:
                f = (invariant_features(Shape(X)) - ref["mu"]) / ref["sd"]
                j = keys[int(np.argmin([((ref["centres"][k] - f) ** 2).sum()
                                        if k < len(ref["centres"]) else np.inf
                                        for k in keys]))]
            except Exception:
                j = keys[0]
        ens, pl = ref["trees"][j], ref["pts"][j]

        P = normalise_cloud(X)[0] @ R.T
        rotations = OCTAHEDRAL
        locked = []
        if "lock_confidence" in policy:
            for axis, name in ((0, "forward"), (2, "up")):
                if info.get(name + "_confidence", 0.0) >= policy["lock_confidence"]:
                    rotations = [F for F in rotations if F[axis, axis] > 0.99]
                    locked.append(name)
        F = snap_to_reference(P, ens, rotations=rotations,
                              min_improvement=policy.get("min_improvement", 0.0))
        R = F @ R
        tweak = 0.0
        if refine and (refine_classes is None or cls in refine_classes):
            Pf = P @ F.T
            k = int(np.argmin([_one_way_chamfer(Pf, t) for t in ens]))
            M = refine_to_reference(Pf, ens[k], pl[k])
            R = M @ R
            tweak = math.degrees(math.acos(float(np.clip((np.trace(M) - 1) / 2, -1, 1))))
        if "recheck_symmetry" in policy:
            # A reference can swap the initial axes. Test symmetry about the
            # FINAL up axis, rather than carrying a stale C2z through a swap.
            info = dict(info)
            info["symmetry"], info["sym_scores"] = detect_axial_symmetry(
                Shape(X), R[2], **policy["recheck_symmetry"])
        info = dict(info, cluster=int(j), snapped=bool(not np.allclose(F, np.eye(3))),
                    refined_deg=tweak, reference_locked_axes=locked)
        return R, info

    return canon


def generic_frame(shape):
    """Fallback when the label is unknown: strongest mirror plane, support cue
    for up, longest remaining direction for forward."""
    X = shape.X
    n, mscore = best_mirror(shape)
    up, _ = _circle_argmax(X, n, lambda d: support_score(X, d))
    fwd = unit(np.cross(n, up))
    lo, hi = end_spread(X, fwd)
    if lo < hi:
        fwd = -fwd
    return frame_from(fwd, up), {"mirror_score": mscore, "symmetry": "I"}


# Only these geometric operations may appear in rule data.
_SIGNS = tuple(UP_SIGNS)
OPERATIONS = {
    "bilateral": Operation(bilateral_frame, choices={
        "lateral": ("mirror", "principal"),
        "up_axis": ("plate", "plate_or_pca", "support"),
        "up_sign": (*_SIGNS, "taper", "upper_structure"),
        "forward": ("crown", "crown_vote", "roof_offset")}),
    "upright": Operation(upright_frame, choices={
        "up_axis": ("plate", "principal", "sharpest", "tallest"), "up_sign": _SIGNS}),
    "revolution": Operation(revolution_frame),
    "slab": Operation(slab_frame, choices={"up_sign": _SIGNS}),
    "principal": Operation(principal_frame),
    "winged": Operation(winged_frame),
    "platform": Operation(platform_frame),
    "open_basin": Operation(open_basin_frame),
    "supported_case": Operation(supported_case_frame),
    "profile_box": Operation(profile_box_frame),
    "generic": Operation(generic_frame),
    "crown_forward": Operation(crown_forward, modifier=True),
    "axial_symmetry": Operation(axial_symmetry, modifier=True,
                                choices={"allow": ("C2z", "C4z", "Cinfz")}),
    "panel_symmetry": Operation(panel_symmetry, modifier=True),
    "end_closure": Operation(end_closure, modifier=True),
}
DEFAULT_RULES_PATH = Path(__file__).with_name("rules.json")


def load_rule_database(path=DEFAULT_RULES_PATH):
    """Load an independent validated snapshot, without changing active rules."""
    return RuleDatabase.load(path, OPERATIONS)


def configure_rules(path=DEFAULT_RULES_PATH):
    """Activate a validated store before loading data/references or serving work.

    For concurrent experiments use load_rule_database + make_canonicaliser's
    rule_database argument instead of changing the process default.
    """
    database = load_rule_database(path)  # validate everything before mutation
    global RULE_DATABASE, RULES, CLASS_ORDER, SYNSET_TO_CLASS, DEFAULT_SYMMETRY
    global REFERENCE_CLASSES, REFINE_CLASSES, CLUSTERS_PER_CLASS
    RULE_DATABASE = database
    RULES = database.rules
    CLASS_ORDER = list(database.classes)
    SYNSET_TO_CLASS = dict(database.synsets)
    DEFAULT_SYMMETRY = {c: r["symmetry"] for c, r in database.classes.items()}
    REFERENCE_CLASSES = tuple(c for c in database.classes if database.reference_policy(c).get("enabled", False))
    REFINE_CLASSES = tuple(c for c in database.classes if database.reference_policy(c).get("refine", False))
    CLUSTERS_PER_CLASS = {c: database.reference_policy(c)["clusters"] for c in database.classes
                          if "clusters" in database.reference_policy(c)}
    return database


configure_rules()


PCA_FIRST = True        # see canonicalise(); cleared by --no-pca-first


def canonical_class_name(cls):
    """Normalise common spelling variants before rule/reference dispatch."""
    return RULE_DATABASE.canonical_name(cls)


def canonicalise(X, cls, *, rule_database=None):
    """Full pipeline for one cloud: label selects the rule, the rule returns a
    canonical rotation and the symmetry group that rotation is defined up to.

    With PCA_FIRST the cloud is first put into its principal-axis frame and the
    rule is applied to *that*.  Every pose of a given cloud collapses to the
    same principal frame, so the rule sees one fixed input and cannot answer
    differently: the composite is exactly pose-invariant and stability is zero
    by construction rather than by measurement.  The canonical pose is still
    whatever the rule decides -- PCA only removes the arbitrary input attitude
    before the rule looks at it, it does not get a vote on the answer.

    Worth being clear about what this does and does not buy.  It fixes the
    output against *rotation* of the input.  It does not fix it against the
    points themselves moving: principal axes are ill-conditioned when two
    eigenvalues are close, which is the normal state of a bowl or a square
    table, so resampling or noise still moves the frame.  That shows up under
    --robustness, not here.  It also does not make the rules any more correct;
    an instance whose cue sits near its decision boundary now falls the same
    side every time instead of wavering, so consistency stops drifting between
    runs at whatever value it already had."""
    database = RULE_DATABASE if rule_database is None else rule_database
    cls = database.canonical_name(cls)
    pre = np.eye(3)
    if PCA_FIRST:
        pre, _ = pca_baseline(X)
        X = normalise_cloud(X)[0] @ pre.T
    shape = Shape(X)
    rule = database.rules.get(cls, database.fallback)
    try:
        R, info = rule(shape)
    except Exception as exc:                      # never lose a whole run to one cloud
        R, info = pca_baseline(X)
        info = dict(info)
        info["fallback"] = f"{type(exc).__name__}: {exc}"
    if not is_rotation(R):                        # numerical guard
        U, _, Vt = np.linalg.svd(R)
        R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    info.setdefault("symmetry", database.classes.get(cls, {}).get("symmetry", "I"))
    info["ambiguous_axes"] = [name for name in ("forward", "up")
                              if info.get(name + "_confidence", 1.0) < 0.15]
    return R @ pre, info


def pca_baseline(X, cls=None):
    """Reference canonicaliser: principal axes with third-moment sign fixing.
    Included so the class-conditional rules have something to be compared to."""
    Y, _, _ = normalise_cloud(X)
    w, V = principal_axes(Y)
    A = V.T.copy()
    for i in range(3):
        if float(((Y @ A[i]) ** 3).mean()) < 0:
            A[i] *= -1
    if np.linalg.det(A) < 0:
        A[2] *= -1
    return A, {"symmetry": "I"}


# ----------------------------------------------------------------------------
# Symmetry groups and rotation metrics
#
# If S is a rotational self-map of the canonical shape then R and S R produce
# the same canonical cloud, so distances are taken modulo left multiplication
# by the group.  For a surface of revolution the minimisation over the group is
# available in closed form, so no sampling error enters the metric.
# ----------------------------------------------------------------------------

def _rz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


FINITE_GROUPS = {
    "I": [np.eye(3)],
    # 180 deg about canonical x swaps up with down and left with right.  For a
    # shelf unit that is a plain open box -- a flat panel top and bottom, alike
    # -- that is a genuine self-map, so the frame is only ever determined up to
    # it and the metrics must not charge for it.
    "C2x": [np.eye(3), _rx(np.pi)],
    "C2y": [np.eye(3), np.diag([-1., 1., -1.])],
    "D2": [np.eye(3), np.diag([1., -1., -1.]),
           np.diag([-1., 1., -1.]), np.diag([-1., -1., 1.])],
    "C2z": [np.eye(3), _rz(np.pi)],
    "C3z": [_rz(2 * k * np.pi / 3) for k in range(3)],
    "C4z": [_rz(k * np.pi / 2) for k in range(4)],
    "C6z": [_rz(k * np.pi / 3) for k in range(6)],
}


def align_in_group(R, M, group):
    """The group element g minimising the angle between g R and M, applied."""
    C = R @ M.T
    if group == "Cinfz":
        theta = math.atan2(C[0, 1] - C[1, 0], C[0, 0] + C[1, 1])
        return _rz(theta) @ R
    best, best_tr = R, -np.inf
    for g in FINITE_GROUPS.get(group, FINITE_GROUPS["I"]):
        tr = np.trace(g @ C)
        if tr > best_tr:
            best_tr, best = tr, g @ R
    return best


def geodesic_deg(A, B):
    c = (np.trace(A @ B.T) - 1.0) / 2.0
    return float(math.degrees(math.acos(float(np.clip(c, -1.0, 1.0)))))


def group_distance_deg(A, B, group):
    """Geodesic distance on SO(3) modulo the symmetry group."""
    if group == "Cinfz":
        C = A @ B.T
        tr = math.hypot(C[0, 0] + C[1, 1], C[0, 1] - C[1, 0]) + C[2, 2]
        return float(math.degrees(math.acos(float(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))))
    return min(geodesic_deg(g @ A, B) for g in FINITE_GROUPS.get(group, FINITE_GROUPS["I"]))


def project_so3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    return U @ D @ Vt


def dispersion_deg(rotations, groups, iters=15):
    """RMS angular deviation from the symmetry-aware chordal mean, in degrees.

    With the trivial group this is exactly the usual chordal-mean dispersion;
    with a non-trivial group each sample is first moved to its representative
    closest to the current mean, which is what stops a bowl from being charged
    for a rotation about its own axis."""
    rotations = [np.asarray(R, float) for R in rotations]
    if isinstance(groups, str):
        groups = [groups] * len(rotations)
    if len(rotations) == 1:
        return 0.0, np.zeros(1), rotations[0]
    M = rotations[0].copy()
    for _ in range(iters):
        aligned = [align_in_group(R, M, g) for R, g in zip(rotations, groups)]
        M_new = project_so3(np.mean(aligned, axis=0))
        if geodesic_deg(M_new, M) < 1e-7:
            M = M_new
            break
        M = M_new
    angles = np.array([group_distance_deg(R, M, g) for R, g in zip(rotations, groups)])
    return float(np.sqrt(np.mean(angles ** 2))), angles, M


# ----------------------------------------------------------------------------
# Rotation-invariant descriptors and a training-free label step
# ----------------------------------------------------------------------------

def invariant_features(shape):
    """Semantic descriptors that do not change when the cloud is rotated: every
    one is either a spectral quantity or a maximum over all directions."""
    X, w = shape.X, shape.w
    w = np.maximum(w, EPS)
    mirror = float(shape.mirror_score(shape.cands).max())
    rot = float(max(shape.rot_score(d) for d in shape.cands))
    plate = max(slab_peak(X, d, width=0.07, ends_only=True)[0] for d in shape.cands)
    mid_plate = max(slab_peak(X, d, width=0.07, ends_only=False)[0] for d in shape.cands)
    supp = max(support_score(X, d) for d in np.vstack([shape.cands, -shape.cands]))
    r = np.sqrt((X ** 2).sum(1))
    return np.array([
        w[1] / w[0], w[2] / w[0], w[2] / max(w[1], EPS),
        mirror, rot, plate, mid_plate, min(supp, 12.0),
        float(r.std() / (r.mean() + EPS)),
        float(((r - r.mean()) ** 3).mean() / (r.std() ** 3 + EPS)),
    ], float)


FEATURE_NAMES = ["lam2/lam1", "lam3/lam1", "lam3/lam2", "mirror", "revolution",
                 "end_plate", "any_plate", "support", "radial_cv", "radial_skew"]


def classify_leave_one_out(features, labels):
    """Nearest class-prototype in standardised feature space, prototypes being
    medians of the other instances.  No fitted parameters are carried over from
    anywhere: the prototypes are summary statistics of the data at hand."""
    F = np.asarray(features, float)
    F = (F - F.mean(0)) / (F.std(0) + EPS)
    labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    pred = []
    for i in range(len(F)):
        best, best_d = None, np.inf
        for c in classes:
            m = (labels == c)
            m[i] = False
            if not m.any():
                continue
            d = float(np.abs(F[i] - np.median(F[m], axis=0)).sum())
            if d < best_d:
                best_d, best = d, c
        pred.append(best)
    return np.array(pred)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def stability_of_instance(X, cls, canon, k, rng, return_info=False):
    """Equivariance test.  The estimator should satisfy f(X A^T) = f(X) A^T, so
    R_k A_k is the same rotation for every random A_k when the estimator is
    perfectly stable.  Nothing is cached between rotations: every canonical
    frame is recomputed from the rotated points."""
    mats = [np.eye(3)] + random_rotations(k, rng)
    frames, groups, diagnostics = [], [], []
    for A in mats:
        R, info = canon(X @ A.T, cls)
        frames.append(R @ A)
        groups.append(info.get("symmetry", "I"))
        diagnostics.append(info)
    grp_free = dispersion_deg(frames, "I")[0]
    grp_quot = dispersion_deg(frames, groups)[0]
    # The frame handed to the consistency metric is the one recovered from a
    # RANDOMLY ROTATED copy, with the known rotation undone -- not the one from
    # the pose the dataset happened to store.  Measuring consistency on the
    # stored pose feeds every rule the same easy input every run, so a cue
    # sitting near its decision boundary falls the same way each time.  A random
    # pose samples around that boundary and is what the object would arrive as
    # in use.
    i = 1 if len(frames) > 1 else 0
    result = (grp_free, grp_quot, frames[i], groups[i])
    return result + (diagnostics[i],) if return_info else result


def evaluate(dataset, canon, k_rot, seed, use_pred_labels=False, want_features=True,
             baseline=False):
    results, feats, labels = {}, [], []
    for cls in [c for c in CLASS_ORDER if c in dataset]:
        clouds, ids = dataset[cls]
        rng = np.random.default_rng(seed + 977 * (CLASS_ORDER.index(cls)
                                                      if cls in CLASS_ORDER else 7))
        t0 = time.time()

        stab_raw, stab_sym, frames, groups, kept, diagnostics = [], [], [], [], [], []
        for idx, X in enumerate(clouds):
            try:
                a, b, R0, g0, info = stability_of_instance(
                    X, cls, canon, k_rot, rng, return_info=True)
            except Exception as exc:
                print(f"  [warn] {cls} instance {ids[idx]} skipped: {exc}")
                continue
            stab_raw.append(a)
            stab_sym.append(b)
            frames.append(R0)
            groups.append(g0)
            kept.append(ids[idx])
            diagnostics.append(info)
        if not frames:
            print(f"  [warn] no usable instances for {cls}")
            continue
        ids = kept

        cons_raw = dispersion_deg(frames, "I")
        cons_sym = dispersion_deg(frames, groups)

        if want_features and not baseline:
            for X in clouds:
                try:
                    feats.append(invariant_features(Shape(X)))
                    labels.append(cls)
                except Exception:
                    pass

        results[cls] = {
            "n_instances": len(frames),
            "stability_raw_deg": float(np.mean(stab_raw)),
            "stability_sym_deg": float(np.mean(stab_sym)),
            "stability_sym_median_deg": float(np.median(stab_sym)),
            "consistency_raw_deg": float(cons_raw[0]),
            "consistency_sym_deg": float(cons_sym[0]),
            "consistency_sym_median_deg": float(np.median(cons_sym[1])),
            "consistency_within10": float(np.mean(np.asarray(cons_sym[1]) < 10.0)),
            "stability_within10": float(np.mean(np.asarray(stab_sym) < 10.0)),
            "per_instance_consistency_deg": cons_sym[1].tolist(),
            "per_instance_rule_info": diagnostics,
            "fallback_count": sum("fallback" in info for info in diagnostics),
            "symmetry_groups": groups,
            "mean_frame": cons_sym[2].tolist(),
            "seconds": time.time() - t0,
            "ids": ids,
        }
        print(f"  {cls:9s} n={len(frames):3d}  "
              f"stability {np.mean(stab_sym):7.3f} deg   "
              f"consistency {cons_sym[0]:7.3f} deg   "
              f"({time.time() - t0:.1f}s)")
    return results, (np.array(feats) if feats else None), labels


def ground_truth_error(dataset, canon):
    """Only meaningful for procedural shapes, which are built in the canonical
    pose: the estimated frame should then be the identity, up to symmetry."""
    out = {}
    for cls in [c for c in CLASS_ORDER if c in dataset]:
        errs = []
        for X in dataset[cls][0]:
            try:
                R, info = canon(X, cls)
            except Exception:
                continue
            errs.append(group_distance_deg(R, np.eye(3), info.get("symmetry", "I")))
        if not errs:
            continue
        out[cls] = {"mean_deg": float(np.mean(errs)),
                    "median_deg": float(np.median(errs)),
                    "frac_within_10deg": float(np.mean(np.array(errs) < 10.0))}
    return out


# ----------------------------------------------------------------------------
# Robustness of the canonical frame to input degradation
#
# Stability answers "does the frame follow the object when the object turns".
# Robustness answers "does the frame survive a cloud that is noisy, sparse, or
# incomplete".  A canonicaliser can be perfectly equivariant and still useless
# if 1% jitter moves the frame by 40 degrees, so both are reported.
# ----------------------------------------------------------------------------

PERTURBATIONS = ("noise 1%", "noise 2%", "decimate 50%", "crop 15%")


def perturb_cloud(X, kind, rng):
    """Degrade a cloud.  Noise is expressed as a fraction of the RMS radius, so
    the level means the same thing whatever the object's absolute size is."""
    Y = X - X.mean(axis=0)
    rms = math.sqrt(float((Y ** 2).sum(axis=1).mean()))
    if kind.startswith("noise"):
        frac = float(kind.split()[1].rstrip("%")) / 100.0
        return X + rng.normal(scale=frac * rms, size=X.shape)
    if kind.startswith("decimate"):
        frac = float(kind.split()[1].rstrip("%")) / 100.0
        keep = max(24, int(round(len(X) * frac)))
        return X[rng.choice(len(X), keep, replace=False)]
    if kind.startswith("crop"):
        frac = float(kind.split()[1].rstrip("%")) / 100.0
        d = rng.normal(size=3)
        t = Y @ (d / np.linalg.norm(d))
        return X[t > np.quantile(t, frac)]          # slice off one side
    raise ValueError(kind)


def robustness(dataset, canon, seed, kinds=PERTURBATIONS):
    """Frame shift caused by each degradation, modulo the symmetry group.

    The degraded cloud is also randomly rotated, so the number is the error a
    downstream network would actually see: canonicalise a clean cloud, then
    canonicalise a damaged copy in an unrelated pose, and compare the two
    frames after undoing the known rotation."""
    out = {}
    for cls in [c for c in CLASS_ORDER if c in dataset]:
        rng = np.random.default_rng(seed + 5099 * (CLASS_ORDER.index(cls) + 1))
        per_kind = {k: [] for k in kinds}
        for X in dataset[cls][0]:
            try:
                R0, info = canon(X, cls)
            except Exception:
                continue
            g = info.get("symmetry", "I")
            for k in kinds:
                try:
                    Y = perturb_cloud(X, k, rng)
                    A = random_rotations(1, rng)[0]
                    R, _ = canon(Y @ A.T, cls)
                    per_kind[k].append(group_distance_deg(R @ A, R0, g))
                except Exception:
                    continue
        if any(per_kind.values()):
            out[cls] = {k: {"mean_deg": float(np.mean(v)),
                            "median_deg": float(np.median(v)),
                            "frac_within_10deg": float(np.mean(np.array(v) < 10.0))}
                        for k, v in per_kind.items() if v}
    return out


def print_robustness(rob, kinds=PERTURBATIONS):
    if not rob:
        return
    print("\nROBUSTNESS: frame shift under input degradation, modulo symmetry")
    print("-" * 88)
    print(f"{'class':10s}" + "".join(f"{k:>19s}" for k in kinds))
    print(f"{'':10s}" + "".join(f"{'median / <10 deg':>19s}" for _ in kinds))
    print("-" * 88)
    for cls, r in rob.items():
        print(f"{cls:10s}" + "".join(
            f"{r[k]['median_deg']:>11.2f} /{100 * r[k]['frac_within_10deg']:4.0f}%"
            if k in r else f"{'-':>19s}" for k in kinds))
    print("-" * 88)
    print("each degraded cloud is also randomly rotated.  The two numbers matter")
    print("separately: cropping is bimodal, so a small median can hide a minority")
    print("of clouds where the cut removed the feature the rule depends on")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Is the dataset pre-aligned?  A diagnostic, not a metric.
#
# Consistency is the spread of the predicted rotations across instances, and it
# only means "do the rules agree" when every instance arrives in the same pose.
# If the stored clouds are already in arbitrary poses then each R must undo a
# different rotation, the spread is large whatever the rules do, and the number
# says nothing.  Chamfer distance between whole clouds does not care about that,
# so comparing three levels of it separates the two cases.
# ----------------------------------------------------------------------------

def chamfer(A, B):
    ta, tb = cKDTree(A), cKDTree(B)
    da, _ = tb.query(A, workers=-1)
    db, _ = ta.query(B, workers=-1)
    return 0.5 * (float(da.mean()) + float(db.mean()))


def mean_pair_chamfer(clouds, n_pairs, rng):
    n = len(clouds)
    if n < 2:
        return float("nan")
    out = []
    for _ in range(n_pairs):
        i, j = rng.choice(n, 2, replace=False)
        out.append(chamfer(clouds[i], clouds[j]))
    return float(np.mean(out))


def diagnose_alignment(dataset, canon, seed, n_pairs=40):
    """Per class, mean Chamfer between pairs of clouds at three stages:

      stored     as they sit in the dataset
      scrambled  each cloud given its own random rotation -- the level that
                 means 'these are definitely not mutually aligned'
      canonical  after the pipeline's rotation

    stored close to scrambled  -> the dataset is not pre-aligned
    canonical well below both  -> the rules are aligning the class
    """
    out = {}
    for cls in [c for c in CLASS_ORDER if c in dataset]:
        rng = np.random.default_rng(seed + 811 * (CLASS_ORDER.index(cls) + 1))
        stored, scrambled, canonical = [], [], []
        for X in dataset[cls][0]:
            Y = normalise_cloud(X)[0]
            stored.append(Y)
            scrambled.append(Y @ random_rotations(1, rng)[0].T)
            try:
                R, _ = canon(X, cls)
                canonical.append(Y @ R.T)
            except Exception:
                pass
        if len(stored) < 2:
            continue
        out[cls] = {
            "stored": mean_pair_chamfer(stored, n_pairs, np.random.default_rng(seed)),
            "scrambled": mean_pair_chamfer(scrambled, n_pairs, np.random.default_rng(seed)),
            "canonical": mean_pair_chamfer(canonical, n_pairs, np.random.default_rng(seed)),
        }
    return out


def print_diagnosis(diag):
    if not diag:
        return
    print("\nDATASET ALIGNMENT CHECK (mean Chamfer between pairs of clouds)")
    print("-" * 78)
    print(f"{'class':10s}{'stored':>12s}{'scrambled':>12s}{'canonical':>12s}   verdict")
    print("-" * 78)
    for cls, d in diag.items():
        s, r, c = d["stored"], d["scrambled"], d["canonical"]
        aligned = s < 0.65 * r
        helped = c < 0.85 * s
        if aligned and c <= 1.15 * s:
            v = "pre-aligned; rules keep it"
        elif aligned:
            v = "pre-aligned; RULES BREAK IT"
        elif helped:
            v = "not pre-aligned; rules align it"
        else:
            v = "not pre-aligned; rules do not align it"
        print(f"{cls:10s}{s:12.4f}{r:12.4f}{c:12.4f}   {v}")
    print("-" * 78)
    print("consistency is only meaningful when 'stored' is well below 'scrambled'.")
    print("if it is not, the instances were saved in arbitrary poses and the")
    print("spread of the predicted rotations measures the data, not the method.")


# ----------------------------------------------------------------------------
# Rendering
#
# A raw 3-D scatter of 1024 points reads as a cloud of dots, which makes it hard
# to see whether two canonical frames actually agree.  These helpers project the
# cloud themselves, shade each point by its estimated surface normal and draw
# the splats back to front, so the object reads as a solid surface.
# ----------------------------------------------------------------------------

RENDER_BASE = np.array([0.42, 0.48, 0.58])       # slate blue, like a CAD render

# Viewing angle per class, chosen so that the failure each class actually has is
# visible.  Aircraft and cars fail by turning end for end, and a three-quarter
# view hides exactly that, so both are drawn from the side.
CLASS_VIEW = {"airplane": (10.0, -90.0), "car": (8.0, -90.0),
              "chair": (14.0, -62.0), "table": (16.0, -60.0), "bowl": (18.0, -58.0)}


def camera_basis(elev, azim):
    e, a = math.radians(elev), math.radians(azim)
    right = np.array([-math.sin(a), math.cos(a), 0.0])
    up = np.array([-math.sin(e) * math.cos(a), -math.sin(e) * math.sin(a), math.cos(e)])
    fwd = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return right, up, fwd


def surface_normals(X, fwd, k=18, smooth=2):
    """Local-PCA normals, turned towards the camera and then smoothed.

    Turning them towards the camera avoids the speckle a raw orientation gives
    on thin parts such as chair legs, where neighbouring points would otherwise
    disagree about which side is outside."""
    tree = cKDTree(X)
    _, idx = tree.query(X, k=min(k, len(X)), workers=-1)
    N = np.empty_like(X)
    for i, nb in enumerate(np.atleast_2d(idx)):
        _, V = np.linalg.eigh(np.cov(X[nb].T))
        N[i] = V[:, 0]
    N *= np.sign(N @ fwd)[:, None]
    for _ in range(smooth):
        N = N[idx].mean(axis=1)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        N *= np.sign(N @ fwd)[:, None]
    nn, _ = tree.query(X, k=2, workers=-1)
    return N, float(np.median(nn[:, 1]))


def render_cloud(ax, fig, X, elev=18, azim=-58, base=RENDER_BASE, shadow=True,
                 cover=2.1, lim=None):
    """Draw one cloud as a shaded solid.  `cover` is the splat radius in units
    of the point spacing: about 2 closes the surface without erasing detail."""
    right, up, fwd = camera_basis(elev, azim)
    N, spacing = surface_normals(X, fwd)
    u, v, d = X @ right, X @ up, X @ fwd

    L = 0.55 * (-right) + 0.70 * up + 0.75 * fwd
    L /= np.linalg.norm(L)
    H = L + fwd
    H /= np.linalg.norm(H)
    inten = 0.46 + 0.54 * np.clip(N @ L, 0.0, 1.0)
    spec = np.clip(N @ H, 0.0, 1.0) ** 40
    col = np.clip(base[None, :] * inten[:, None] + 0.30 * spec[:, None], 0.0, 1.0)

    r = lim if lim is not None else 1.08 * max(np.abs(u).max(), np.abs(v).max())
    ax.set_xlim(-r, r)
    ax.set_ylim(-r * 0.95, r * 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    w_in = fig.get_size_inches()[0] * ax.get_position().width
    s = math.pi * (cover * spacing * (w_in * 72.0) / (2 * r)) ** 2
    order = np.argsort(d)
    if shadow:
        S = X.copy()
        S[:, 2] = X[:, 2].min() - 0.03
        ax.scatter(S @ right, S @ up, s=s * 1.3, c=[[0.70, 0.72, 0.76]],
                   alpha=0.05, linewidths=0)
    ax.scatter(u[order], v[order], s=s, c=col[order], linewidths=0, marker="o")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def make_figures(dataset, canon, out_dir, seed, n_show=16, n_cols=8, n_rot=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cls in [c for c in CLASS_ORDER if c in dataset]:
        clouds, ids = dataset[cls]
        m = min(n_show, len(clouds))
        elev, azim = CLASS_VIEW.get(cls, (18.0, -58.0))
        rng = np.random.default_rng(seed + 31 * (CLASS_ORDER.index(cls) + 1))

        # --- consistency ------------------------------------------------------
        # Inputs are randomly rotated, matching how consistency is measured: the
        # stored dataset pose is one convenient special case, not what an object
        # looks like when it arrives.
        ins, outs = [], []
        for X in clouds[:m]:
            A = random_rotations(1, rng)[0]
            Xr = normalise_cloud(X @ A.T)[0]
            R, _ = canon(X @ A.T, cls)
            ins.append(Xr)
            outs.append(Xr @ R.T)

        blocks = int(math.ceil(m / n_cols))
        fig, axes = plt.subplots(2 * blocks, n_cols,
                                 figsize=(2.7 * n_cols, 3.0 * 2 * blocks))
        fig.patch.set_facecolor("white")
        axes = np.atleast_2d(axes)
        for k in range(2 * blocks * n_cols):
            axes[k // n_cols, k % n_cols].axis("off")
        for i in range(m):
            b, c = i // n_cols, i % n_cols
            render_cloud(axes[2 * b, c], fig, ins[i], elev=elev, azim=azim, lim=1.7)
            render_cloud(axes[2 * b + 1, c], fig, outs[i], elev=elev, azim=azim, lim=1.7)
            axes[2 * b, c].set_title(str(ids[i])[:12], fontsize=8)
        for b in range(blocks):
            axes[2 * b, 0].text(-0.10, 0.5, "RANDOM INPUT", rotation=90,
                                transform=axes[2 * b, 0].transAxes, va="center",
                                ha="right", fontsize=9, weight="bold")
            axes[2 * b + 1, 0].text(-0.10, 0.5, "CANONICAL", rotation=90,
                                    transform=axes[2 * b + 1, 0].transAxes,
                                    va="center", ha="right", fontsize=9, weight="bold")
        fig.suptitle(f"{cls}: consistency -- {m} instances, each from a random pose",
                     fontsize=13, weight="bold")
        fig.tight_layout(rect=(0.02, 0, 1, 0.96))
        fig.savefig(out_dir / f"{cls}_consistency.png", dpi=110)
        plt.close(fig)
        print(f"  saved {out_dir / f'{cls}_consistency.png'}")

        # --- stability: one instance seen from several poses -------------------
        X = clouds[0]
        fig, axes = plt.subplots(2, n_rot, figsize=(2.9 * n_rot, 6.2))
        fig.patch.set_facecolor("white")
        axes = np.atleast_2d(axes)
        for j, A in enumerate(random_rotations(n_rot, rng)):
            Xr = normalise_cloud(X @ A.T)[0]
            R, _ = canon(X @ A.T, cls)
            render_cloud(axes[0, j], fig, Xr, lim=1.7, elev=elev, azim=azim)
            render_cloud(axes[1, j], fig, Xr @ R.T, lim=1.7, elev=elev, azim=azim)
        axes[0, 0].text(-0.08, 0.5, "ROTATED INPUT", transform=axes[0, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=10, weight="bold")
        axes[1, 0].text(-0.08, 0.5, "CANONICAL", transform=axes[1, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=10, weight="bold")
        fig.suptitle(f"{cls}: stability -- one instance, {n_rot} random input poses",
                     fontsize=13, weight="bold")
        fig.tight_layout(rect=(0.02, 0, 1, 0.95))
        fig.savefig(out_dir / f"{cls}_stability.png", dpi=110)
        plt.close(fig)
        print(f"  saved {out_dir / f'{cls}_stability.png'}")


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def print_table(results, title):
    print(f"\n{title}")
    print("-" * 78)
    print(f"{'class':10s} {'n':>4s} {'stab raw':>9s} {'stab sym':>9s}"
          f" {'cons raw':>9s} {'cons sym':>9s} {'cons med':>9s} {'<10deg':>7s} {'group':>7s}")
    print("-" * 86)
    for cls, r in results.items():
        grp = max(set(r["symmetry_groups"]), key=r["symmetry_groups"].count)
        print(f"{cls:10s} {r['n_instances']:4d} {r['stability_raw_deg']:9.3f} "
              f"{r['stability_sym_deg']:9.3f} "
              f"{r['consistency_raw_deg']:9.3f} {r['consistency_sym_deg']:9.3f} "
              f"{r['consistency_sym_median_deg']:9.3f} "
              f"{100 * r.get('consistency_within10', float('nan')):6.0f}% {grp:>7s}")
    print("-" * 86)
    print(f"{'MEAN':10s} {'':4s} "
          f"{np.mean([r['stability_raw_deg'] for r in results.values()]):9.3f} "
          f"{np.mean([r['stability_sym_deg'] for r in results.values()]):9.3f} "
          f"{np.mean([r['consistency_raw_deg'] for r in results.values()]):9.3f} "
          f"{np.mean([r['consistency_sym_deg'] for r in results.values()]):9.3f} "
          f"{np.mean([r['consistency_sym_median_deg'] for r in results.values()]):9.3f} "
          f"{100 * np.mean([r.get('consistency_within10', 0.0) for r in results.values()]):6.0f}%")
    print("sym = modulo the detected symmetry group.  RMS squares the errors, so one")
    print("inverted instance in 25 shows up as 36 degrees: read the median too.")


def write_latex(path, results, baseline, gt, rob, meta):
    """booktabs tables ready to \\input into the capstone report."""
    def grp(r):
        g = max(set(r["symmetry_groups"]), key=r["symmetry_groups"].count)
        return {"I": "$\\mathbb{1}$", "C2z": "$C_2$", "C3z": "$C_3$",
                "C4z": "$C_4$", "C6z": "$C_6$", "Cinfz": "$C_\\infty$"}.get(g, g)

    L = ["% generated by geo_canon.py -- " + json.dumps(meta),
         "\\begin{table}[htbp]", "  \\centering",
         "  \\caption{Stability and consistency of the canonical frame, in degrees. "
         "Raw columns are the plain SO(3) dispersion; symmetry columns quotient out "
         "the detected rotational self-symmetry group.}",
         "  \\label{tab:canon-metrics}",
         "  \\begin{tabular}{lrrrrrc}", "    \\toprule",
         "    Class & $n$ & \\multicolumn{2}{c}{Stability} & "
         "\\multicolumn{2}{c}{Consistency} & Group \\\\",
         "    \\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
         "     & & raw & sym. & raw & sym. & \\\\", "    \\midrule"]
    for cls, r in results.items():
        L.append(f"    {cls.capitalize()} & {r['n_instances']} & "
                 f"{r['stability_raw_deg']:.2f} & {r['stability_sym_deg']:.2f} & "
                 f"{r['consistency_raw_deg']:.2f} & {r['consistency_sym_deg']:.2f} & "
                 f"{grp(r)} \\\\")
    if results:
        L += ["    \\midrule",
              f"    Mean & & {np.mean([r['stability_raw_deg'] for r in results.values()]):.2f} & "
              f"{np.mean([r['stability_sym_deg'] for r in results.values()]):.2f} & "
              f"{np.mean([r['consistency_raw_deg'] for r in results.values()]):.2f} & "
              f"{np.mean([r['consistency_sym_deg'] for r in results.values()]):.2f} & \\\\"]
    if baseline:
        L += ["    \\midrule",
              "    \\multicolumn{7}{l}{\\emph{PCA baseline}} \\\\"]
        for cls, r in baseline.items():
            L.append(f"    {cls.capitalize()} & {r['n_instances']} & "
                     f"{r['stability_raw_deg']:.2f} & {r['stability_sym_deg']:.2f} & "
                     f"{r['consistency_raw_deg']:.2f} & {r['consistency_sym_deg']:.2f} & "
                     f"{grp(r)} \\\\")
    L += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]

    if rob:
        kinds = list(next(iter(rob.values())).keys())
        L += ["\\begin{table}[htbp]", "  \\centering",
              "  \\caption{Median shift of the canonical frame under input "
              "degradation, in degrees, modulo symmetry.}",
              "  \\label{tab:canon-robustness}",
              "  \\begin{tabular}{l" + "r" * len(kinds) + "}", "    \\toprule",
              "    Class & " + " & ".join(k.replace("%", "\\%") for k in kinds) + " \\\\",
              "    \\midrule"]
        for cls, r in rob.items():
            L.append(f"    {cls.capitalize()} & " + " & ".join(
                f"{r[k]['median_deg']:.2f}" if k in r else "--" for k in kinds) + " \\\\")
        L += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]

    if gt:
        L += ["\\begin{table}[htbp]", "  \\centering",
              "  \\caption{Error against the known canonical pose of the "
              "procedural shapes, in degrees, modulo symmetry.}",
              "  \\label{tab:canon-pose-error}",
              "  \\begin{tabular}{lrrr}", "    \\toprule",
              "    Class & Mean & Median & Within $10^\\circ$ \\\\", "    \\midrule"]
        for cls, g in gt.items():
            L.append(f"    {cls.capitalize()} & {g['mean_deg']:.2f} & "
                     f"{g['median_deg']:.2f} & {100 * g['frac_within_10deg']:.0f}\\% \\\\")
        L += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]

    Path(path).write_text("\n".join(L), encoding="utf-8")
    print(f"latex tables written to {Path(path).resolve()}")


def write_report(path, results, baseline, gt, cls_report, meta, rob=None,
                 kinds=PERTURBATIONS):
    lines = ["Geometric canonicalisation: stability and consistency", "=" * 78, ""]
    lines.append("settings: " + json.dumps(meta))
    lines.append("")
    lines.append("All figures are degrees; lower is better.")
    lines.append("stability   : RMS spread of R_k A_k over random input rotations A_k")
    lines.append("consistency : RMS spread of the canonical frames across instances")
    lines.append("'sym' columns quotient out the object's rotational symmetry group.")
    lines.append("")
    for name, res in (("CLASS-CONDITIONAL GEOMETRIC RULES", results),
                      ("PCA BASELINE", baseline)):
        if not res:
            continue
        lines.append(name)
        lines.append("-" * 78)
        lines.append(f"{'class':10s} {'n':>4s} {'stab sym':>10s} {'stab med':>10s} "
                     f"{'cons sym':>10s} {'cons med':>10s} {'<10deg':>8s} {'group':>8s}")
        for cls, r in res.items():
            grp = max(set(r["symmetry_groups"]), key=r["symmetry_groups"].count)
            lines.append(f"{cls:10s} {r['n_instances']:4d} {r['stability_sym_deg']:10.3f} "
                         f"{r['stability_sym_median_deg']:10.3f} "
                         f"{r['consistency_sym_deg']:10.3f} "
                         f"{r['consistency_sym_median_deg']:10.3f} "
                         f"{100 * r.get('consistency_within10', 0.0):7.0f}% {grp:>8s}")
        lines.append("")
    if gt:
        lines.append("ERROR AGAINST THE KNOWN POSE (procedural shapes only)")
        lines.append("-" * 78)
        for cls, g in gt.items():
            lines.append(f"{cls:10s} mean {g['mean_deg']:8.3f}  median {g['median_deg']:8.3f}"
                         f"  within 10 deg: {100 * g['frac_within_10deg']:5.1f}%")
        lines.append("")
    if rob:
        lines.append("ROBUSTNESS: frame shift under input degradation "
                     "(modulo symmetry)")
        lines.append("-" * 78)
        lines.append(f"{'class':10s}" + "".join(f"{k + ' (med/<10)':>19s}" for k in kinds))
        for cls, r in rob.items():
            lines.append(f"{cls:10s}" + "".join(
                f"{r[k]['median_deg']:>11.2f} /{100 * r[k]['frac_within_10deg']:4.0f}%"
                if k in r else f"{'-':>19s}" for k in kinds))
        lines.append("")
    if cls_report:
        lines.append("LABEL STEP (nearest prototype on rotation-invariant features)")
        lines.append("-" * 78)
        lines.append(cls_report)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport written to {Path(path).resolve()}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    global PARTIAL_MIRROR, PCA_FIRST, REFERENCE_CLASSES, REFINE_CLASSES
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="processed_data", help="root with one folder per class")
    ap.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH,
                    help="JSON rule database (validated before processing clouds)")
    ap.add_argument("--synthetic", action="store_true", help="force procedural shapes")
    ap.add_argument("--instances", type=int, default=25)
    ap.add_argument("--points", type=int, default=1024)
    ap.add_argument("--mesh-points", type=int, default=MESH_POINTS,
                    help="points taken off each .obj/.ply geometry (stored "
                         "clouds keep --points)")
    ap.add_argument("--rotations", type=int, default=8, help="random rotations per instance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-reference", action="store_true",
                    help="rules only, without the per-class reference attitude")
    ap.add_argument("--ref-instances", type=int, default=10,
                    help="instances used to build each class reference")
    ap.add_argument("--save-reference", default=None,
                    help="write the built references to this .npz for reuse")
    ap.add_argument("--load-reference", default=None,
                    help="reuse references from a .npz instead of building them")
    ap.add_argument("--clusters", type=int, default=0,
                    help="shape clusters per class reference; 0 uses the "
                         "per-class defaults in CLUSTERS_PER_CLASS")
    ap.add_argument("--no-pca-first", action="store_true",
                    help="skip the principal-axis pre-canonicalisation; the "
                         "rules then see the raw input pose and stability "
                         "measures the rules themselves")
    ap.add_argument("--reference-classes", default=None,
                    help="comma-separated classes that get a reference attitude "
                         "('none' to disable)")
    ap.add_argument("--no-refine", action="store_true",
                    help="snap axes to the reference but skip the small ICP tweak")
    ap.add_argument("--baseline", action="store_true", help="also run the PCA baseline")
    ap.add_argument("--classifier", action="store_true", help="also run the label step")
    ap.add_argument("--robustness", action="store_true",
                    help="also measure noise, decimation and cropping")
    ap.add_argument("--diagnose", action="store_true",
                    help="also run the dataset alignment check")
    ap.add_argument("--partial-mirror", action="store_true",
                    help="allow the mirror plane to leave the centroid "
                         "(for heavily occluded or single-view scans)")
    ap.add_argument("--latex", action="store_true",
                    help="also write booktabs tables for the report")
    ap.add_argument("--out", default=".", help="where to write report, json and figures")
    args = ap.parse_args()

    configure_rules(args.rules)
    PARTIAL_MIRROR = args.partial_mirror
    PCA_FIRST = not args.no_pca_first
    if PCA_FIRST:
        print("PCA pre-canonicalisation on: the pipeline is pose-invariant by "
              "construction, so stability\nmeasures only the conditioning of the "
              "principal axes, not the rules.  Read consistency,\nwhich is itself "
              "measured on randomly rotated inputs, and --robustness.")

    t_start = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, synthetic = {}, args.synthetic
    if not args.synthetic:
        root = Path(args.data)
        if root.is_dir():
            print(f"loading point clouds from {root.resolve()}")
            dataset = load_real(root, args.instances, args.points, args.seed,
                                args.mesh_points)
        if not dataset:
            print(f"no usable data under {Path(args.data).resolve()}; "
                  f"falling back to procedural shapes")
            synthetic = True
    if synthetic:
        print("generating procedural shapes (canonical pose known, so the pose "
              "error below is a true accuracy figure)")
        dataset = load_synthetic(args.instances, args.points, args.seed)

    for cls, (clouds, _) in dataset.items():
        print(f"  {cls:9s} {len(clouds):3d} clouds, {clouds[0].shape[0]} points each")

    chosen = (list(REFERENCE_CLASSES) if args.reference_classes is None else
              [canonical_class_name(c) for c in args.reference_classes.split(",") if c.strip()])
    if args.reference_classes is None:
        pass  # keep both reference and refinement selections from the database
    elif chosen and chosen != ["none"]:
        REFERENCE_CLASSES = tuple(chosen)
        REFINE_CLASSES = tuple(chosen)
    else:
        REFERENCE_CLASSES = REFINE_CLASSES = ()

    references = {}
    if args.load_reference:
        references, rmeta = load_references(args.load_reference)
        PCA_FIRST = bool(rmeta.get("pca_first", PCA_FIRST))
        print(f"\nloaded references from {Path(args.load_reference).resolve()} "
              f"({', '.join(references)}; pca_first={PCA_FIRST})")
    elif not args.no_reference:
        print(f"\nbuilding class references for "
              f"{', '.join(REFERENCE_CLASSES)} from {args.ref_instances} instances each")
        for cls, (clouds, _) in dataset.items():
            if cls not in REFERENCE_CLASSES:
                continue
            references[cls] = build_reference(
                clouds, cls, canonicalise, max_ref=args.ref_instances,
                seed=args.seed,
                n_clusters=args.clusters if args.clusters > 0 else None)
    if args.save_reference and references:
        save_references(args.save_reference, references,
                        {"built_from": args.ref_instances})
        print(f"references written to {Path(args.save_reference).resolve()}")

    canon = make_canonicaliser(references, refine=not args.no_refine,
                               refine_classes=REFINE_CLASSES)

    diag = {}
    if args.diagnose:
        diag = diagnose_alignment(dataset, canon, args.seed)
        print_diagnosis(diag)

    print(f"\nevaluating ({args.rotations} random rotations per instance)")
    results, feats, labels = evaluate(dataset, canon, args.rotations, args.seed,
                                      want_features=args.classifier)
    print_table(results, "STABILITY AND CONSISTENCY (degrees)")

    baseline = {}
    if args.baseline:
        print("\nevaluating PCA baseline")
        baseline, _, _ = evaluate(dataset, pca_baseline, args.rotations, args.seed,
                                  want_features=False, baseline=True)
        print_table(baseline, "PCA BASELINE (for comparison)")

    gt = {}
    if synthetic:
        gt = ground_truth_error(dataset, canon)
        print("\nERROR AGAINST THE KNOWN POSE (procedural shapes)")
        print("-" * 60)
        for cls, g in gt.items():
            print(f"{cls:10s} mean {g['mean_deg']:8.3f} deg   median {g['median_deg']:8.3f} deg"
                  f"   within 10 deg: {100 * g['frac_within_10deg']:5.1f}%")

    rob = {}
    if args.robustness:
        print("\nevaluating robustness to noise, decimation and cropping")
        rob = robustness(dataset, canon, args.seed)
        print_robustness(rob)

    cls_report = ""
    if feats is not None and args.classifier and len(set(labels)) > 1:
        pred = classify_leave_one_out(feats, labels)
        labels_arr = np.array(labels)
        acc = float((pred == labels_arr).mean())
        rows = [f"overall accuracy {100 * acc:.1f}%  "
                f"({len(labels_arr)} instances, leave-one-out)"]
        for c in CLASS_ORDER:
            m = labels_arr == c
            if m.any():
                rows.append(f"  {c:10s} {100 * float((pred[m] == c).mean()):5.1f}%")
        cls_report = "\n".join(rows)
        print("\nLABEL STEP (rotation-invariant features, nearest prototype)")
        print("-" * 60)
        print(cls_report)

    if not args.no_figures:
        print("\nfigures")
        make_figures(dataset, canon, out_dir / "figures", args.seed)

    meta = {"source": "synthetic" if synthetic else str(Path(args.data).resolve()),
            "instances": args.instances, "points": args.points,
            "rotations": args.rotations, "seed": args.seed,
            "rules_file": str(args.rules.resolve()),
            "rules_fingerprint": RULE_DATABASE.fingerprint}
    write_report(out_dir / "canonicalisation_report.txt", results, baseline, gt,
                 cls_report, meta, rob)
    (out_dir / "canonicalisation_results.json").write_text(
        json.dumps({"meta": meta, "rules": results, "baseline": baseline,
                    "pose_error": gt, "robustness": rob,
                    "alignment_check": diag}, indent=2), encoding="utf-8")
    if args.latex:
        write_latex(out_dir / "canonicalisation_tables.tex", results, baseline,
                    gt, rob, meta)
    print(f"json written to {(out_dir / 'canonicalisation_results.json').resolve()}")
    print(f"total time {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
