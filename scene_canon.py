#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene-level geometric canonicalisation for indoor point clouds (S3DIS).

Unlike object-level canonicalisation (geo_canon.py), which aligns isolated
shapes by class-specific semantic rules, scene alignment recovers the
*Manhattan frame* — the rotation that puts floors on XY and walls on XZ/YZ —
plus a deterministic horizontal disambiguation so that the same room always
lands in the same pose regardless of input orientation.

No learning, no weights, no .NET runtime.  Pure NumPy + SciPy.

The pipeline:
    1.  Estimate surface normals from the point cloud.
    2.  Extract the dominant three mutually-orthogonal directions from those
        normals (the Manhattan frame) via iterative RANSAC-style voting.
    3.  Assign the most vertical axis as "up" and orient it so +Z is ceiling.
    4.  Disambiguate the two horizontal axes by a deterministic rule:
        the longer bounding-box span maps to X, and the denser end of that
        axis points toward +X.
    5.  Return the 3×3 rotation matrix R such that P_canon = X @ R.T.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Data constants
# ---------------------------------------------------------------------------

SCENE_ROOT_NAME = "scene_s3dis"
AREAS = ["Area_1", "Area_4", "Area_5", "Area_6"]
SCENE_POINTS = 20480          # how many points each .pt file carries

# Normal estimation defaults
NORMAL_K = 30                 # neighbours for normal estimation
NORMAL_RADIUS = 0.15          # hybrid search radius (metres)

# Manhattan-frame extraction
ANGLE_BIN_DEG = 1.0           # histogram resolution for normal voting
ORTHO_TOL_DEG = 12.0          # how close to 90° two axes must be
UP_TOL_DEG = 30.0             # how close to vertical a normal must be to
                               # vote for "floor/ceiling"

# Sub-sampling for the demo (keeps rendering fast)
DEMO_SUBSAMPLE = 8192


# ---------------------------------------------------------------------------
# Data discovery and loading
# ---------------------------------------------------------------------------

def discover_scene_types(root: Path) -> list[str]:
    """Return sorted list of room types found under root/scene_s3dis/."""
    scene_root = root / SCENE_ROOT_NAME
    if not scene_root.is_dir():
        return []
    types = set()
    for area_dir in scene_root.iterdir():
        if not area_dir.is_dir() or area_dir.name not in AREAS:
            continue
        for room_dir in area_dir.iterdir():
            if not room_dir.is_dir():
                continue
            # room_dir.name is like "office_12" → type is "office"
            room_type = room_dir.name.rsplit("_", 1)[0]
            types.add(room_type)
    return sorted(types)


def discover_rooms(root: Path, room_type: Optional[str] = None) -> list[dict]:
    """Return metadata for every room matching the type (or all rooms).

    Each dict has keys: area, room, room_type, path.
    """
    scene_root = root / SCENE_ROOT_NAME
    rooms = []
    for area_name in AREAS:
        area_dir = scene_root / area_name
        if not area_dir.is_dir():
            continue
        for room_dir in sorted(area_dir.iterdir()):
            if not room_dir.is_dir():
                continue
            rtype = room_dir.name.rsplit("_", 1)[0]
            if room_type is not None and rtype != room_type:
                continue
            pt_file = room_dir / f"{room_dir.name}.pt"
            if pt_file.is_file():
                rooms.append({
                    "area": area_name,
                    "room": room_dir.name,
                    "room_type": rtype,
                    "path": pt_file,
                })
    return rooms


def load_scene_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a scene .pt file.  Returns (xyz, rgb) both as float64.

    xyz: (N, 3) world coordinates in metres
    rgb: (N, 3) colours in [0, 1]
    """
    import torch
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(tensor, (np.ndarray,)):
        tensor = tensor.detach().cpu().numpy()
    data = np.asarray(tensor, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(f"expected (N, >=6) tensor, got {data.shape} from {path}")
    xyz = np.ascontiguousarray(data[:, :3])
    rgb = np.clip(data[:, 3:6] / 255.0, 0.0, 1.0)
    return xyz, rgb


def subsample_scene(xyz: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Deterministic random sub-sample of indices."""
    if len(xyz) <= n:
        return np.arange(len(xyz))
    rng = np.random.default_rng(seed)
    return rng.choice(len(xyz), n, replace=False)


# ---------------------------------------------------------------------------
# Normal estimation (pure NumPy/SciPy, no Open3D needed)
# ---------------------------------------------------------------------------

def estimate_normals(xyz: np.ndarray, k: int = NORMAL_K,
                     radius: float = NORMAL_RADIUS) -> np.ndarray:
    """Estimate surface normals via PCA of local neighbourhoods.

    Uses a hybrid search: up to k neighbours within radius.
    Returns (N, 3) unit normals.
    """
    tree = cKDTree(xyz)
    normals = np.zeros_like(xyz)

    # Query k+1 because the point itself is included
    dists, idxs = tree.query(xyz, k=min(k + 1, len(xyz)), workers=-1)

    for i in range(len(xyz)):
        nbr_idx = idxs[i]
        nbr_dist = dists[i]
        # hybrid filter: keep only those within radius (and skip self)
        mask = (nbr_dist > 0) & (nbr_dist <= radius)
        if mask.sum() < 3:
            # fall back: take closest k neighbours regardless of radius
            mask = nbr_dist > 0
        if mask.sum() < 3:
            normals[i] = [0, 0, 1]
            continue
        pts = xyz[nbr_idx[mask]]
        centroid = pts.mean(axis=0)
        cov = (pts - centroid).T @ (pts - centroid) / len(pts)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]  # smallest eigenvalue → normal direction

    # Make normals unit length
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-12)
    normals = normals / lengths
    return normals


def _estimate_normals_fast(xyz: np.ndarray, k: int = NORMAL_K) -> np.ndarray:
    """Faster batch normal estimation — processes all points at once using
    vectorised covariance.  Falls back to the iterative version for very
    large clouds where memory would be an issue."""
    n = len(xyz)
    if n > 100_000:
        # For very large clouds, subsample for speed
        rng = np.random.default_rng(42)
        idx = rng.choice(n, min(n, 50000), replace=False)
        sub_xyz = xyz[idx]
    else:
        sub_xyz = xyz
        idx = np.arange(n)

    tree = cKDTree(sub_xyz)
    _, nbr_idx = tree.query(sub_xyz, k=min(k + 1, len(sub_xyz)), workers=-1)

    normals = np.zeros((len(sub_xyz), 3))
    for i in range(len(sub_xyz)):
        pts = sub_xyz[nbr_idx[i]]
        centroid = pts.mean(axis=0)
        diff = pts - centroid
        cov = diff.T @ diff
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12)

    if n > 100_000:
        # Propagate normals to full cloud via nearest-neighbour
        full_normals = np.zeros((n, 3))
        _, nn_idx = tree.query(xyz, k=1, workers=-1)
        full_normals = normals[nn_idx.ravel()]
        return full_normals

    return normals


# ---------------------------------------------------------------------------
# Manhattan frame extraction
# ---------------------------------------------------------------------------

def _sphere_histogram(normals: np.ndarray, bin_deg: float = ANGLE_BIN_DEG):
    """Vote for dominant normal directions on a sphere histogram.

    Since normals are sign-ambiguous (n and -n describe the same plane),
    we fold everything into the upper hemisphere (z ≥ 0) before binning.
    """
    # Fold into upper hemisphere
    signs = np.sign(normals[:, 2])
    signs[signs == 0] = 1.0
    folded = normals * signs[:, None]

    # Convert to spherical (theta = polar from +z, phi = azimuth)
    theta = np.arccos(np.clip(folded[:, 2], -1, 1))
    phi = np.arctan2(folded[:, 1], folded[:, 0])

    bin_rad = np.deg2rad(bin_deg)
    n_theta = int(np.ceil(np.pi / 2 / bin_rad)) + 1
    n_phi = int(np.ceil(2 * np.pi / bin_rad)) + 1

    theta_idx = np.clip((theta / bin_rad).astype(int), 0, n_theta - 1)
    phi_idx = np.clip(((phi + np.pi) / bin_rad).astype(int), 0, n_phi - 1)

    hist = np.zeros((n_theta, n_phi))
    np.add.at(hist, (theta_idx, phi_idx), 1)

    return hist, bin_rad, n_theta, n_phi, folded


def extract_manhattan_axes(normals: np.ndarray,
                           ortho_tol_deg: float = ORTHO_TOL_DEG
                           ) -> np.ndarray:
    """Extract three mutually orthogonal dominant directions from normals.

    Returns a (3, 3) matrix whose rows are the three Manhattan axes (not yet
    assigned to up/forward/left).
    """
    hist, bin_rad, n_theta, n_phi, folded = _sphere_histogram(normals)

    # Smooth the histogram with a small Gaussian kernel
    from scipy.ndimage import gaussian_filter
    hist_smooth = gaussian_filter(hist, sigma=2.0)

    # Find the dominant direction (highest vote)
    def _peak_direction(h):
        idx = np.unravel_index(np.argmax(h), h.shape)
        theta = idx[0] * bin_rad
        phi = idx[1] * bin_rad - np.pi
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])

    def _suppress_direction(h, d, tol_deg):
        """Zero out bins whose direction is within tol_deg of d or -d."""
        h_copy = h.copy()
        for ti in range(h.shape[0]):
            for pi in range(h.shape[1]):
                theta = ti * bin_rad
                phi = pi * bin_rad - np.pi
                v = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ])
                cos_angle = abs(np.dot(v, d))
                if cos_angle > np.cos(np.deg2rad(tol_deg)):
                    h_copy[ti, pi] = 0
        return h_copy

    # Extract three orthogonal axes
    axis1 = _peak_direction(hist_smooth)
    axis1 = axis1 / np.linalg.norm(axis1)

    hist2 = _suppress_direction(hist_smooth, axis1, ortho_tol_deg * 2)
    axis2 = _peak_direction(hist2)
    # Force orthogonality to axis1
    axis2 = axis2 - np.dot(axis2, axis1) * axis1
    n2 = np.linalg.norm(axis2)
    if n2 < 1e-8:
        # Degenerate: pick an arbitrary orthogonal direction
        arb = np.array([1, 0, 0]) if abs(axis1[0]) < 0.9 else np.array([0, 1, 0])
        axis2 = arb - np.dot(arb, axis1) * axis1
    axis2 = axis2 / np.linalg.norm(axis2)

    # Third axis is the cross product
    axis3 = np.cross(axis1, axis2)
    axis3 = axis3 / np.linalg.norm(axis3)

    return np.vstack([axis1, axis2, axis3])


# ---------------------------------------------------------------------------
# Full scene canonicalisation
# ---------------------------------------------------------------------------

def canonicalise_scene(xyz: np.ndarray,
                       normals: Optional[np.ndarray] = None
                       ) -> tuple[np.ndarray, dict]:
    """Compute a canonical rotation for a room point cloud.

    Parameters
    ----------
    xyz : (N, 3) point cloud in world coordinates.
    normals : (N, 3) surface normals, estimated if None.

    Returns
    -------
    R : (3, 3) rotation matrix.  Canonical coords = xyz @ R.T
    info : dict with diagnostic keys.
    """
    # 1. Centre the cloud
    centroid = xyz.mean(axis=0)
    pts = xyz - centroid

    # 2. Estimate normals if not provided
    if normals is None:
        normals = _estimate_normals_fast(pts)

    # 3. Extract the Manhattan frame
    axes = extract_manhattan_axes(normals)

    # 4. Identify the "up" axis: the one most aligned with vertical
    #    In S3DIS, z is already close to vertical, but we don't assume that
    #    after the cloud has been arbitrarily rotated.
    #    The "up" axis is the one whose normal-votes are most consistent
    #    with floor/ceiling (i.e., most normals point along it).
    votes = np.zeros(3)
    for i in range(3):
        # Count how many normals are roughly parallel to this axis
        dots = np.abs(normals @ axes[i])
        votes[i] = (dots > np.cos(np.deg2rad(UP_TOL_DEG))).sum()

    up_idx = int(np.argmax(votes))
    up = axes[up_idx]

    # Ensure up points toward +z (ceiling).  In most buildings the floor is
    # the low-z region, so "up" should point the way the z extent grows.
    z_proj = pts @ up
    lower_mass = (z_proj < np.median(z_proj)).sum()
    upper_mass = (z_proj >= np.median(z_proj)).sum()
    # The floor has more area → more points.  Floor normals point UP,
    # so we keep up pointing the same way.  But if most mass is at the top
    # in the projected direction, flip.
    if lower_mass < upper_mass:
        up = -up

    # 5. Pick the two horizontal axes
    horiz_indices = [i for i in range(3) if i != up_idx]
    h1, h2 = axes[horiz_indices[0]], axes[horiz_indices[1]]

    # 6. Disambiguate horizontal orientation:
    #    a) The longer bounding-box span becomes X (forward).
    #    b) The denser end of X becomes +X.
    span1 = float(np.ptp(pts @ h1))
    span2 = float(np.ptp(pts @ h2))

    if span1 >= span2:
        forward, left = h1, h2
    else:
        forward, left = h2, h1

    # Ensure right-handed: left = up × forward
    left_check = np.cross(up, forward)
    if np.dot(left_check, left) < 0:
        left = -left

    # Denser end → +X
    proj = pts @ forward
    lo_bound = np.percentile(proj, 10)
    hi_bound = np.percentile(proj, 90)
    lo_count = (proj <= lo_bound).sum()
    hi_count = (proj >= hi_bound).sum()
    if lo_count > hi_count:
        forward = -forward
        left = -left  # keep right-handed

    # 7. Build the rotation matrix: rows = [forward, left, up]
    z = up / np.linalg.norm(up)
    x = forward - np.dot(forward, z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])

    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        R[1] *= -1

    info = {
        "up_votes": int(votes[up_idx]),
        "total_normals": len(normals),
        "bbox_span_x": round(float(np.ptp(pts @ R[0])), 3),
        "bbox_span_y": round(float(np.ptp(pts @ R[1])), 3),
        "bbox_span_z": round(float(np.ptp(pts @ R[2])), 3),
        "symmetry": "scene",
    }

    return R, info


def manhattan_axis_error_deg(normals: np.ndarray) -> float:
    """Median angle (degrees) between each normal and its nearest axis.

    A perfectly Manhattan-aligned cloud scores 0°.
    """
    abs_normals = np.abs(normals)
    max_component = np.clip(np.max(abs_normals, axis=1), 0.0, 1.0)
    angles = np.degrees(np.arccos(max_component))
    return float(np.median(angles))


# ---------------------------------------------------------------------------
# Scene index for the demo app
# ---------------------------------------------------------------------------

def build_scene_index(root: Path) -> dict:
    """Build {room_type: [{area, room, path}, ...]} for all S3DIS rooms."""
    rooms = discover_rooms(root)
    index = {}
    for r in rooms:
        rtype = r["room_type"]
        if rtype not in index:
            index[rtype] = []
        index[rtype].append({
            "area": r["area"],
            "room": r["room"],
            "id": f"{r['area']}/{r['room']}",
            "path": str(r["path"]),
        })
    return index


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Quick test: load one room, canonicalise it, print diagnostics."""
    root = Path("processed_data")
    types = discover_scene_types(root)
    if not types:
        print("no scenes found under processed_data/scene_s3dis/")
        return

    print(f"scene types: {', '.join(types)}")
    rooms = discover_rooms(root)
    print(f"total rooms: {len(rooms)}")

    # Pick the first office
    for r in rooms:
        if r["room_type"] == "office":
            break
    else:
        r = rooms[0]

    print(f"\ntest room: {r['area']}/{r['room']}")
    xyz, rgb = load_scene_cloud(r["path"])
    print(f"  points: {len(xyz)}")
    print(f"  xyz range: {xyz.min(0)} -> {xyz.max(0)}")

    R, info = canonicalise_scene(xyz)
    print(f"  R det: {np.linalg.det(R):.6f}")
    print(f"  info: {info}")

    canon = xyz @ R.T
    print(f"  canonical range: {canon.min(0).round(3)} -> {canon.max(0).round(3)}")


if __name__ == "__main__":
    _self_test()
