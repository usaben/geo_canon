"""Class-independent evidence for a resting surface in a point cloud.

These are geometric plausibility scores, not physical stability certificates:
the point-cloud centroid is only a proxy for a solid object's centre of mass.
All calculations use relative heights, areas and distances, so no dataset pose,
units, number of legs, or class name enters the measurements.
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError


def surface_evidence(X, up, contact_frac=0.04, lower_frac=0.30,
                     max_lower_mass=0.35, min_points=10):
    """Score sparse supports and filled bases for one signed up direction."""
    eps = 1e-12
    up = up / max(np.linalg.norm(up), eps)
    h = X @ up
    lo, hi = np.quantile(h, [0.005, 0.995])
    height = max(float(hi - lo), eps)
    contact = h <= lo + contact_frac * height
    count = int(contact.sum())
    result = dict(legs=0.0, flat_base=0.0, contact_points=count,
                  footprint_fraction=0.0, balanced=False, central_fill=0.0,
                  lower_mass=float(np.mean(h <= lo + lower_frac * height)))
    if count < min_points:
        return result

    # Any orthonormal projection has the same hull areas/distances. The choice
    # of basis cannot enter a grid or a bounding-box approximation here.
    seed = np.eye(3)[int(np.argmin(np.abs(up)))]
    a = seed - (seed @ up) * up
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    P = np.column_stack([X @ a, X @ b])
    foot = P[contact]
    try:
        whole_hull, foot_hull = ConvexHull(P), ConvexHull(foot)
    except QhullError:
        return result
    if min(whole_hull.volume, foot_hull.volume) <= eps:
        return result

    centre = P.mean(0)
    distances = -(foot_hull.equations[:, :2] @ centre + foot_hull.equations[:, 2])
    balanced = bool(np.min(distances) >= -1e-8 * np.sqrt(whole_hull.volume))
    footprint = min(1.0, float(foot_hull.volume / whole_hull.volume))
    result.update(balanced=balanced, footprint_fraction=footprint)
    if not balanced:
        return result

    # Whitening makes central occupancy insensitive to round vs rectangular
    # footprints and to an arbitrary in-plane basis. A ring or four corner
    # feet leave the interior empty; a filled base has interior samples.
    centred = foot - foot.mean(0)
    w, V = np.linalg.eigh(centred.T @ centred / len(centred))
    if w[0] <= eps:
        return result
    radius = np.linalg.norm((centred @ V) / np.sqrt(w), axis=1)
    fill = float(np.mean(radius < 0.5 * np.quantile(radius, 0.95)))
    # Coplanar contacts concentrate height; a rounded end or a thick volume
    # slice provides weaker evidence. Density alone cannot establish a base.
    flatness = max(0.0, 1.0 - float(np.std(h[contact])) / (0.5 * contact_frac * height))
    sparse = max(0.0, 1.0 - result['lower_mass'] / max_lower_mass)
    result.update(central_fill=fill, contact_flatness=flatness,
                  legs=float(np.sqrt(footprint) * sparse),
                  flat_base=float(np.sqrt(footprint) * flatness * min(1.0, fill / 0.18)))
    return result
