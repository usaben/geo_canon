# CanonNet — class-conditional geometric canonicalisation

Given a point cloud and its class label, this computes a canonical rotation in
SO(3) by a deterministic sequence of geometric constructions — principal axes,
symmetry detection, then a class-specific rule. **No learning, no weights, no
training data** in the canonicaliser itself. (Uni3D is used only to *name* the
class, and only if you turn it on.)

The current rule changes and validation are documented in
[RULES.md](RULES.md). There are now 25 explicit class rules, including cone,
sink, stool, and vase. Existing Uni3D centroids still cover the original 21
classes; use explicit labels for the additions or rebuild the centroids.
Saved references from earlier rules should be rebuilt, because their axis
conventions can otherwise reintroduce flips.

The pipeline reports two numbers, and they answer different questions:

- **stability** — one cloud, many input poses. Does the frame follow the object?
- **consistency** — different instances of a class. Do they agree with each other?

---

## Setup

Python 3.10. Core requirements:

```
pip install -r requirements.txt
```

That covers everything except class *prediction*. If you want the demo to name
clouds for you (`--classify`), you also need the Uni3D source tree, which is not
pip-installable:

```
git clone --depth 1 https://github.com/baaivision/Uni3D vendor/Uni3D
```

Run that command from `geo_canon`. The loader also accepts `~/Uni3D`, or set
`UNI3D_ROOT` to wherever you cloned it. The weights
(~44 MB) download automatically from the HuggingFace hub on first run and are
cached. `uni3d_cpu.py` stubs out the CUDA extension the official repo asks for,
so this runs on CPU with no nvcc.

**Without `--classify`** the demo falls back to a PCA-overlay classifier that
only knows three classes and calls nearly everything a car. Use `--classify`, or
always pick the class from the dropdown yourself.

---

## Data

```
processed_data/
    <class>/<class>_<n>.pt      # 1024-point clouds, one folder per class
    manifest.csv                # maps each file back to its ShapeNet/ModelNet id
```

21 classes: airplane, bathtub, bed, bench, bookshelf, bottle, bowl, car, chair,
door, flower_pot, guitar, keyboard, lamp, laptop, monitor, piano, sofa, table,
toilet, wardrobe.

**What this repo carries: 50 clouds per class** (614 files, 8.8 MB) — enough for
the demo and for `class_report.py`, which defaults to fewer than that. The full
set is larger and lives outside git; `manifest.csv` lists every file of it, not
just the ones here, so it doubles as the index of what you are missing.

The ModelNet-derived files are dicts of `{points, variant}`. They originally also
carried `knn_idx`, a 16-nearest-neighbour index that is 128 KB of each 140 KB
file and that nothing in this pipeline reads; it was dropped here to keep the
clone small. It is exactly recoverable from the points if you ever need it:

```python
from scipy.spatial import cKDTree
knn_idx = cKDTree(points).query(points, k=17)[1][:, 1:]   # self excluded
```

The loaders also read `.npy`, `.npz`, and — sampled on the fly — `.obj` and
`.ply` meshes, so you can point `--data` at a raw ShapeNet tree. A mesh is
sampled at `--mesh-points` (default 8192); a `.ply` holding bare vertices, such
as a Gaussian splat, is read as-is with its splat attributes dropped.

---

## Running things

**The browser demo** — turn an object any way you like, watch the canonical
frame stay put:

```
python demo_app.py --clouds photos --classify
# open http://localhost:8000
```

You can drag-drop or upload `.pt`, `.npy`, `.npz`, `.obj`, `.ply`. Uploads are
kept in the `--clouds` folder and reloaded next start. The chip on each object
shows its detected symmetry group and, where references are used, its cluster.

**The headline evaluation** — stability and consistency per class, plus figures:

```
python geo_canon.py --data processed_data --instances 25 --rotations 8
python geo_canon.py --synthetic            # procedural shapes, no data needed
python geo_canon.py --help                 # --mesh-points, --robustness, etc.
```

**Per-class detail** (writes `class_report.txt` and `stability_consistency.txt`):

```
python class_report.py --data processed_data
```

**Rebuild the class centroids** — needed whenever the class list changes, since
Uni3D can only name classes that have a centroid:

```
python uni3d_probe.py --data processed_data --ref-instances 12
```

**Sample fresh clouds from ShapeNet meshes** into the `<class>_<n>.pt`
convention, appending to `manifest.csv`:

```
python obj_to_pt.py
```

---

## Historical measurements

The figures below predate the current airplane, bed, bathtub, bench, and panel
changes. Use [the current validation notes](RULES.md#validation) for those
classes; these older figures are retained as background.

Measured against each dataset's own stored orientation, which is ground truth
for the up axis and needs no convention. "up<15°" is the share of instances
whose canonical up lands within 15 degrees of the true up.

| class | up<15° | verdict |
|---|---|---|
| table | 94% | works |
| car | 92% | works |
| bench | 80% | works |
| flower_pot | 79% | works |
| monitor | 70% | works |
| chair | 72% | works |
| wardrobe | 65% | weak — a box upside down is still a box |
| door | 65% | axes are exact; the up sign is near-symmetric |
| lamp | 57% | shade-up is confirmed; the sign is the weak part |
| bookshelf | 50% (85% modulo its own flip symmetry) | weak |
| **toilet** | **30–36%** | **does not work — do not quote it** |
| keyboard | n/a by construction | shares the door's frame; see below |

Read the rule docstrings in `geo_canon.py` before trusting any class — each one
records what was measured and what was rejected, including the dead ends, so
nobody repeats them.

Three things worth knowing before you write anything up:

- **Consistency is not correctness.** Cars once scored a healthy 3.3° consistency
  while being upside down on 11 of 12 instances: they agreed beautifully on the
  wrong pose. Always read the up-accuracy alongside it.
- **Door and keyboard share one frame on purpose.** Neither Uni3D nor any
  geometric rule can separate a door from a keyboard — both are thin rectangular
  panels — so they get the same alignment and a misprediction costs nothing. A
  keyboard therefore canonicalises standing on edge. Measuring its "up" against
  ModelNet's gives 0% by construction, not by failure.
- **Some classes are not one shape.** Bench holds long backless planks and
  compact backed benches (aspect ratio 2.30 ± 0.91, against 1.43 ± 0.29 for
  chair). A plank has no front, so its azimuth cannot be pinned down.

---

## Files

| file | what it is |
|---|---|
| `geo_canon.py` | loaders, symmetry detection, 25 class rules, references, metrics, CLI |
| `demo_app.py` | the browser demo and its HTTP API |
| `uni3d_cpu.py` | loads Uni3D-S on CPU and encodes clouds |
| `uni3d_probe.py` | builds `uni3d_centroids.npz` from labelled clouds |
| `class_report.py` | per-class stability/consistency report and figures |
| `obj_to_pt.py` | samples ShapeNet `.obj` meshes into `processed_data` |
| `uni3d_centroids.npz` | 21 class centroids (1024-d), leave-one-out accuracy 75.4% |

**Stale — regenerate before citing:** `refs.npz`, `class_report.txt`,
`class_rules.txt`, `stability_consistency.txt`, `pipeline.txt` and `figures/`
all predate the current rules, including the up-sign fixes to car and chair.
