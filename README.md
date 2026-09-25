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

## Rule database

Class rules now live in [`rules.json`](rules.json). Add classes by selecting or
composing recipes from reusable geometric operations; no per-class Python
function is needed. All three CLIs accept `--rules path/to/rules.json`.
See [the database guide](RULE_DATABASE.md) for examples, validation, reference
policies, and migration checks.

[Support cues](SUPPORT_RULES.md) cover panel stands, spaced legs, and filled
flat bases. The active recipes select these operations and their thresholds
through JSON; weak or competing evidence keeps the preceding frame.

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

## Part check and vision-model check (branch `local_llm_integration`)

Two optional rule steps on top of the rules, switched on only through a rules
file, so the default pipeline and every score are unchanged:

- `part_check` (`part_cues.py`) labels object parts with PatchAlign3D.
  - Model: CVPR 2026, MIT licence, CPU only. Its 89 MB of weights download on
    first use.
  - It then re-orients by part layout: tail behind wings, legs below the top.
  - `rules_parts.json` switches it on for airplane and table.
- `vlm_check` (`vlm_cues.py`) asks a vision-language model, served by vLLM or
  Ollama, whether each of the six ways up looks upright. It reads the model's
  yes-probability.
  - It switches only past a margin, and only for classes that a pilot on both
    data halves approved.
  - If the server is down it falls back to the rules.

Both steps only choose among the 24 axis-aligned turns of the rule's frame, and
they keep the symmetry group, so they can't make the metric more lenient.
Details: [VLM_CHECK.md](VLM_CHECK.md).

### Step by step

```bash
# 0. code and environment
git clone https://github.com/usaben/geo_canon.git && cd geo_canon
git checkout local_llm_integration
conda create -n geocanon python=3.10 -y && conda activate geocanon
pip install -r requirements.txt
# processed_data/ is not in git: copy it next to geo_canon.py

# 1. baseline (plain rules)
python class_report.py --instances 100 --rotations 6 --refs none --no-figures \
    --out results_newb.txt --simple-out newb_simple.txt

# 2. rules + part check (first run downloads the PatchAlign3D weights)
python class_report.py --rules rules_parts.json --instances 100 --rotations 6 --refs none \
    --no-figures --out results_parts.txt --simple-out parts_simple.txt

# 3. vision-model server in its own env (vLLM pins its own torch).
#    Full-precision 8B needs about 17 GB of VRAM.
conda create -n vllm python=3.11 -y && conda activate vllm && pip install vllm
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --max-model-len 8192
#    leave it running; in a second terminal:
conda activate geocanon
export GEOCANON_VLM_URL=http://localhost:8000/v1
export GEOCANON_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct

# 4. pilot: which classes does the model fix without breaking any?
python vlm_pilot.py --rules rules_parts.json --half A --out pilot_A.json
python vlm_pilot.py --rules rules_parts.json --half B --out pilot_B.json
#    keep a class only if at the same margin it fixes more than it breaks on
#    BOTH halves and breaks nothing on half A

# 5. switch the check on for those classes (example: chair, monitor at margin 2)
python rules_tool.py --base rules_parts.json --op vlm_check \
    --classes chair,monitor --params '{"margin": 2.0}' --out rules_vlm.json

# 6. final report (answers come from vlm_cache/, so this is quick)
python class_report.py --rules rules_vlm.json --instances 100 --rotations 6 --refs none \
    --no-figures --out results_vlm.txt --simple-out vlm_simple.txt
diff results_newb.txt results_vlm.txt

# 7. look at it in the browser: new pipeline on 8001, plain rules on 8000
python demo_app.py --rules rules_vlm.json --port 8001 --clouds photos --classify
python demo_app.py --port 8000 --clouds photos --classify
```

Use `rules_parts.json` in steps 6 and 7 if no class passes the pilot. Without
a GPU, step 3 can use Ollama instead: `ollama pull qwen3-vl:8b-instruct`, then
`GEOCANON_VLM_URL=http://127.0.0.1:11434/v1` and
`GEOCANON_VLM_MODEL=qwen3-vl:8b-instruct`.

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
| `part_cues.py` | `part_check` step: PatchAlign3D part labels to part-layout re-orientation |
| `vlm_cues.py` | `vlm_check` step: vision-model upright check through vLLM or Ollama |
| `vlm_pilot.py` | per-class fixed/broken counts that decide where `vlm_check` may run |
| `rules_tool.py` | adds `part_check` / `vlm_check` to chosen classes of a rules file |
| `rules_parts.json` | `rules.json` + `part_check` for airplane and table |
| `uni3d_centroids.npz` | 21 class centroids (1024-d), leave-one-out accuracy 75.4% |

**Stale — regenerate before citing:** `refs.npz`, `class_report.txt`,
`class_rules.txt`, `stability_consistency.txt`, `pipeline.txt` and `figures/`
all predate the current rules, including the up-sign fixes to car and chair.
