# Support cues configured through JSON

The main `rules.json` assigns reusable support operations to table, bench,
chair, sofa, monitor, laptop, bed, bathtub, piano, wardrobe, and flower_pot.
There are no new class-specific Python functions. Existing frame constructors
provide the starting frame; the support operations measure evidence and either
correct that frame or keep it.

## Supports under a panel

Monitor's `monitor_stand` recipe first uses the existing upright construction,
with `crown_pct: 88`, then applies `panel_on_support`:

```json
{
  "op": "panel_on_support",
  "params": {
    "panel_frac": 0.18,
    "slab_frac": 0.08,
    "stem_width_ratio": 0.55,
    "base_expansion": 1.25,
    "min_points": 10,
    "score_threshold": 0.12,
    "margin_threshold": 0.08,
    "forward_policy": "crown",
    "crown_pct": 88,
    "foot_bands": [0.04, 0.08, 0.12],
    "shoulder_bands": [0.10, 0.14, 0.20, 0.30, 0.40, 0.50]
  }
}
```

The operation fits a broad panel, fits its rectangular edges to avoid unstable
in-plane PCA axes on square screens, and searches both signs of both panel
directions. It requires a narrower, centered stem and a wider foot below it.
Several relative-height bands handle short and tall stems. The winning score
must clear an absolute threshold and a margin over the competing direction.
No stand or weak evidence leaves the incoming frame untouched.

`forward_policy` controls the independent front/back decision:

- `preserve`: retain the previous forward sign when it aligns with the fitted
  normal; otherwise use crown offset.
- `crown`: use the upper slice's offset along the fitted normal. This is the
  active monitor setting, consistent with the previous recipe's convention.
- `density`: use face-density asymmetry when it exceeds `face_threshold`, then
  fall back as above. This experimental option worsened development consistency
  and is **not** enabled in the main database.

Detecting the stand determines up, not necessarily the screen-facing direction.
Metadata exposes `stand_detected`, `stand_reason`, `stand_score`, `stand_margin`,
the measured width/expansion ratios, and separate up/forward confidence margins.
The operation keeps symmetry `I`; it does not hide front/back mistakes as yaw
symmetry.

## Legs and filled flat bases

`support_base` measures the physical-looking contact geometry without assuming
four legs, or even separate legs. The same operation can handle a pedestal,
three/four spaced feet, or a solid base:

```json
{
  "op": "support_base",
  "params": {
    "mode": "legs",
    "axis_search": "sign",
    "contact_frac": 0.04,
    "lower_frac": 0.30,
    "max_lower_mass": 0.35,
    "min_points": 10,
    "score_threshold": 0.30,
    "margin_threshold": 0.20,
    "forward_policy": "crown",
    "crown_pct": 50,
    "symmetry_allow": ["I"]
  }
}
```

Append this to an existing recipe, after the frame constructor. Modes are:

| Mode | Evidence |
| --- | --- |
| `legs` | A broad contact footprint below a relatively sparse lower body. Does not count or require four legs. |
| `flat_base` | Approximately planar contacts with points filling the footprint's interior; an open rim receives weaker evidence. |
| `either` | Prefer credible sparse-support evidence; otherwise test a filled base. This prevents a solid tabletop from overruling its legs and accommodates both legged and solid-base forms. |

Both modes require enough contact points and a support polygon containing the
projected cloud centroid. A ring and separated corner feet have low central
occupancy; a filled base has interior samples. The calculations use relative
areas and heights, so the same parameters work under scale changes.

The cloud centroid is a geometric proxy, not a measured centre of mass. Sparse
surface samples do not establish physical balance. These scores are heuristic
evidence and their margins are not calibrated probabilities.

Search choices are:

- `sign`: compare the two ends of the existing up axis. This preserves the
  object's unsigned geometric axes and is the conservative default.
- `principal`: also consider principal axes, permitting an axis correction.
- `planes`: also fit strong planar regions, permitting a seat/base plane to
  replace an incorrect initial axis.

Axis search is explicit because a flat screen, tabletop or cabinet top can also
look like a stable base. Scores below `score_threshold`, competing candidates
within `margin_threshold`, or a centroid outside the contact polygon preserve
the previous frame. Symmetric top and bottom surfaces often have no observable
up/down distinction; this cue deliberately abstains then.

`forward_policy` is `preserve`, `crown`, or `crown_vote` (multiple crown bands).
It is evaluated only when a correction is accepted. An axis change rechecks
rotational symmetry using `symmetry_allow`; its default `["I"]` grants no new
yaw ambiguity. A sign-only correction preserves the old symmetry group.

Diagnostics include `support_mode`, `support_applied`, `support_reason`,
`support_margin`, both original end scores, and the winning support evidence.
Evidence includes contact count, footprint fraction, balance, central fill,
lower-body point fraction, and contact flatness when measurable.

## Adding another object or changing a cue

For an object that shares a known construction, add a class record pointing at
one of the supported recipes. A sign on a stand can reuse `monitor_stand`; a
legged seating object can reuse `chair_supported` or `bench_supported`.

To combine different cues, copy a recipe under a new name and edit its steps.
All thresholds, support type, candidate-axis search and forward policy are data.
Editing a shared recipe affects every class referring to it.

The same interface is available in Python without changing process defaults:

```python
import json
from geo_canon import geo_canon as g

document = json.loads(g.DEFAULT_RULES_PATH.read_text())
document["classes"].append({
    "name": "freestanding_sign",
    "recipe": "monitor_stand",
    "symmetry": "I"
})
database = g.RuleDatabase(document, g.OPERATIONS)
canon = g.make_canonicaliser(rule_database=database)
rotation, diagnostics = canon(points, "freestanding_sign")
```

No per-object rule function or dispatch edit is necessary. A genuinely new
geometric measurement can be implemented once and registered as another
operation. JSON never executes Python source.

## Validation

`tests/test_support_cues.py` covers one/three/four supports, filled base versus
open rim, symmetric-base abstention, invalid balance footprints, portrait and
landscape panels, absent stands, rotations, scale, translation, point order,
noise, decimation, validation, and adding a new class through data.

The historical rules and their 50-result fixture remain separately available
under `tests/fixtures/legacy_rules_v2.json` and `geo_rule_results.json`.

Run the targeted tests and compare databases on disjoint real objects:

```powershell
python -m unittest discover -s tests -p test_support_cues.py
python -m unittest discover -s tests -p test_rule_database.py
python scripts/benchmark_support_rules.py --data modelnet_aug0_by_category --offset 75 --instances 25 --rotations 2 --dataset-up z
```

The benchmark uses identical objects and rotations for both databases, no
references, and writes per-instance scores, correction counts, stability,
consistency and optional stored-up errors to `validation/support_rules.json`.
The `--dataset-up` value is strictly an evaluation assumption, never an input to
the rule. Stored-pose consistency and semantic up accuracy are different checks;
inspect both, plus stability, before claiming a rule improves a class.

Measured outcomes and limitations are summarized in
[validation/SUPPORT_RESULTS.md](validation/SUPPORT_RESULTS.md). These cues do
not improve every class or every metric; conservative abstention is common.
The active numerical ruleset is version 3. Saved reference caches from older
rulesets should be rebuilt; the loader warns on a version or database mismatch.
