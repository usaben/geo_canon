# Editing the rule database

`rules.json` is the source of truth for class names, aliases, ShapeNet synsets,
frame recipes, default symmetry, and reference policies. Add a class by adding a
record and selecting or composing a recipe. There is no class dispatch table or
`rule_<class>` function to maintain in Python.

JSON supports typed parameters and ordered steps without putting nested JSON
inside CSV cells. It needs no additional dependency. `RuleDatabase(document,
OPERATIONS)` also accepts decoded records, so a future SQLite/Postgres adapter
can supply the same document without changing the geometry pipeline.

## Add a class

Append this record to `classes` to reuse the existing panel recipe:

```json
{
  "name": "display_panel",
  "aliases": ["display_board"],
  "recipe": "flat_panel",
  "symmetry": "I"
}
```

The class is now available to canonicalisation, class discovery, the real-data
loader and the demo. Put its clouds in `<data>/display_panel/`. An optional
eight-digit string `synset` enables loading a ShapeNet folder instead.
Names and aliases are case-insensitive at dispatch; spaces and hyphens become
underscores. Records must use normalized names. Class order in this file is
preserved, including the original evaluation seed ordering.

The synthetic generator and the optional learned classifier still need their own
shape generator/training data for a new category. Adding geometric rules does
not teach the classifier a new label.

## Compose a recipe

Add a named recipe under `recipes`, then point a class record at its name:

```json
"raised_panel": [
  {
    "op": "upright",
    "params": {"up_axis": "principal", "up_sign": "wide_up", "crown_pct": 80}
  },
  {"op": "crown_forward"},
  {"op": "axial_symmetry", "params": {"allow": ["C2z"], "min_score": 0.65}}
]
```

The first operation builds a frame. Later operations receive the current frame
and diagnostics and refine them. Steps execute in order. The example illustrates
composition; validate a new recipe on representative held-out objects before
claiming its orientation is correct. Changing a shared recipe changes every
class that uses it; copy the recipe under a new name for an independent variant.

Available frame constructors:

| Operation | Parameters / construction |
| --- | --- |
| `bilateral` | `lateral`: `mirror` or `principal`; `up_axis`: `plate`, `plate_or_pca`, `support`; `interior_only`, `min_frac`, `plate_cue`, `project_up`; `up_sign`; `forward`: `crown`, `crown_vote`, `roof_offset`; `crown_pct` |
| `upright` | `up_axis`: `plate`, `principal`, `sharpest`, `tallest`; `up_sign`, `crown_pct`, `flip_symmetry` |
| `revolution` | `wide_end_up`, `sign_forward`, `narrow_tie_up`, `end_frac` |
| `slab` | Principal `up_axis` and `fwd_axis` indices (0–2, distinct), `up_sign` |
| `principal` | Principal `up_axis`, `fwd_axis`, `end_frac`; narrow end up and positive face skew forward |
| `winged` | Bilateral wings, wing plate, fin/stabilizer/end-profile evidence |
| `platform` | Dominant terminal plate and its minimum-area rectangle |
| `open_basin` | Closed floor versus open rim; long horizontal axis and crown |
| `supported_case` | Leg-supported horizontal case versus upright box |
| `profile_box` | Tight side-profile rectangle, support, and empty upper corner |
| `generic` | Mirror plane, support direction, end spread; unknown-label fallback |

`up_sign` accepts `support`, `floor`, `top_short`, `wide_up`, `third`, or `base`.
`bilateral` also supports `taper` and `upper_structure`.

Available modifiers:

| Operation | Parameters / effect |
| --- | --- |
| `crown_forward` | Sign forward using multiple crown bands; report confidence |
| `axial_symmetry` | `allow`: subset of `C2z`, `C4z`, `Cinfz`; `min_score` |
| `panel_symmetry` | `min_score` (default 0.75); measure all three half-turns |
| `end_closure` | `min_score` (default 0.65); refine a vessel axis and compare local end closure |

Omitted parameters use the defaults on the registered geometric operation.
`schema_version` is currently `1`. `fallback` names the recipe used for unknown
labels. A class's `symmetry` is only a default when its operations do not report
one; measured symmetry takes precedence.

The numerical algorithms remain Python operations. Complex features such as fin
detection retain their existing numerical decisions. Supporting a genuinely new
geometric feature may require a reusable operation, registered once in
`OPERATIONS`; it does not require a new function for every class. Rule files
cannot contain executable Python, dynamic imports, or expressions.

## Load and validate

The bundled database loads automatically, relative to the module (independent
of the working directory). Each CLI accepts an alternate file:

```powershell
python geo_canon/geo_canon.py --rules my_rules.json --synthetic --instances 3 --no-reference --no-figures
python geo_canon/demo_app.py --rules my_rules.json --data modelnet_aug0_by_category
python geo_canon/class_report.py --rules my_rules.json --data modelnet_aug0_by_category
```

For isolated experiments without changing process defaults:

```python
from geo_canon import geo_canon as g

database = g.load_rule_database("my_rules.json")
canon = g.make_canonicaliser(rule_database=database)
rotation, diagnostics = canon(points, "display_panel")
# Or: g.canonicalise(points, "display_panel", rule_database=database)
```

`g.configure_rules(path)` explicitly changes the process default and updates
class discovery and reference defaults. Call it at startup before loading data
or references; do not use it for concurrent reloads. Bound canonicalisers retain
their database snapshot. The existing `PCA_FIRST` and geometric tuning globals
are still process-wide settings.

Loading validates the entire store before activation: duplicate keys, names,
aliases and synsets; unknown recipes and operations; step ordering; parameter
names, types, choices and relevant ranges. Invalid configuration raises
`RuleValidationError` before clouds are processed. It cannot silently turn a
whole class into PCA fallback. The existing per-cloud numerical failure fallback
and its diagnostics remain unchanged.

Recipes are compiled once and class lookup is a dictionary lookup. There is no
per-cloud file access or parsing, and adding classes does not make existing
classes scan more rules. Most runtime remains geometric nearest-neighbour and
direction searches.

## References and reproducibility

A class may specify a `reference` object with:

- `enabled` and `refine`: include it in default reference building and ICP refinement.
- `clusters`: positive number of reference clusters (default 1).
- `preserve_semantics`: retain reference frames instead of freely aligning them.
- `lock_confidence`: threshold for protecting confident forward/up axes.
- `min_improvement`: minimum improvement before reference snapping.
- `recheck_symmetry`: `allow` and `min_score` for checking the final up axis.

Saved references and evaluation reports include a SHA-256 fingerprint of the
validated database. Loading references with a missing or different fingerprint
warns that they should be rebuilt. Older caches remain readable. The numerical
ruleset version remains 2 because this migration preserves the algorithms.
`build_reference`, `save_references`, and `load_references` accept an optional
`rule_database` for isolated experiments; pass the same database and a bound
canonicaliser throughout.

## Validation of this migration

The old and new implementations matched **exactly** on 200 real-cloud runs:
two instances per each of the 25 classes, original and transformed clouds, with
PCA preprocessing on and off. Both rotation matrices and all diagnostics matched;
neither version fell back. See `validation/rule_database_migration.json`.

`tests/fixtures/geo_rule_results.json` records 50 deterministic procedural/random
cloud results from the original implementation, covering all classes with PCA on
and off. The tests also exercise new classes, changed parameters, composition,
validation, discovery, isolated snapshots, and reference fingerprints:

```powershell
python -m unittest discover -s tests -p test_rule_database.py
python -m unittest discover -s tests -p test_geo_canon.py
```

At migration time the existing geometry suite had eight errors, reproduced
before and after the rewrite: missing airplane half-turn helpers and missing
`up_sign_cue` metadata. Those pre-existing failures are not hidden by these tests.
