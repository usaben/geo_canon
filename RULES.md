# Orientation rule changes

Class selection and cue recipes are now stored in [`rules.json`](rules.json).
See [RULE_DATABASE.md](RULE_DATABASE.md) to add a class, compose operations, or
load another database. The geometric decisions described below are preserved.

The frame remains right-handed: **x forward, y left, z up**. Rules use geometry
only. Stored dataset pose, file names, and evaluation up axes never enter a
rule. Unknown labels retain the generic fallback. `plane` and `aeroplane`
dispatch to the airplane rule, including its references.

## Cases handled

| Family | Construction and fallback |
| --- | --- |
| Airplane | Reject the thin wing-plane normal as a lateral axis. Use a wing plate for roll. Look for a localized dorsal protrusion, then asymmetric end-fin profiles, then a secondary horizontal stabilizer, then several end-width bands. These branches cover inset stabilizers, canards, twin fins, high wings, and wide spans. Robust tail quantiles replace a single extreme point for up. |
| Bed | Choose the platform plane. Compare **spatial** height bands: a headboard occupies less fore/aft extent than the platform below it. Multiple crown bands identify the back. Dense mattresses no longer consume an entire mass-quantile band and reverse up. |
| Bathtub / sink | Compare interior coverage of opposite end slabs to distinguish a closed floor from an open rim. Test principal and fitted plate normals. Use the long horizontal direction and crown asymmetry for head/foot orientation. Complex sinks remain difficult. |
| Bench | Keep the seat/support construction and compare multiple crown bands for backed benches. Backless benches may have a measured `C2z` yaw ambiguity; they are no longer automatically treated as upside-down equivalents. |
| Door / keyboard | Preserve the shared panel frame. Measure half-turns about all three canonical axes. A featureless rectangle can have `D2`; an asymmetric panel does not automatically get that group. This improves the symmetry-aware metric, not the raw frame dispersion. |
| Cone | Find the rotational axis and point toward the narrow apex. |
| Stool | Use the existing table surface/support rule and measured yaw symmetry. |
| Vase | Find the rotational axis, refine against a terminal plane, and compare end closure using each end's own radius. This separates a narrow open neck from a closed base. |

The airplane stabilizer profile treats flat maxima as a plateau and includes
bin boundaries symmetrically. Picking the first maximum had allowed an
unsigned PCA axis reversal to change the nose decision by 180 degrees.

## Confidence, symmetry, and references

`up_confidence` and `forward_confidence` are normalized geometric margins,
**not calibrated probabilities**. `ambiguous_axes` lists available margins
below 0.15. The evaluator now saves `per_instance_rule_info` and `fallback_count`
alongside errors and IDs, so an outlier can be traced to its actual cue.

Weak cues do not create a symmetry. New basin/bench half-turn tests require an
absolute match score of at least 0.65 as well as a relative advantage over
generic rotations. Panels require at least 0.75 for each accepted half-turn;
`D2` requires all three, including the product of the generators. Raw error is
still reported. Small details below the sampling/kernel resolution can remain
unresolved by these approximate symmetry tests.

Default reference classes are airplane, car, chair, **bed, and bathtub**. Build
references separately from the evaluation objects when assessing generalization.
Airplane reference clouds retain the semantic rule frame rather than being
freely turned into agreement. A margin of at least 0.35 locks that signed axis
against reference permutations; weak axes remain eligible for correction.
Airplane snapping must improve shape distance by 3%. The existing ICP correction
is still limited to 25 degrees. Bathtub symmetry is checked again about the final
up axis after reference correction.

Saved reference files now record `ruleset_version`. Older caches remain readable
but emit a rebuild warning. From the repository root, rebuild a cache with:

```powershell
python geo_canon/geo_canon.py --data modelnet_aug0_by_category --instances 75 --rotations 2 --save-reference geo_canon/refs_v2.npz --no-figures --out geo_canon/evaluation
```

`class_report.py` loads references rather than building them. To use the same
pipeline there, pass `--refs geo_canon/refs_v2.npz` when running it from the
repository root. `--no-reference` remains available on the main CLI, but the
bathtub consistency improvement depends on the rebuilt reference stage.

Explicit class rules work without Uni3D. The bundled classifier centroids do not
automatically acquire the four new labels.

## Validation

Run the repository tests from its root:

```powershell
python -m unittest discover -s tests -v
```

The geometry tests cover known-pose aircraft variants, noise and decimation,
rotation/scale/translation, point reordering, headboards, open basins and vases,
reference save/load, adversarial backward references, plateau sign invariance,
and both symmetric and asymmetric panels.

Reproduce the current held-out evaluation:

```powershell
python scripts/benchmark_geo_canon_rules.py --data modelnet_aug0_by_category --classes airplane,bathtub,bed,bench,door,keyboard,cone,sink,stool,vase --instances 25 --offset 50 --rotations 2 --dataset-up z --reference-instances 10 --out geo_canon/validation/heldout_final.json
```

For a before/after comparison, also pass `--baseline-module` pointing to a trusted
copy of the earlier `geo_canon.py`. The local snapshot used during this review is
`.tmp/geo_canon_review/baseline.py`. The benchmark uses exact single-threaded tree
queries to avoid thread-pool overhead on small probe sets.

Objects 1–10 supply references; objects 51–75 supply the held-out evaluation.
Objects 11–50 in `airplane_near_split.json` are a development regression check,
not a second unseen validation set. Every JSON records IDs, seed, rotations,
raw and symmetry-aware dispersion, upright accuracy, and cue diagnostics.
`--dataset-up z` is an explicit ModelNet evaluation assumption. It is never
passed to the canonicaliser. Stored yaw conventions vary in this local data,
so consistency and upright accuracy should be read together.

The final measurements are in [validation/RESULTS.md](validation/RESULTS.md).
Earlier `heldout_rules.json` and `heldout_airplane_references.json` record
intermediate development states and are superseded by `heldout_final.json`.

## Remaining limitations

These are geometric heuristics, not a guarantee of a unique semantic pose.
Occlusion, tiny fins, dense landing gear, closed/solid vase models, and compound
sink basins can still defeat them. Nearly symmetric objects cannot supply a
reliable sign that their geometry does not contain. Sofa, toilet, monitor, and
piano experiments were not retained when they failed the consistency/pose
checks; their remaining outliers are deferred.
