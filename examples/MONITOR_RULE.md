# Improving the monitor recipe

The main database now includes the reusable `panel_on_support` operation and
the `monitor_stand` recipe. See [SUPPORT_RULES.md](../SUPPORT_RULES.md) for the
current implementation. The high-crown example and measurements below record
the earlier JSON-only experiment and remain available for comparison.

JSON selects and configures the geometric operations available in the engine.
A new geometric measurement, such as detecting a screen supported by a stem
and base, first needs a reusable operation in Python. It can then be selected
and tuned through JSON for monitors, signs, displays, and other related objects.

## A runnable example using current operations

The default monitor recipe is `wide_top_upright`: find the tallest principal
axis outside the mirror normal, put the wide end up, then sign forward using
the mean position of points above the 72nd height percentile.

`monitor_high_crown.rules.json` is a complete alternative database. It copies
the default database, adds the following recipe, and changes only monitor's
recipe assignment:

```json
"monitor_high_crown": [
  {
    "op": "upright",
    "params": {
      "up_axis": "principal",
      "up_sign": "wide_up",
      "crown_pct": 88,
      "flip_symmetry": false
    }
  }
]
```

In the monitor class record, set `"recipe": "monitor_high_crown"`. Keep
`"symmetry": "I"`: monitor front/back or upside-down errors must still count.
An 88th-percentile crown means the uppermost 12% of points by projected height,
not the uppermost 12% of the object's physical height. The remaining calculation
is unchanged. This is a forward-sign experiment, not a stand detector.

Run from the workspace root:

```powershell
python geo_canon/class_report.py --rules geo_canon/examples/monitor_high_crown.rules.json --data modelnet_aug0_by_category --no-figures --out monitor_candidate_report.txt --simple-out monitor_candidate_summary.txt
```

`class_report.py` evaluates every available class. Its output paths are relative
to the working directory unless absolute paths are supplied. The following API
example evaluates monitor alone without changing process defaults:

```python
from geo_canon import geo_canon as g

database = g.load_rule_database("geo_canon/examples/monitor_high_crown.rules.json")
canon = g.make_canonicaliser(rule_database=database)
clouds, ids = g.load_folder("modelnet_aug0_by_category/monitor")
results, _, _ = g.evaluate(
    {"monitor": (clouds[25:75], ids[25:75])},
    canon, k_rot=3, seed=713, want_features=False,
)
print(results["monitor"]["consistency_sym_deg"])
```

## Measured result and limits

Eighteen combinations of existing up-sign cues and crown percentiles were tested
on the first 25 naturally sorted monitor files from `modelnet_aug0_by_category`.
The selected candidate was then evaluated once on files 26–75, with three random
rotations per instance, PCA preprocessing enabled, and no references. The
candidate does not change other classes or the active default database.

| Held-out metric, 50 monitors | Default | High crown |
| --- | ---: | ---: |
| Consistency RMS, degrees | 117.64 | 109.66 |
| Consistency median, degrees | 113.80 | 99.93 |
| Stability, degrees | 0.042 | 0.042 |
| Up within 15 degrees of stored z-up | 62% | 62% |
| Numerical fallbacks | 0 | 0 |

Lower angular errors are better. This is a modest improvement with substantial
remaining errors. Consistency measures agreement, not semantic correctness;
the up check additionally assumes these data are stored z-up. Generic support
and floor sign cues were worse on the development sample, so substituting them
does not establish stand detection.

Full development results, held-out filenames, per-instance diagnostics, and
metrics are saved in `../validation/monitor_high_crown.json`. The benchmark used
one worker per nearest-neighbour query for speed; the query calculation is
unchanged. The example is an experimental snapshot of the database, so unrelated
future changes to the default database will not automatically appear in it.

## What proper stand detection would require

A reusable panel-on-support operation should score a geometric relationship:

1. Fit the broad screen panel and use its normal as the unsigned forward axis.
2. Look around its in-plane boundary for a narrower stem leading to a wider,
   approximately perpendicular foot. Score support on both sides and require
   enough points and a clear score margin before selecting down.
3. Set up from the support toward the screen, perpendicular to the screen normal.
4. Resolve front/back separately from screen-face or rear-housing geometry;
   finding the stand alone does not determine the screen-facing direction.
5. Report confidence and retain the existing recipe when the stand is missing,
   occluded, or not distinguishable (for example, wall-mounted displays).

Register that operation once in `OPERATIONS`, expose geometric thresholds such
as stem/panel width ratio, minimum support points, and the required score margin,
and select those parameters in a new JSON recipe. This is now implemented as
`panel_on_support`; the active forward policy uses crown offset because the
experimental face-density sign was less consistent on the development data.

Test the detector on known-pose synthetic examples (wide and narrow screens,
tilted panels, portrait screens, stands with different bases, and no stand),
then on disjoint real instances, rotations, noise, and point removal. Check both
raw consistency and semantic axis errors before promoting it to the default.
