"""Validated, data-only recipes for geometric canonicalisation.

No imports, Python expressions or function bodies are accepted from the store.
Operations are explicitly registered by the application. Loading compiles recipes
once; processing a cloud does no file I/O and only looks up its class in a dict.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from functools import partial
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import MappingProxyType


class RuleValidationError(ValueError):
    """Invalid rule data (never treated as a geometric/PCA fallback)."""


def normalise_label(value):
    return value.strip().lower().replace("-", "_").replace(" ", "_") if value else ""


def _check(condition, message):
    if not condition:
        raise RuleValidationError(message)


def _fields(value, allowed, where, required=()):
    _check(isinstance(value, dict), f"{where}: expected an object")
    _check(not (value.keys() - set(allowed)), f"{where}: unknown fields {value.keys() - set(allowed)}")
    _check(set(required) <= value.keys(), f"{where}: required fields {required}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _check(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class Operation:
    function: object
    modifier: bool = False
    choices: object = None

    def compile(self, params, where):
        signature = inspect.signature(self.function)
        defaults = {k: v.default for k, v in signature.parameters.items()
                    if v.default is not inspect.Parameter.empty}
        _fields(params, defaults, where)
        for key, value in params.items():
            default = defaults[key]
            if isinstance(default, bool):
                valid = type(value) is bool
            elif isinstance(default, int):
                valid = type(value) is int
            elif isinstance(default, float):
                valid = type(value) in (int, float) and math.isfinite(value)
            elif isinstance(default, tuple):
                valid = isinstance(value, (list, tuple)) and len(value) > 0
            else:
                valid = isinstance(value, type(default))
            _check(valid, f"{where}.{key}: expected {type(default).__name__}")
            choices = (self.choices or {}).get(key)
            if choices is not None:
                values = value if isinstance(default, tuple) else [value]
                _check(all(v in choices for v in values), f"{where}.{key}: allowed values {choices}")
            if key.endswith("_axis") and isinstance(default, int):
                _check(0 <= value <= 2, f"{where}.{key}: must be 0, 1 or 2")
            if key.endswith("_pct"):
                _check(0 <= value <= 100, f"{where}.{key}: must be in [0, 100]")
            if key.endswith(("_frac", "_score", "_threshold")):
                _check(0 <= value <= 1, f"{where}.{key}: must be in [0, 1]")
            if key in ("panel_frac", "slab_frac", "stem_width_ratio", "contact_frac",
                       "lower_frac", "max_lower_mass", "end_frac"):
                _check(type(value) in (int, float) and 0 < value < 1,
                       f"{where}.{key}: must be in (0, 1)")
            if key == "depth_quantile":
                _check(0 < value < 0.5, f"{where}.{key}: must be in (0, 0.5)")
            if key in ("min_depth_separation", "min_up_alignment"):
                _check(0 <= value <= 1, f"{where}.{key}: must be in [0, 1]")
            if key == "base_expansion":
                _check(value > 1, f"{where}.{key}: must exceed 1")
            if key == "min_points":
                _check(value >= 3, f"{where}.{key}: must be at least 3")
            if key in ("foot_bands", "shoulder_bands"):
                _check(all(type(v) in (int, float) and 0 < v < 0.8 for v in value),
                       f"{where}.{key}: expected fractions in (0, 0.8)")
                _check(all(a < b for a, b in zip(value, value[1:])),
                       f"{where}.{key}: must be strictly increasing")
        values = dict(defaults, **params)
        if "fwd_axis" in values and "up_axis" in values:
            _check(values["fwd_axis"] != values["up_axis"], f"{where}: forward and up axes must differ")
        if "foot_bands" in values and "shoulder_bands" in values:
            _check(min(values["foot_bands"]) + 0.03 < max(values["shoulder_bands"]),
                   f"{where}: no shoulder band leaves room above a foot band")
        if "contact_frac" in values and "lower_frac" in values:
            _check(values["contact_frac"] < values["lower_frac"],
                   f"{where}: contact_frac must be smaller than lower_frac")
        # Tuples prevent a caller mutating a compiled recipe through its source
        # document after validation.
        return partial(self.function, **{k: tuple(v) if isinstance(v, list) else v
                                        for k, v in params.items()})


@dataclass(frozen=True)
class Recipe:
    steps: tuple

    def __call__(self, shape):
        result = self.steps[0](shape)
        for step in self.steps[1:]:
            result = step(shape, *result)
        return result


class RuleDatabase:
    """An independent snapshot; construct from JSON or DB-decoded records."""

    @classmethod
    def load(cls, path, operations):
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"),
                                  object_pairs_hook=_unique_object)
        except json.JSONDecodeError as exc:
            raise RuleValidationError(f"{path}: {exc}") from exc
        return cls(document, operations)

    def __init__(self, document, operations):
        document = deepcopy(document)
        _fields(document, ("schema_version", "recipes", "classes", "fallback"), "database",
                ("schema_version", "recipes", "classes", "fallback"))
        _check(type(document["schema_version"]) is int and document["schema_version"] == 1,
               "unsupported schema_version; expected 1")
        _check(isinstance(document["recipes"], dict) and document["recipes"], "recipes must be a nonempty object")
        recipes = {}
        for name, steps in document["recipes"].items():
            _check(isinstance(steps, list) and steps, f"recipe {name}: expected nonempty steps")
            compiled = []
            for i, step in enumerate(steps):
                where = f"recipe {name}, step {i}"
                _fields(step, ("op", "params"), where, ("op",))
                _check(isinstance(step["op"], str) and step["op"] in operations, f"{where}: unknown operation {step['op']}")
                operation = operations[step["op"]]
                _check(operation.modifier == (i > 0), f"{where}: first step must build a frame; later steps must modify it")
                compiled.append(operation.compile(step.get("params", {}), where))
            recipes[name] = Recipe(tuple(compiled))
        _check(isinstance(document["fallback"], str) and document["fallback"] in recipes, "unknown fallback recipe")
        self.fallback = recipes[document["fallback"]]
        records, aliases, synsets, rules = {}, {}, {}, {}
        _check(isinstance(document["classes"], list) and document["classes"], "classes must be a nonempty list")
        groups = {"I", "C2x", "C2y", "C2z", "C3z", "C4z", "C6z", "Cinfz", "D2"}
        for record in document["classes"]:
            _fields(record, ("name", "recipe", "symmetry", "aliases", "synset", "reference"), "class",
                    ("name", "recipe", "symmetry"))
            name = record["name"]
            _check(isinstance(name, str) and name and normalise_label(name) == name, "class name must be normalized")
            _check(name not in records, f"duplicate class {name}")
            _check(isinstance(record["recipe"], str) and record["recipe"] in recipes, f"{name}: unknown recipe")
            _check(isinstance(record["symmetry"], str) and record["symmetry"] in groups, f"{name}: unsupported symmetry")
            names = record.get("aliases", [])
            _check(isinstance(names, list) and all(isinstance(a, str) and a.strip() for a in names), f"{name}: invalid aliases")
            for alias in [name, *names]:
                alias = normalise_label(alias)
                _check(alias not in aliases, f"duplicate class/alias {alias}")
                aliases[alias] = name
            synset = record.get("synset")
            if synset is not None:
                _check(isinstance(synset, str) and len(synset) == 8 and synset.isdigit(), f"{name}: synset must be an 8-digit string")
                _check(synset not in synsets, f"duplicate synset {synset}")
                synsets[synset] = name
            ref = record.get("reference", {})
            _fields(ref, ("enabled", "refine", "clusters", "preserve_semantics", "lock_confidence", "min_improvement", "recheck_symmetry"), f"{name}.reference")
            for key in ("enabled", "refine", "preserve_semantics"):
                _check(type(ref.get(key, False)) is bool, f"{name}.reference.{key}: expected boolean")
            _check(type(ref.get("clusters", 1)) is int and ref.get("clusters", 1) >= 1, f"{name}: clusters must be positive")
            for key in ("lock_confidence", "min_improvement"):
                value = ref.get(key, 0.0)
                _check(type(value) in (int, float) and 0 <= value <= 1, f"{name}.{key}: expected number in [0, 1]")
            if "recheck_symmetry" in ref:
                operations["axial_symmetry"].compile(ref["recheck_symmetry"], f"{name}.recheck_symmetry")
            frozen_ref = dict(ref)
            if "recheck_symmetry" in ref:
                frozen_ref["recheck_symmetry"] = MappingProxyType({
                    k: tuple(v) if isinstance(v, list) else v
                    for k, v in ref["recheck_symmetry"].items()})
            records[name] = MappingProxyType(dict(record, reference=MappingProxyType(frozen_ref), aliases=tuple(names)))
            rules[name] = recipes[record["recipe"]]
        self.classes = MappingProxyType(records)
        self.aliases = MappingProxyType(aliases)
        self.synsets = MappingProxyType(synsets)
        self.rules = MappingProxyType(rules)
        self.fingerprint = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def canonical_name(self, label):
        name = normalise_label(label)
        return self.aliases.get(name, name)

    def reference_policy(self, label):
        return self.classes.get(self.canonical_name(label), {}).get("reference", {})
