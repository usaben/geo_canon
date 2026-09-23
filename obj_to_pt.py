"""Convert ShapeNet .obj meshes into sampled point-cloud .pt files.

Reads model_normalized.obj entries directly out of shapenet_subset.zip (no
extraction to disk needed) and writes a (NUM_POINTS, 3) float32 tensor to
processed_data/<class>/<class>_<n>.pt, numbering each class from one.  The
model id the cloud came from is the only way back to the mesh it was sampled
off, so it is kept in processed_data/manifest.csv rather than in the filename.

geo_canon reads .obj directly now, so this is for pre-sampling a whole set once
instead of resampling it on every run.
"""

import csv
import io
import zipfile
import trimesh
import numpy as np
import torch
from pathlib import Path

import geo_canon as g          # for SYNSET_TO_CLASS

ZIP_PATH = Path("shapenet_subset.zip")
OUTPUT_ROOT = Path("processed_data")
NUM_POINTS = 1024
SEED = 42


def load_mesh_from_bytes(obj_bytes):
    mesh = trimesh.load(io.BytesIO(obj_bytes), file_type="obj", process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    return mesh


def obj_bytes_to_pointcloud(obj_bytes, n_points, seed):
    mesh = load_mesh_from_bytes(obj_bytes)
    np.random.seed(seed)
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points.astype(np.float32)


def cat_and_model_id(name):
    parts = Path(name).parts
    if parts[-2] == "models":
        # raw ShapeNet layout: <cat>/<model_id>/models/model_normalized.obj
        return parts[-4], parts[-3]
    # flattened layout: <cat>/<model_id>/model_normalized.obj
    return parts[-3], parts[-2]


def convert_all(zip_path=ZIP_PATH, output_root=OUTPUT_ROOT, num_points=NUM_POINTS, seed=SEED, limit=None):
    count, rows = 0, []
    n_of = {}                                  # next free number per class
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith("model_normalized.obj"))
        if limit is not None:
            names = names[:limit]
        for name in names:
            synset, model_id = cat_and_model_id(name)
            cls = g.SYNSET_TO_CLASS.get(synset, synset)
            out_dir = output_root / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            if cls not in n_of:                # carry on from whatever is there
                n_of[cls] = 1 + sum(1 for _ in out_dir.glob(f"{cls}_*.pt"))

            try:
                points = obj_bytes_to_pointcloud(zf.read(name), num_points, seed + count)
            except Exception as e:
                print(f"skip {name}: {e}")
                continue

            fname = f"{cls}_{n_of[cls]}.pt"
            torch.save(torch.from_numpy(points), out_dir / fname)
            rows.append((cls, fname, f"{model_id}.pt", synset))
            n_of[cls] += 1
            count += 1

    manifest = output_root / "manifest.csv"
    fresh = not manifest.is_file()
    with open(manifest, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if fresh:
            w.writerow(["class", "file", "source_id", "source_folder"])
        w.writerows(rows)

    print(f"Converted {count} .obj files to .pt point clouds under {output_root}")
    return count


if __name__ == "__main__":
    convert_all()
