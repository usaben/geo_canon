"""
uni3d_cpu.py -- run Uni3D-S on the CPU, without the CUDA extension.

The official repo asks for pointnet2_ops, a CUDA C++ extension, which needs nvcc
to build and an NVIDIA device to run.  It uses exactly two symbols from it --
furthest_point_sample and gather_operation -- and both are a few lines of plain
PyTorch, so we register a stand-in under that name before importing the repo's
encoder.  Nothing in the Uni3D source tree is edited; it is imported as cloned.

The other thing the README makes you download is EVA02-E-14-plus (10.1 GB) to
turn class names into text embeddings.  We do not need it: with labelled clouds
already on disk, a class centroid in Uni3D's own embedding space classifies just
as well, and the Uni3D module itself never touches CLIP -- encode_pc() stands
alone.

Config mirrors scripts/inference.sh for the `small` scale.

    import uni3d_cpu
    model = uni3d_cpu.load()
    feats = uni3d_cpu.encode(model, clouds)      # (B, 1024), L2-normalised
"""

import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

UNI3D_ROOT = Path(os.environ.get("UNI3D_ROOT", Path.home() / "Uni3D"))
CKPT_REPO, CKPT_FILE = "BAAI/Uni3D", "modelzoo/uni3d-s/model.pt"

# scripts/inference.sh, the `small` branch plus the flags shared by every scale.
# embed_dim is 1024 because all Uni3D scales are aligned to the same
# EVA02-E-14-plus space -- the point tower is small, the space it lands in is not.
CFG = dict(pc_model="eva02_small_patch14_224", pretrained_pc="", pc_feat_dim=384,
           embed_dim=1024, group_size=64, num_group=512, pc_encoder_dim=512,
           patch_dropout=0.0, drop_path_rate=0.0)

# Colourless clouds get a flat 0.4 fill.  That is not an arbitrary grey: Uni3D's
# training loop replaces real colour with exactly ones*0.4 half the time
# (data/datasets.py, rgb_random_drop_prob = 0.5), so it is a value the encoder
# has seen constantly rather than one it has to extrapolate to.
GREY = 0.4


# ---------------------------------------------------------------------------
# the stand-in for the CUDA extension
# ---------------------------------------------------------------------------

def _furthest_point_sample(xyz, npoint):
    """(B, N, 3) -> (B, npoint) int32, the indices of a farthest-point subset.

    Matches the CUDA kernel's convention of seeding with point 0, so the centres
    come out in the same order the pretrained weights were fitted against.
    """
    xyz = xyz[..., :3].contiguous()
    B, N, _ = xyz.shape
    dev = xyz.device
    idx = torch.zeros(B, npoint, dtype=torch.long, device=dev)
    dist = torch.full((B, N), float("inf"), dtype=xyz.dtype, device=dev)
    far = torch.zeros(B, dtype=torch.long, device=dev)
    rows = torch.arange(B, device=dev)
    for i in range(npoint):
        idx[:, i] = far
        d = ((xyz - xyz[rows, far].unsqueeze(1)) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(-1)
    return idx.int()


def _gather_operation(features, idx):
    """(B, C, N) gathered along N by (B, S) int32 indices -> (B, C, S)."""
    idx = idx.long()
    B, C, _ = features.shape
    return torch.gather(features, 2, idx.unsqueeze(1).expand(B, C, idx.shape[1])).contiguous()


def _install_shim():
    if "pointnet2_ops" in sys.modules:
        return
    utils = types.ModuleType("pointnet2_ops.pointnet2_utils")
    utils.furthest_point_sample = _furthest_point_sample
    utils.gather_operation = _gather_operation
    pkg = types.ModuleType("pointnet2_ops")
    pkg.pointnet2_utils = utils
    sys.modules["pointnet2_ops"] = pkg
    sys.modules["pointnet2_ops.pointnet2_utils"] = utils


def _install_easydict():
    """Supply easydict if the installed copy is out of reach.

    Uni3D's PointcloudEncoder.__init__ imports it, so it cannot simply be
    dropped -- but it is a dict with attribute access and nothing more.  On this
    machine the real package sits in the per-user site-packages, a path that
    several launch contexts skip (-s, -E, PYTHONNOUSERSITE, an elevated shell, a
    fresh venv), and the demo should not die on a two-line dependency.
    """
    try:
        import easydict  # noqa: F401
        return
    except ImportError:
        pass

    class EasyDict(dict):
        def __init__(self, d=None, **kw):
            super().__init__()
            for k, v in dict(d or {}, **kw).items():
                self[k] = v

        def __setitem__(self, k, v):
            if isinstance(v, dict) and not isinstance(v, EasyDict):
                v = EasyDict(v)
            elif isinstance(v, (list, tuple)):
                v = type(v)(EasyDict(x) if isinstance(x, dict) else x for x in v)
            super().__setitem__(k, v)
            super().__setattr__(k, v)

        __setattr__ = __setitem__

        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

    mod = types.ModuleType("easydict")
    mod.EasyDict = EasyDict
    sys.modules["easydict"] = mod


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def _checkpoint_path(ckpt=None):
    if ckpt:
        return str(ckpt)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(CKPT_REPO, CKPT_FILE)


def load(ckpt=None, root=None):
    """Build Uni3D-S and load the pretrained weights.  Returns an eval-mode model."""
    _install_shim()
    _install_easydict()
    root = Path(root or UNI3D_ROOT)
    if not (root / "models" / "uni3d.py").is_file():
        raise FileNotFoundError(
            f"Uni3D source not found at {root}. Clone it, or set UNI3D_ROOT.")
    # appended, not prepended: the Uni3D tree has modules called utils/ and data/,
    # and those names are generic enough to shadow something of yours otherwise.
    if str(root) not in sys.path:
        sys.path.append(str(root))

    import importlib
    from easydict import EasyDict

    # models/uni3d.py opens with `from . import losses`, and that one line pulls
    # the whole training stack in behind it -- h5py, open_clip, DeepSpeed.  The
    # only thing it is wanted for is a loss factory we never call at inference,
    # so a stub in sys.modules satisfies the import and stops the cascade.
    importlib.import_module("models")
    if "models.losses" not in sys.modules:
        stub = types.ModuleType("models.losses")
        sys.modules["models.losses"] = stub
        setattr(sys.modules["models"], "losses", stub)
    uni3d_mod = importlib.import_module("models.uni3d")

    model = uni3d_mod.create_uni3d(EasyDict(CFG))

    # a DeepSpeed checkpoint: the weights sit under "module", alongside optimiser state
    blob = torch.load(_checkpoint_path(ckpt), map_location="cpu", weights_only=False)
    sd = blob.get("module", blob.get("state_dict", blob))
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: {len(missing)} missing "
                           f"{missing[:4]}, {len(unexpected)} unexpected {unexpected[:4]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def pc_norm(xyz):
    """Centre on the mean, scale so the furthest point sits at radius 1.

    This is Uni3D's own pc_norm (data/datasets.py), reproduced rather than reused
    so the scale the encoder sees is the one it was trained on -- and so it stays
    independent of however the canonicalisation pipeline normalises.
    """
    xyz = np.asarray(xyz, dtype=np.float32)[:, :3]
    xyz = xyz - xyz.mean(0)
    return xyz / max(float(np.sqrt((xyz ** 2).sum(1)).max()), 1e-9)


def pca_frame(xyz):
    """Rotate a cloud into its principal-axis frame, signs fixed by third moment.

    This is geo_canon's pca_baseline, reimplemented here so uni3d_cpu stays
    free of that import.  It matters because Uni3D has no rotation invariance:
    naming a randomly turned cloud is markedly worse than naming the same cloud
    after its arbitrary attitude has been taken out, and PCA can do that without
    knowing the class -- which is the whole point, since the class is what we
    are trying to find.  Degenerate eigenvalues make it imperfect, not useless.
    """
    Y = np.asarray(xyz, dtype=np.float64)[:, :3]
    Y = Y - Y.mean(0)
    _, V = np.linalg.eigh(np.cov(Y.T))
    A = V.T[::-1].copy()                     # rows = axes, largest first
    for i in range(3):
        if float(((Y @ A[i]) ** 3).mean()) < 0:
            A[i] *= -1
    if np.linalg.det(A) < 0:
        A[2] *= -1
    return (Y @ A.T).astype(np.float32)


@torch.no_grad()
def encode(model, clouds, swap_yz=False, batch=8, normalise=True, framed=False):
    """Embed clouds into Uni3D's 1024-d space.

    clouds:   one (N, >=3) array, or a sequence of them.  Only xyz is read; any
              further channels are ignored, because Uni3D reads channels 3:6 as
              RGB and your files carry unit normals there -- feeding those in
              would put surface directions where colour belongs.
    swap_yz:  exchange the y and z axes first.  Uni3D's eval path does this for
              OpenShape-convention data, and since the encoder is not rotation
              invariant the up-axis convention is worth testing both ways.
    """
    single = isinstance(clouds, np.ndarray) and clouds.ndim == 2
    if single:
        clouds = [clouds]

    prepped = []
    for X in clouds:
        X = np.asarray(X, dtype=np.float32)
        if framed:
            X = pca_frame(X)                 # before pc_norm: framing is a rotation
        X = pc_norm(X)
        if swap_yz:
            X = X[:, [0, 2, 1]]
        prepped.append(X)

    out = []
    for i in range(0, len(prepped), batch):
        xyz = torch.from_numpy(np.stack(prepped[i:i + batch])).float()
        rgb = torch.full_like(xyz, GREY)
        f = model.encode_pc(torch.cat((xyz, rgb), dim=-1))
        if normalise:
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        out.append(f)
    feats = torch.cat(out).numpy()
    return feats[0] if single else feats
