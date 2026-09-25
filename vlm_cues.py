"""Upright evidence from a vision-language model (vLLM or Ollama).

`vlm_check` is a rule step like `part_check`.  For the rule-canonical cloud it
renders the six ways the object could stand -- each signed axis of the rule
frame turned to point up -- from four sides, and asks the model, one picture
at a time, "is this <class> upright?".  The score of a picture is
log P(yes) - log P(no) read from the answer's token log-probabilities, so no
A/B ordering exists for the model to be biased by.  The rule's own frame is
replaced only when another way up beats it by `margin`.

Only the up direction is decided here; the replacement is the smallest
axis-aligned turn that brings the chosen axis up, so the rule's forward is kept
whenever the new up allows it.  The symmetry group is passed through untouched.

Any server with an OpenAI-compatible /v1/chat/completions endpoint that returns
logprobs works:

    vLLM    vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
            GEOCANON_VLM_URL=http://localhost:8000/v1
            GEOCANON_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
    Ollama  (default) GEOCANON_VLM_URL=http://127.0.0.1:11434/v1
            GEOCANON_VLM_MODEL=qwen3-vl:8b-instruct

If the server cannot be reached the step returns the rule's frame unchanged,
so the pipeline degrades to plain newb instead of failing.  Answers are cached
on disk (vlm_cache/) keyed by model, prompt and the rounded canonical cloud,
which is pose-invariant: all input poses of one object share one answer.
"""
import base64
import hashlib
import io
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_CACHE = Path(os.environ.get("GEOCANON_VLM_CACHE", _ROOT / "vlm_cache"))
PROMPT_VERSION = "up4-v1"
SIDE_VIEWS = ((15.0, -60.0), (15.0, 30.0), (15.0, 120.0), (15.0, 210.0))
UP_PROMPT = ("The image shows one object, a {name}, rendered as a 3D point cloud from "
             "four viewpoints around it. The vertical direction is the same in all four "
             "views. Is the {name} upright, the way it normally stands or is placed, "
             "rather than upside down or lying on its side? Answer yes or no.")
_DOWN = set()                      # servers found unreachable in this process


def _cfg():
    return (os.environ.get("GEOCANON_VLM_URL", "http://127.0.0.1:11434/v1").rstrip("/"),
            os.environ.get("GEOCANON_VLM_MODEL", "qwen3-vl:8b-instruct"),
            float(os.environ.get("GEOCANON_VLM_TIMEOUT", "300")))


def _geo():
    g = sys.modules.get("geo_canon")
    if g is None:
        import geo_canon as g
    return g


def _up_turns():
    """For each signed axis of the rule frame (identity first), the turn
    closest to the identity that makes that axis the new up."""
    g = _geo()
    out = []
    for axis in (2, 0, 1):
        for sign in (1.0, -1.0):
            cand = [O for O in g.OCTAHEDRAL if abs(O[2, axis] - sign) < 1e-9]
            out.append(max(cand, key=lambda O: np.trace(O)))
    return out


def render_sheet(P, lim, px=768):
    """Four side views of cloud P (z up) in one PNG, with geo_canon's own
    shaded-splat renderer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = _geo()
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    fig.patch.set_facecolor("white")
    axes = fig.subplots(2, 2).ravel()
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.02, hspace=0.02)
    for ax, (e, a) in zip(axes, SIDE_VIEWS):
        g.render_cloud(ax, fig, P, elev=e, azim=a, shadow=False, lim=lim)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _shared_lim(clouds, floor=1.2, pad=1.04):
    g = _geo()
    need = 0.0
    for C in clouds:
        for e, a in SIDE_VIEWS:
            right, up, _ = g.camera_basis(e, a)
            u, v = C @ right, C @ up
            need = max(need, np.abs(u).max(), -v.min() / 0.95, v.max() / 1.05)
    return max(floor, pad * need)


def yes_score(prompt, png):
    """log P(yes) - log P(no) of the first answer token."""
    url, model, timeout = _cfg()
    body = {"model": model, "max_tokens": 1, "temperature": 0, "seed": 0,
            "logprobs": True, "top_logprobs": 20,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}}]}]}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    tops = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    p_yes = sum(math.exp(t["logprob"]) for t in tops if t["token"].strip().lower() == "yes")
    p_no = sum(math.exp(t["logprob"]) for t in tops if t["token"].strip().lower() == "no")
    return math.log(p_yes + 1e-9) - math.log(p_no + 1e-9)


def up_score(P, name, lim):
    url, model, _ = _cfg()
    prompt = UP_PROMPT.format(name=name)
    h = hashlib.sha256(f"{model}|{PROMPT_VERSION}|{prompt}|{lim:.3f}".encode())
    h.update(np.round(np.asarray(P, float), 3).tobytes())
    f = _CACHE / (h.hexdigest() + ".json")
    if f.exists():
        return json.loads(f.read_text())["score"]
    s = yes_score(prompt, render_sheet(P, lim))
    _CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"score": s, "model": model, "prompt": PROMPT_VERSION}))
    return s


def vlm_check(shape, R, info, name="object", margin=2.0):
    """Replace the rule's up by the way up the model finds most upright, if it
    beats the rule's own by more than `margin` (log-odds units)."""
    url, _, _ = _cfg()
    if url in _DOWN:
        return R, dict(info, vlm_check={"error": "server unavailable"})
    try:
        P = shape.X @ R.T
        turns = _up_turns()
        cands = [P @ O.T for O in turns]
        lim = _shared_lim(cands)
        scores = [up_score(C, name.replace("_", " "), lim) for C in cands]
    except OSError as exc:                            # server down: behave like newb
        _DOWN.add(url)
        return R, dict(info, vlm_check={"error": f"server unavailable: {exc}"})
    except Exception as exc:                          # never lose a cloud to this step
        return R, dict(info, vlm_check={"error": f"{type(exc).__name__}: {exc}"})
    k = int(np.argmax(scores))
    trace = {"scores": [round(s, 2) for s in scores], "best": k,
             "gain": round(scores[k] - scores[0], 2), "switched": False}
    if k != 0 and scores[k] - scores[0] > margin:
        trace["switched"] = True
        return turns[k] @ R, dict(info, vlm_check=trace)
    return R, dict(info, vlm_check=trace)
