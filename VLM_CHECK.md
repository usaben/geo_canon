# Part check and vision-model check

Two optional rule steps sit on top of the newb rules. Both only choose among
the 24 axis-aligned turns of the rule's frame, so they never change its
precision, and both pass the symmetry group through unchanged, so scoring
stays exactly as before. Without `--rules`, nothing changes.

| step | file | what it uses |
|---|---|---|
| `part_check` | `part_cues.py` | PatchAlign3D part labels (CPU, weights fetched on first use) |
| `vlm_check` | `vlm_cues.py` | a vision-language model's yes-probability for "is this <class> upright?" |

`rules_parts.json` switches `part_check` on for airplane and table.

## vlm_check on a GPU box (vLLM)

```bash
pip install vllm
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --max-model-len 8192   # full precision, ~17 GB of VRAM
export GEOCANON_VLM_URL=http://localhost:8000/v1
export GEOCANON_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

With Ollama instead, which is the default, the settings are
`GEOCANON_VLM_URL=http://127.0.0.1:11434/v1` and
`GEOCANON_VLM_MODEL=qwen3-vl:8b-instruct`.

1. **Pilot:** find which classes it helps. Tune on half A and confirm on half B.

   ```bash
   python vlm_pilot.py --rules rules_parts.json --half A --out pilot_A.json
   python vlm_pilot.py --rules rules_parts.json --half B --out pilot_B.json
   ```

   Keep a class only if, at the same margin, it fixes more than it breaks on
   both halves and breaks nothing on half A.

2. **Switch it on** for those classes:

   ```bash
   python rules_tool.py --base rules_parts.json --op vlm_check \
       --classes chair,monitor --params '{"margin": 2.0}' --out rules_vlm.json
   ```

3. **Score it** with the unchanged report. Answers are cached in `vlm_cache/`,
   so repeated runs are fast.

   ```bash
   python class_report.py --rules rules_vlm.json --instances 100 --rotations 6 \
       --refs none --no-figures --out results_vlm.txt
   ```

If the server can't be reached, `vlm_check` returns the rule's frame unchanged,
so the output is plain newb.
