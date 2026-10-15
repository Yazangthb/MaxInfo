"""Check that a loaded LLaVA-Video model holds exactly the checkpoint's weights.

Needed when the vision tower is built from a weightless stub (see load_llava_video_model): every
parameter must then come from the checkpoint. Compares *all* model parameters with the safetensors
shards and fails on any mismatch, any parameter missing from the checkpoint, or any meta tensor.

  python scripts/verify_llava_weights.py --vision-tower data/siglip_stub
  python scripts/verify_llava_weights.py --make-stub data/siglip_stub   # create the stub only
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def make_stub(path: str):
    """config.json of google/siglip-so400m-patch14-384 + an empty safetensors file."""
    import torch  # noqa: F401  (safetensors.torch needs it)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    cfg = hf_hub_download("google/siglip-so400m-patch14-384", "config.json")
    with open(cfg) as f, open(os.path.join(path, "config.json"), "w") as g:
        json.dump(json.load(f), g, indent=2)
    save_file({}, os.path.join(path, "model.safetensors"))
    print(f"stub written to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="lmms-lab/LLaVA-Video-7B-Qwen2")
    ap.add_argument("--vision-tower", default=None)
    ap.add_argument("--make-stub", default=None)
    args = ap.parse_args()
    if args.make_stub:
        make_stub(args.make_stub)
        return

    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    from maxinfo.models import load_llava_video_model

    _, model, _ = load_llava_video_model(args.model_path, vision_tower=args.vision_tower)
    ckpt_dir = snapshot_download(args.model_path, allow_patterns=["*.json"])
    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    params = dict(model.named_parameters())
    missing = [n for n in params if n not in weight_map]
    meta = [n for n, p in params.items() if p.is_meta]
    by_shard: dict = {}
    for n in params:
        if n in weight_map:
            by_shard.setdefault(weight_map[n], []).append(n)
    mismatched, checked = [], 0
    for shard, names in by_shard.items():
        with safe_open(os.path.join(ckpt_dir, shard), framework="pt") as f:
            for n in names:
                ref = f.get_tensor(n)
                got = params[n].detach().to("cpu", ref.dtype)
                if n in ("model.embed_tokens.weight", "lm_head.weight") and got.shape[1:] == ref.shape[1:]:
                    # The builder resizes the vocabulary to len(tokenizer) (152064 -> 151647 for
                    # LLaVA-Video-Qwen2; the dropped rows are padding). Compare the shared rows.
                    k = min(got.shape[0], ref.shape[0])
                    got, ref = got[:k], ref[:k]
                if got.shape != ref.shape or not torch.equal(got, ref):
                    mismatched.append(n)
                checked += 1
    vt = [n for n in params if "vision_tower" in n]
    unused_ckpt = [k for k in weight_map if k not in params]
    print(f"parameters: {len(params)} (vision tower: {len(vt)}), compared with checkpoint: {checked}")
    print(f"mismatched: {len(mismatched)} {mismatched[:5]}")
    print(f"not in checkpoint: {len(missing)} {missing[:5]}")
    print(f"meta tensors: {len(meta)} {meta[:5]}")
    print(f"checkpoint keys not in model: {len(unused_ckpt)} {unused_ckpt[:5]}")
    ok = not (mismatched or missing or meta)
    print("OK: model weights equal the checkpoint" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
