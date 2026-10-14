"""Video-MME evaluation of frame-selection conditions with one VLLM.

Every condition uses the same decoder, prompt template, parser and frame budget; only frame selection
and the presence of timestamps differ. The model is loaded once.

  # inspect exactly what would be sent to the model (no VLLM needed):
  python scripts/run_videomme.py --model qwen2vl --dry-run --limit 3 ...
  # run:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_videomme.py --model qwen2vl --parquet P --videos V --out runs/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import time
from importlib import metadata

from maxinfo import FrameEncoder, MaxInfo, MaxInfoConfig
from maxinfo.eval import load_videomme, run_mcq, score
from maxinfo.models import build_prompt, make_answer_fn, make_frames_fn

MODELS = {
    "qwen2vl": ("Qwen/Qwen2-VL-2B-Instruct", 2),  # (default checkpoint, temporal group size)
    "llava": ("lmms-lab/LLaVA-Video-7B-Qwen2", 1),
}


def conditions(group_size: int) -> dict:
    """name -> (MaxInfoConfig or None for uniform, use_time_instruction)."""
    return {
        "uniform": (None, False),
        "uniform_time": (None, True),
        # What the paper describes (tol 0.3, rank 8, even subsampling of pivots), no timestamps, no pairing.
        "paper_intended": (MaxInfoConfig(rank=8, tol=0.3, budget_mode="uniform", group_size=1), False),
        "greedy_time": (MaxInfoConfig(rank=8, tol=0.3, budget_mode="greedy", group_size=group_size), True),
        "chunked_time": (MaxInfoConfig(rank=8, tol=0.3, n_chunks=8, group_size=group_size), True),
    }


def last_video_cache(fn):
    """Selection is question-agnostic and Video-MME lists a video's 3 questions consecutively."""
    state = {"key": None, "val": None}

    def wrapped(item):
        if state["key"] != item.video_path:
            state["key"], state["val"] = item.video_path, fn(item)
        return state["val"]
    return wrapped


def manifest(args, conds) -> dict:
    def ver(p):
        try:
            return metadata.version(p)
        except metadata.PackageNotFoundError:
            return None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:
        commit = None
    return {"args": vars(args), "git_commit": commit, "python": platform.python_version(),
            "packages": {p: ver(p) for p in ["torch", "transformers", "qwen-vl-utils", "llava", "decord", "numpy"]},
            "conditions": {k: {"maxinfo": dataclasses.asdict(c) if c else "uniform", "time_instruction": t}
                           for k, (c, t) in conds.items()},
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--candidates", type=int, default=128)
    ap.add_argument("--encoder-device", default=None)
    ap.add_argument("--llava-vision-tower", default=None,
                    help="local SigLIP stub dir (see scripts/verify_llava_weights.py); verify before use")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only")
    ap.add_argument("--dry-run", action="store_true", help="print selections and prompts, do not load the VLLM")
    args = ap.parse_args()

    model_path, group = MODELS[args.model]
    args.model_path = args.model_path or model_path
    conds = conditions(group)
    if args.conditions:
        unknown = set(args.conditions) - set(conds)
        if unknown:
            raise SystemExit(f"unknown conditions {unknown}; choose from {list(conds)}")
        conds = {k: conds[k] for k in args.conditions}

    items = load_videomme(args.parquet, args.videos, available_only=True)
    if args.limit:
        items = items[: args.limit]
    out_dir = os.path.join(args.out, args.model)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json" if not args.dry_run else "manifest.dryrun.json"), "w") as f:
        json.dump(manifest(args, conds), f, indent=2)

    encoder = FrameEncoder(device=args.encoder_device) if any(c for c, _ in conds.values()) else None
    feature_cache: dict = {}  # embeddings are ~128 x 768 floats per video; shared by all MaxInfo conditions
    frames_fns = {k: last_video_cache(make_frames_fn(MaxInfo(encoder, c, feature_cache=feature_cache) if c else None,
                                                     args.budget,
                                                     num_candidates=args.candidates if c else None))
                  for k, (c, _) in conds.items()}

    if args.dry_run:
        print(f"{len(items)} questions, budget {args.budget}, candidates {args.candidates}, model {args.model_path}")
        for k, (c, use_time) in conds.items():
            print(f"\n=== {k}: {dataclasses.asdict(c) if c else 'uniform'} | time_instruction={use_time}")
            for i, item in enumerate(items):
                v = frames_fns[k](item)
                print(f"  {item.qid} [{item.meta.get('duration')}] {len(v.indices)} frames, "
                      f"t = {', '.join(f'{t:.1f}' for t in v.timestamps[:12])}{' ...' if len(v.indices) > 12 else ''}")
                if i == 0:
                    print("  --- prompt (text part) ---\n  " + build_prompt(item, v, use_time_instruction=use_time)
                          .replace("\n", "\n  ")[:1500] + "\n  --- gold: " + str(item.answer))
        return

    if args.model == "qwen2vl":
        from maxinfo.models import load_qwen2_vl
        generate = load_qwen2_vl(args.model_path)
    else:
        from maxinfo.models import load_llava_video
        generate = load_llava_video(args.model_path, vision_tower=args.llava_vision_tower)

    for k, (c, use_time) in conds.items():
        path = os.path.join(out_dir, f"{k}.jsonl")
        t0 = time.time()
        run_mcq(items, make_answer_fn(generate, frames_fns[k], use_time_instruction=use_time), path)
        s = score(path)["overall"]
        print(f"[{k}] acc={s['accuracy']:.3f} n={s['n']} errors={s['errors']} unparsable={s['unparsable']} "
              f"({time.time() - t0:.0f}s)", flush=True)

    summary = {k: score(os.path.join(out_dir, f"{k}.jsonl"), by="duration") for k in conds}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: round(v["overall"]["accuracy"], 3) for k, v in summary.items()}))


if __name__ == "__main__":
    main()
