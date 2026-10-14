"""Select key frames from one video and compare with uniform sampling.

python examples/quickstart.py path/to/video.mp4 --candidates 128 --budget 32
"""

import argparse

from maxinfo import FrameEncoder, MaxInfo, MaxInfoConfig, uniform_baseline
from maxinfo.prompts import time_instruction

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--candidates", type=int, default=128)
ap.add_argument("--budget", type=int, default=32)
ap.add_argument("--tol", type=float, default=0.3)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--chunks", type=int, default=None)
args = ap.parse_args()

mi = MaxInfo(FrameEncoder(), MaxInfoConfig(rank=args.rank, tol=args.tol, n_chunks=args.chunks))
out = mi.from_video(args.video, num_candidates=args.candidates, budget=args.budget)
uni = uniform_baseline(args.video, args.budget)

print(f"candidates: {len(out.candidates.indices)} | MaxVol picked {out.selection.num_before_budget} "
      f"| kept {len(out.video.indices)} (budget {args.budget})")
print("MaxInfo timestamps:", ", ".join(f"{t:.1f}" for t in out.video.timestamps))
print("Uniform timestamps:", ", ".join(f"{t:.1f}" for t in uni.timestamps))
print("\nPrompt prefix for the VLLM:\n" + time_instruction(out.video.duration, out.video.timestamps)[:300] + " ...")
