"""Cache frame embeddings for a folder of videos (input for compare_official.py).

python scripts/extract_features.py VIDEO_DIR OUT_DIR --num-frames 128
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

from maxinfo.encoders import DEFAULT_ENCODER, FrameEncoder
from maxinfo.video import sample_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--num-frames", type=int, default=128)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    enc = FrameEncoder(args.encoder, device=args.device, batch_size=args.batch_size)
    videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    for path in videos:
        out = os.path.join(args.out_dir, f"{os.path.splitext(os.path.basename(path))[0]}_{args.num_frames}.npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        v = sample_video(path, num_frames=args.num_frames)
        feats = enc(v.pil())
        np.savez(out, feats=feats, idx=v.indices, timestamps=v.timestamps, fps=v.native_fps, duration=v.duration)
        print(f"{os.path.basename(path)}: {len(v.indices)} frames, {v.duration:.0f}s video, {time.time() - t0:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
