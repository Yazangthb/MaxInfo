"""Download selected Video-MME videos without downloading the 20 x 5 GB zip archives.

Uses HTTP range requests (remotezip) to read each archive's index and extract only the wanted
members. Files already present with the right size are skipped; partial files are replaced.

  python scripts/fetch_videomme_videos.py OUT_DIR --ids ZBKUqc_ICpg zKyWRRJQbkM ...
  python scripts/fetch_videomme_videos.py OUT_DIR --ids-file ids.txt
"""

from __future__ import annotations

import argparse
import os
import shutil

BASE = "https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/main/videos_chunked_{:02d}.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--ids-file")
    ap.add_argument("--chunks", type=int, default=20)
    args = ap.parse_args()

    from remotezip import RemoteZip

    wanted = set(args.ids)
    if args.ids_file:
        with open(args.ids_file) as f:
            wanted |= {line.strip() for line in f if line.strip()}
    if not wanted:
        raise SystemExit("no video ids given")
    os.makedirs(args.out_dir, exist_ok=True)

    remaining = set(wanted)
    for c in range(1, args.chunks + 1):
        if not remaining:
            break
        with RemoteZip(BASE.format(c)) as z:
            for info in z.infolist():
                vid = os.path.splitext(os.path.basename(info.filename))[0]
                if vid not in remaining or info.is_dir():
                    continue
                dst = os.path.join(args.out_dir, os.path.basename(info.filename))
                if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
                    print(f"[chunk {c:02d}] {vid}: already complete")
                else:
                    tmp = dst + ".part"
                    with z.open(info) as src, open(tmp, "wb") as out:
                        shutil.copyfileobj(src, out, length=8 << 20)
                    if os.path.getsize(tmp) != info.file_size:
                        raise RuntimeError(f"{vid}: size mismatch {os.path.getsize(tmp)} != {info.file_size}")
                    os.replace(tmp, dst)
                    print(f"[chunk {c:02d}] {vid}: {info.file_size / 1e6:.1f} MB", flush=True)
                remaining.discard(vid)
    if remaining:
        raise SystemExit(f"not found in any chunk: {sorted(remaining)}")
    print(f"done: {len(wanted)} videos in {args.out_dir}")


if __name__ == "__main__":
    main()
