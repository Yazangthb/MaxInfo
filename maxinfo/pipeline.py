from __future__ import annotations


_CLIP_CACHE: dict[tuple[str, str | None, str | None], tuple[object, object]] = {}


def _get_clip(model_name: str, device_map: str | None, torch_dtype: str | None, progress: bool = True):
    key = (str(model_name), device_map, torch_dtype)
    if key in _CLIP_CACHE:
        return _CLIP_CACHE[key]
    from .clip import load_clip

    model, processor = load_clip(model_name, device_map=device_map, torch_dtype=torch_dtype, progress=progress)
    _CLIP_CACHE[key] = (model, processor)
    return model, processor


def compare_sampling_on_video(
    *,
    video_path: str,
    choose_fps: float = 3.0,
    max_candidates: int = 256,
    budget: int = 32,
    svd_rank: int = 8,
    tol: float = 0.3,
    batch_size: int = 16,
    clip_model_name: str = "openai/clip-vit-large-patch14-336",
    device_map: str | None = "auto",
    torch_dtype: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    remove_similar: bool = False,
    similarity_thr: float = 0.9,
    cols: int = 8,
    show: bool = True,
    progress: bool = True,
):
    """Replicate notebook section **Experiment pipeline 2** using the module.

    Compares:
      (1) uniform sampling baseline
      (2) MaxInfo (SVD + RectMaxVol greedy with tolerance)

    Args:
        progress: Whether to show progress updates.

    Returns:
        dict with frames/indices and candidate metadata.
    """

    from .video import encode_video_candidates
    from .selection import uniform_select, maxinfo_select_rect_maxvol
    from .viz import show_grid, show_timeline

    if progress:
        print(f"=" * 60)
        print(f"🚀 Starting MaxInfo pipeline")
        print(f"   Video: {video_path}")
        print(f"   FPS sample: {choose_fps}, Max candidates: {max_candidates}")
        if start_sec is not None or end_sec is not None:
            print(f"   Segment: {start_sec}s - {end_sec}s")
        print(f"=" * 60)

    vision_model, vision_processor = _get_clip(clip_model_name, device_map, torch_dtype, progress=progress)

    if progress:
        print(f"📹 Loading video candidates (fps={choose_fps}, max={max_candidates})...")
    frames, ts_group, orig_frame_idx, timestamps = encode_video_candidates(
        video_path=video_path,
        choose_fps=choose_fps,
        max_candidates=max_candidates,
        start_sec=start_sec,
        end_sec=end_sec,
        time_scale=0.1,
    )
    if progress:
        print(f"✓ Loaded {len(frames)} frames from video")

    if progress:
        print(f"📊 Running uniform selection (budget={budget})...")
    uni_frames, uni_ts, uni_idx = uniform_select(frames, ts_group, budget=budget)
    if progress:
        print(f"✓ Uniform selected {len(uni_idx)} frames")

    if progress:
        print(f"🎯 Running MaxInfo selection...")
    mi_frames, mi_ts, mi_idx = maxinfo_select_rect_maxvol(
        vision_model=vision_model,
        vision_processor=vision_processor,
        frames=frames,
        ts_group=ts_group,
        budget=budget,
        svd_rank=svd_rank,
        tol=tol,
        batch_size=batch_size,
        remove_similar=remove_similar,
        similarity_thr=similarity_thr,
        progress=progress,
    )

    if progress:
        print(f"=" * 60)
        print(f"✅ Pipeline complete!")
        print(f"   Uniform: {len(uni_idx)} frames")
        print(f"   MaxInfo: {len(mi_idx)} frames")
        print(f"=" * 60)

    if show:
        seg_label = (
            f"[{start_sec:.2f}s–{end_sec:.2f}s]" if (start_sec is not None or end_sec is not None) else "[full]"
        )
        show_grid(
            uni_frames,
            uni_idx,
            title=f"Uniform (budget={budget}) {seg_label} — {video_path}",
            cols=cols,
        )
        show_grid(
            mi_frames,
            mi_idx,
            title=f"MaxInfo (tol={tol}, rank={svd_rank}, budget={budget}) {seg_label} — {video_path}",
            cols=cols,
        )
        show_timeline(
            uniform_idx=uni_idx,
            maxinfo_idx=mi_idx,
            total_candidates=len(frames),
            title=f"Uniform vs MaxInfo timeline {seg_label} — {video_path}",
        )

    return {
        "video_path": video_path,
        "segment": {
            "start_sec": 0.0 if start_sec is None else float(start_sec),
            "end_sec": None if end_sec is None else float(end_sec),
        },
        "candidates": {
            "num": len(frames),
            "orig_frame_idx": orig_frame_idx,
            "timestamps": timestamps,
        },
        "uniform": {
            "indices": uni_idx,
            "frames": uni_frames,
            "ts_group": uni_ts,
        },
        "maxinfo": {
            "indices": mi_idx,
            "frames": mi_frames,
            "ts_group": mi_ts,
        },
    }
