import glob
import os

import numpy as np
import pytest

from maxinfo.video import candidate_indices


def test_candidate_indices_count_and_unique():
    idx = candidate_indices(3000, 30.0, num_frames=128)
    assert len(idx) == 128 and len(np.unique(idx)) == 128
    short = candidate_indices(40, 30.0, num_frames=128)  # shorter than requested: no duplicates
    assert len(short) == len(np.unique(short)) == 40


def test_candidate_indices_fps_and_cap():
    idx = candidate_indices(30 * 600, 30.0, fps=1.0)  # 10 min @ 1 fps
    assert len(idx) == 600
    assert np.allclose(np.diff(idx), 30)
    assert len(candidate_indices(30 * 600, 30.0, fps=1.0, max_frames=256)) == 256
    with pytest.raises(ValueError):
        candidate_indices(100, 30.0, num_frames=8, fps=1.0)


def test_candidate_indices_segment():
    idx = candidate_indices(3000, 25.0, num_frames=10, start_sec=20, end_sec=40)
    assert idx.min() >= 500 and idx.max() < 1000


LOCAL_VIDEOS = sorted(glob.glob(os.environ.get("MAXINFO_TEST_VIDEOS", "") + "/*.mp4"))


@pytest.mark.skipif(not LOCAL_VIDEOS, reason="set MAXINFO_TEST_VIDEOS to a folder of .mp4 files")
def test_sample_video_timestamps():
    from maxinfo.video import sample_video

    v = sample_video(LOCAL_VIDEOS[0], num_frames=16)
    assert v.frames.shape[0] == 16 and v.frames.dtype == np.uint8
    assert np.allclose(v.timestamps, v.indices / v.native_fps)
    assert 0 <= v.timestamps.min() and v.timestamps.max() <= v.duration
    v4 = sample_video(LOCAL_VIDEOS[0], num_frames=16, num_threads=4)  # threading must not change frames
    assert np.array_equal(v.frames, v4.frames) and np.array_equal(v.indices, v4.indices)


def test_frame_encoder_returns_projected_embedding(tmp_path):
    """On transformers 5.x get_image_features returns a ModelOutput; the official `[0]` would give
    hidden states [1, tokens, hidden]. The encoder must return [n, projection_dim]."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel

    from maxinfo.encoders import FrameEncoder

    cfg = CLIPConfig(
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                           image_size=28, patch_size=14),
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2),
        projection_dim=16)
    CLIPModel(cfg).save_pretrained(tmp_path)
    CLIPImageProcessor(size={"shortest_edge": 28}, crop_size={"height": 28, "width": 28}).save_pretrained(tmp_path)

    enc = FrameEncoder(str(tmp_path), device="cpu", batch_size=4)
    frames = [np.random.default_rng(i).integers(0, 255, (40, 60, 3), dtype=np.uint8) for i in range(10)]
    feats = enc(frames)
    assert feats.shape == (10, 16) and feats.dtype == np.float32
    with torch.inference_mode():  # batched result equals one-at-a-time result
        single = np.concatenate([enc([f]) for f in frames[:3]])
    assert np.allclose(single, feats[:3], atol=1e-5)
