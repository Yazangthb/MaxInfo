"""Frame encoders producing one embedding per frame.

Fixes relative to the official notebooks:
- batched encoding (the notebooks call CLIP once per frame);
- works on transformers 4.x and 5.x: `get_image_features` returns a tensor in 4.x but a
  `BaseModelOutputWithPooling` in 5.x, where the notebooks' `[0]` silently picks the patch-token
  hidden states instead of the embedding;
- inputs are moved to the device the model actually lives on (the notebooks pick `cuda:0`
  independently of where `device_map="auto"` placed CLIP).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_ENCODER = "openai/clip-vit-large-patch14-336"

# Families whose `get_image_features` gives the (projected) image embedding.
_CONTRASTIVE = {"clip", "siglip", "siglip2", "chinese_clip"}


def _image_embedding(out):
    """Normalise the return value of `get_image_features` across transformers versions."""
    import torch

    if torch.is_tensor(out):
        return out
    for name in ("image_embeds", "pooler_output"):
        val = getattr(out, name, None)
        if val is not None:
            return val
    raise TypeError(f"cannot find an image embedding in {type(out).__name__}")


class FrameEncoder:
    """Encode frames (PIL images or HxWx3 uint8 arrays) into an [n, d] float32 array.

    CLIP / SigLIP: projected image embedding (what the official code uses; the paper's text says
    "[CLS] token", which for CLIP is the pre-projection pooled output).
    DINOv2 and other vision backbones: the [CLS] token of the last hidden state.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ENCODER,
        *,
        device: str | None = None,
        dtype: str | None = None,
        batch_size: int = 32,
    ):
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModel

        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch_dtype = getattr(torch, dtype) if dtype else None

        config = AutoConfig.from_pretrained(model_name)
        self.family = config.model_type
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        # Cast after loading: the kwarg is `torch_dtype` in transformers 4.x and `dtype` in 5.x.
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device, dtype=torch_dtype).eval()

    def __call__(self, frames: Sequence) -> np.ndarray:
        import torch

        frames = list(frames)
        if not frames:
            return np.zeros((0, 0), dtype=np.float32)
        model_dtype = next(self.model.parameters()).dtype
        chunks = []
        with torch.inference_mode():
            for i in range(0, len(frames), self.batch_size):
                inputs = self.processor(images=frames[i : i + self.batch_size], return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device, dtype=model_dtype)
                if self.family in _CONTRASTIVE:
                    emb = _image_embedding(self.model.get_image_features(pixel_values=pixel_values))
                else:
                    emb = self.model(pixel_values=pixel_values).last_hidden_state[:, 0]
                chunks.append(emb.float().cpu().numpy())
        feats = np.concatenate(chunks, axis=0)
        if feats.shape[0] != len(frames) or feats.ndim != 2:
            raise RuntimeError(f"encoder returned shape {feats.shape} for {len(frames)} frames")
        return feats
