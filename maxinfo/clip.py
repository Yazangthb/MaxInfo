from __future__ import annotations


def load_clip(
    model_name: str = "openai/clip-vit-large-patch14-336",
    *,
    device_map: str | None = "auto",
    torch_dtype: str | None = None,
    progress: bool = True,
):
    """Load CLIP model + processor.

    Args:
        device_map: passed through to `from_pretrained` (accelerate style).
        torch_dtype: optional string like "float16"/"bfloat16".
        progress: whether to show download progress.
    """

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    from transformers import AutoProcessor, CLIPModel
    from tqdm.auto import tqdm

    dtype = None
    if torch_dtype is not None:
        dtype = getattr(torch, str(torch_dtype))

    if progress:
        print(f"📦 Loading CLIP model: {model_name}...")

    model = CLIPModel.from_pretrained(model_name, device_map=device_map, torch_dtype=dtype)

    if progress:
        print(f"✓ Model loaded: {model_name}")
        print(f"📦 Loading processor: {model_name}...")

    processor = AutoProcessor.from_pretrained(model_name)

    if progress:
        print(f"✓ Processor loaded")

    return model, processor


def _model_device(model) -> "torch.device":
    import torch

    try:
        return model.device
    except Exception:
        pass
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def extract_clip_features(
    vision_model,
    vision_processor,
    frames,
    *,
    batch_size: int = 16,
    normalize: bool = False,
    progress: bool = True,
):
    """Extract CLIP image embeddings.

    Args:
        frames: List of PIL images or numpy arrays.
        batch_size: Number of frames to process at once.
        normalize: Whether to L2 normalize features.
        progress: Whether to show progress bar.

    Returns:
        torch.Tensor: shape [T, D] float32 on CPU.
    """

    import torch
    from tqdm.auto import tqdm

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    device = _model_device(vision_model)
    feats_all = []
    total_batches = (len(frames) + batch_size - 1) // batch_size

    with torch.no_grad():
        iterator = range(0, len(frames), batch_size)
        if progress:
            iterator = tqdm(iterator, desc="🔍 Extracting CLIP features", total=total_batches)
        
        for i in iterator:
            batch = frames[i : i + batch_size]
            inputs = vision_processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

            if hasattr(vision_model, "get_image_features"):
                feats = vision_model.get_image_features(**inputs)
                # Handle both direct tensor output and BaseModelOutputWithPooling
                if hasattr(feats, "pooler_output"):
                    feats = feats.pooler_output
            else:
                vision_out = vision_model.vision_model(pixel_values=inputs["pixel_values"])
                pooled = vision_out.pooler_output
                feats = vision_model.visual_projection(pooled)

            feats = feats.float()
            if normalize:
                feats = torch.nn.functional.normalize(feats, dim=-1)
            feats_all.append(feats.cpu())

    return torch.cat(feats_all, dim=0)

