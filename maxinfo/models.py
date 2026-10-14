"""VLLM adapters: `make_*` returns answer_fn(item) -> {"text", "n_frames", "timestamps"} for `run_mcq`.

Each adapter: samples candidates -> MaxInfo (or uniform) -> passes frames *with timing information*
to the model. Model calls mirror the official MaxInfo / model-card usage; heavy imports are lazy.
NOTE: these adapters have not been run end-to-end in this repository (no suitable GPU locally);
the selection, sampling, parsing and scoring code they rely on is covered by tests.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .eval.datasets import MCQItem
from .pipeline import MaxInfo
from .prompts import mcq_prompt, time_instruction
from .video import SampledVideo, sample_video


def _frames_for(item: MCQItem, maxinfo: MaxInfo | None, budget: int, num_candidates: int | None,
                fps: float | None, max_candidates: int | None) -> SampledVideo:
    if maxinfo is None:  # uniform baseline through the same decoder
        return sample_video(item.video_path, num_frames=budget)
    return maxinfo.from_video(item.video_path, budget=budget, num_candidates=num_candidates, fps=fps,
                              max_candidates=max_candidates).video


def make_llava_video(model_path: str = "lmms-lab/LLaVA-Video-7B-Qwen2", *, maxinfo: MaxInfo | None,
                     budget: int = 64, num_candidates: int | None = 128, fps: float | None = None,
                     max_candidates: int | None = None, use_time_instruction: bool = True,
                     max_new_tokens: int = 16) -> Callable[[MCQItem], dict]:
    """LLaVA-Video (pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT)."""
    import copy

    import torch
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, "llava_qwen", torch_dtype="bfloat16",
                                                           device_map="auto")
    model.eval()

    def answer(item: MCQItem) -> dict:
        v = _frames_for(item, maxinfo, budget, num_candidates, fps, max_candidates)
        pixels = processor.preprocess(v.frames, return_tensors="pt")["pixel_values"].to(model.device, torch.bfloat16)
        prompt = mcq_prompt(item.question, item.options)
        if use_time_instruction:
            prompt = time_instruction(v.duration, v.timestamps) + "\n" + prompt
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        out = model.generate(ids.unsqueeze(0).to(model.device), images=[pixels], modalities=["video"],
                             do_sample=False, max_new_tokens=max_new_tokens)
        text = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        return {"text": text, "n_frames": len(v.indices), "timestamps": v.timestamps}

    return answer


def make_qwen2_vl(model_path: str = "Qwen/Qwen2-VL-7B-Instruct", *, maxinfo: MaxInfo | None, budget: int = 180,
                  num_candidates: int | None = None, fps: float | None = 1.0, max_candidates: int | None = 768,
                  max_pixels: int = 360 * 420, use_time_instruction: bool = True,
                  max_new_tokens: int = 16) -> Callable[[MCQItem], dict]:
    """Qwen2-VL / Qwen2.5-VL via transformers + qwen_vl_utils.

    Qwen2-VL's patch embedding merges each pair of consecutive frames into one temporal patch, so
    MaxInfo should select frame *pairs*: build the MaxInfo block with MaxInfoConfig(group_size=2).
    """
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if maxinfo is not None and maxinfo.config.group_size != 2:
        raise ValueError("use MaxInfoConfig(group_size=2) with Qwen2-VL (temporal patch size 2)")
    model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(model_path)

    def answer(item: MCQItem) -> dict:
        v = _frames_for(item, maxinfo, budget, num_candidates, fps if maxinfo else None, max_candidates)
        if len(v.indices) % 2:  # pad by repeating the last frame instead of dropping it
            v = v.subset(np.r_[np.arange(len(v.indices)), len(v.indices) - 1])
        prompt = mcq_prompt(item.question, item.options)
        if use_time_instruction:
            prompt = time_instruction(v.duration, v.timestamps) + "\n" + prompt
        messages = [{"role": "user", "content": [
            {"type": "video", "video": v.pil(), "max_pixels": max_pixels},
            {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                           return_tensors="pt").to(model.device)
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        text_out = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return {"text": text_out, "n_frames": len(v.indices), "timestamps": v.timestamps}

    return answer


def make_minicpm_v(model_path: str = "openbmb/MiniCPM-V-4_5", *, maxinfo: MaxInfo | None, budget: int = 128,
                   num_candidates: int | None = None, fps: float | None = 1.0, max_candidates: int | None = 128,
                   time_scale: float = 0.1, subtitles: Callable[[MCQItem], str] | None = None) -> Callable[[MCQItem], dict]:
    """MiniCPM-V 4.5 (trust_remote_code). Temporal ids are passed per selected frame, as the
    official notebook does (this is the one official pipeline that keeps timing information)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation="sdpa",
                                      torch_dtype=torch.bfloat16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def answer(item: MCQItem) -> dict:
        v = _frames_for(item, maxinfo, budget, num_candidates, fps if maxinfo else None, max_candidates)
        prompt = mcq_prompt(item.question, item.options)
        if subtitles is not None:
            prompt = f"This video's subtitles are listed below:\n{subtitles(item)}\n" + prompt
        temporal_ids = [[int(round(t / time_scale))] for t in v.timestamps]
        text = model.chat(msgs=[{"role": "user", "content": v.pil() + [prompt]}], tokenizer=tokenizer,
                          use_image_id=False, max_slice_nums=1, temporal_ids=temporal_ids)
        return {"text": text, "n_frames": len(v.indices), "timestamps": v.timestamps}

    return answer
