"""VLLM adapters.

Split into three parts so the model is loaded once and the exact inputs can be inspected without it:
  frames_fn(item)         -> SampledVideo      (`make_frames_fn`: MaxInfo or uniform)
  prompt_fn(item, video)  -> str               (`build_prompt`)
  generate(video, prompt) -> str               (`load_llava_video` / `load_qwen2_vl` / `load_minicpm_v`)
`make_answer_fn` combines them into the callable `eval.run_mcq` expects.

Model calls mirror each model's official usage; heavy imports are lazy.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .eval.datasets import MCQItem
from .pipeline import MaxInfo
from .prompts import mcq_prompt, time_instruction
from .video import SampledVideo, sample_video

FramesFn = Callable[[MCQItem], SampledVideo]
GenerateFn = Callable[[SampledVideo, str], str]

VIDEOMME_INSTRUCTION = ("Select the best answer to the following multiple-choice question based on the video. "
                        "Respond with only the letter (A, B, C, or D) of the correct option.")


def make_frames_fn(maxinfo: MaxInfo | None, budget: int, *, num_candidates: int | None = None,
                   fps: float | None = None, max_candidates: int | None = None) -> FramesFn:
    """maxinfo=None -> uniform sampling of `budget` frames through the same decoder."""
    def frames(item: MCQItem) -> SampledVideo:
        if maxinfo is None:
            return sample_video(item.video_path, num_frames=budget)
        return maxinfo.from_video(item.video_path, budget=budget, num_candidates=num_candidates, fps=fps,
                                  max_candidates=max_candidates).video
    return frames


def build_prompt(item: MCQItem, video: SampledVideo, *, use_time_instruction: bool,
                 preamble: str = VIDEOMME_INSTRUCTION) -> str:
    body = preamble + "\n" + mcq_prompt(item.question, item.options, instruction="The best answer is:")
    if use_time_instruction:
        body = time_instruction(video.duration, video.timestamps) + "\n" + body
    return body


def make_answer_fn(generate: GenerateFn, frames_fn: FramesFn, *, use_time_instruction: bool) -> Callable[[MCQItem], dict]:
    def answer(item: MCQItem) -> dict:
        v = frames_fn(item)
        prompt = build_prompt(item, v, use_time_instruction=use_time_instruction)
        return {"text": generate(v, prompt), "n_frames": int(len(v.indices)),
                "timestamps": [round(float(t), 2) for t in v.timestamps]}
    return answer


def load_llava_video(model_path: str = "lmms-lab/LLaVA-Video-7B-Qwen2", *, attn_implementation: str = "sdpa",
                     max_new_tokens: int = 16) -> GenerateFn:
    """LLaVA-Video via LLaVA-NeXT (github.com/LLaVA-VL/LLaVA-NeXT). The builder defaults to
    flash_attention_2; `sdpa` avoids that dependency."""
    import copy

    import torch
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, "llava_qwen", torch_dtype="bfloat16",
                                                           device_map="auto", attn_implementation=attn_implementation)
    model.eval()

    def generate(v: SampledVideo, prompt: str) -> str:
        pixels = processor.preprocess(v.frames, return_tensors="pt")["pixel_values"].to(model.device, torch.bfloat16)
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        with torch.inference_mode():
            out = model.generate(ids.unsqueeze(0).to(model.device), images=[pixels], modalities=["video"],
                                 do_sample=False, temperature=0, max_new_tokens=max_new_tokens)
        return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

    return generate


def load_qwen2_vl(model_path: str = "Qwen/Qwen2-VL-2B-Instruct", *, max_pixels: int = 360 * 420,
                  max_new_tokens: int = 16) -> GenerateFn:
    """Qwen2-VL / Qwen2.5-VL via transformers + qwen_vl_utils. Frames are given as an image list, so
    no resampling happens; an odd count is padded by repeating the last frame (temporal patch size 2).
    Select with MaxInfoConfig(group_size=2) so each temporal patch holds two adjacent frames."""
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(model_path)

    def generate(v: SampledVideo, prompt: str) -> str:
        if len(v.indices) % 2:
            v = v.subset(np.r_[np.arange(len(v.indices)), len(v.indices) - 1])
        messages = [{"role": "user", "content": [
            {"type": "video", "video": v.pil(), "max_pixels": max_pixels},
            {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        if len(video_inputs[0]) != len(v.indices):
            raise RuntimeError(f"qwen_vl_utils changed the frame count: {len(v.indices)} -> {len(video_inputs[0])}")
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, do_sample_frames=False,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    return generate


def load_minicpm_v(model_path: str = "openbmb/MiniCPM-V-4_5", *, time_scale: float = 0.1) -> GenerateFn:
    """MiniCPM-V 4.5 (trust_remote_code), with per-frame temporal ids as in the official notebook."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation="sdpa",
                                      torch_dtype=torch.bfloat16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def generate(v: SampledVideo, prompt: str) -> str:
        temporal_ids = [[int(round(t / time_scale))] for t in v.timestamps]
        return model.chat(msgs=[{"role": "user", "content": v.pil() + [prompt]}], tokenizer=tokenizer,
                          use_image_id=False, max_slice_nums=1, temporal_ids=temporal_ids)

    return generate
