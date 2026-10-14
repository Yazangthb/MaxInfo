"""Benchmark loaders into one `MCQItem` format.

Every loader validates required fields and raises on anything missing, instead of the official
notebooks' pattern of a DataLoader worker crashing on an unbound `video_path` (missing file) or
records being silently skipped.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

LETTERS = "ABCDEFGHIJ"


@dataclass
class MCQItem:
    qid: str
    video_path: str
    question: str
    options: list[str]  # bare option texts
    answer: str | None  # gold letter, None if hidden (EgoSchema full set, LVB test)
    meta: dict = field(default_factory=dict)


def _require(rec: dict, keys, src: str):
    missing = [k for k in keys if k not in rec]
    if missing:
        raise KeyError(f"{src}: record missing fields {missing}; available: {sorted(rec)}")


def _find_video(folder: str, stem: str) -> str:
    for ext in ("", ".mp4", ".mkv", ".webm", ".avi", ".mov"):
        p = os.path.join(folder, stem + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"no video for '{stem}' in {folder}")


def _strip_letter(opt: str) -> str:
    return re.sub(r"^\s*\(?[A-J][.)]\s*", "", opt).strip()


def load_videomme(parquet_path: str, video_folder: str, *, durations=None) -> list[MCQItem]:
    """lmms-lab/Video-MME: fields video_id, videoID, duration, domain, sub_category, task_type,
    question_id, question, options (['A. ...', ...]), answer (letter)."""
    import pandas as pd

    items = []
    for rec in pd.read_parquet(parquet_path).to_dict("records"):
        _require(rec, ["videoID", "question_id", "question", "options", "answer"], "Video-MME")
        if durations and rec.get("duration") not in durations:
            continue
        items.append(MCQItem(str(rec["question_id"]), _find_video(video_folder, rec["videoID"]), rec["question"],
                             [_strip_letter(o) for o in rec["options"]], str(rec["answer"]).strip(),
                             {k: rec.get(k) for k in ("duration", "domain", "sub_category", "task_type")}))
    return items


def load_egoschema(questions_json: str, video_folder: str, *, answers_json: str | None = None) -> list[MCQItem]:
    """EgoSchema questions.json (q_uid, question, option 0..4). Gold answers are public only for the
    500-question subset (subset_answers.json: {q_uid: index}); others get answer=None."""
    with open(questions_json, encoding="utf-8") as f:
        questions = json.load(f)
    gold = {}
    if answers_json:
        with open(answers_json, encoding="utf-8") as f:
            gold = json.load(f)
    items = []
    for rec in questions:
        _require(rec, ["q_uid", "question"] + [f"option {i}" for i in range(5)], "EgoSchema")
        qid = rec["q_uid"]
        ans = LETTERS[int(gold[qid])] if qid in gold else None
        items.append(MCQItem(qid, _find_video(video_folder, qid), rec["question"],
                             [rec[f"option {i}"] for i in range(5)], ans))
    return items


def load_longvideobench(json_path: str, video_folder: str) -> list[MCQItem]:
    """LongVideoBench lvb_val.json. Expected fields (from the LVB release; the HF repo is gated,
    so they are checked at load time): id, video_path, question, candidates, correct_choice (int),
    duration_group, question_category."""
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)
    items = []
    for rec in records:
        _require(rec, ["id", "video_path", "question", "candidates"], "LongVideoBench")
        cc = rec.get("correct_choice")
        ans = LETTERS[int(cc)] if isinstance(cc, (int, float)) and cc >= 0 else None
        vp = os.path.join(video_folder, rec["video_path"])
        if not os.path.isfile(vp):
            raise FileNotFoundError(vp)
        items.append(MCQItem(str(rec["id"]), vp, rec["question"], list(rec["candidates"]), ans,
                             {k: rec.get(k) for k in ("duration_group", "question_category", "duration")}))
    return items
