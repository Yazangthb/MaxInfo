"""Resumable multiple-choice evaluation loop and scoring.

The official notebooks wrap each item in `except Exception: write -1`, so a missing import or a
version mismatch turns a whole run into "-1" answers that look like a finished evaluation. Here
every failure is recorded with its traceback, and the run aborts when failures are systematic.
"""

from __future__ import annotations

import json
import os
import traceback
from collections import Counter
from typing import Callable, Iterable

from .datasets import MCQItem
from .parsing import parse_choice


class _Recorder:
    """Appends one JSON line per item to `path`; skips qids already answered there (resume);
    raises when failures are systematic."""

    def __init__(self, path: str, max_consecutive_errors: int, max_error_rate: float, min_items_for_rate: int):
        self.path = path
        self.done = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("error") is None:
                        self.done.add(rec["qid"])
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.limits = (max_consecutive_errors, max_error_rate, min_items_for_rate)
        self.n_run = self.n_err = self.consecutive = 0

    def run(self, item: MCQItem, answer_fn: Callable[[MCQItem], dict]):
        if item.qid in self.done:
            return
        rec = {"qid": item.qid, "gold": item.answer, "n_options": len(item.options), "meta": item.meta}
        try:
            out = answer_fn(item)
            rec.update({k: v for k, v in out.items() if k != "text"})
            rec["raw"] = out["text"]
            rec["pred"] = parse_choice(out["text"], len(item.options))
            rec["error"] = None
            self.consecutive = 0
        except Exception as e:  # recorded, never converted into an answer
            rec.update(raw=None, pred=None, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            self.n_err += 1
            self.consecutive += 1
        self.n_run += 1
        self.f.write(json.dumps(rec, default=_jsonable) + "\n")
        self.f.flush()
        max_consecutive, max_rate, min_items = self.limits
        if self.consecutive >= max_consecutive:
            raise RuntimeError(f"{self.path}: {self.consecutive} consecutive failures; last: {rec['error']}\n"
                               f"{rec['traceback']}")
        if self.n_run >= min_items and self.n_err / self.n_run > max_rate:
            raise RuntimeError(f"{self.path}: error rate {self.n_err}/{self.n_run} exceeds {max_rate:.0%}; "
                               f"last: {rec['error']}")

    def close(self):
        self.f.close()


def run_mcq(
    items: Iterable[MCQItem],
    answer_fn: Callable[[MCQItem], dict],
    out_path: str,
    *,
    max_consecutive_errors: int = 5,
    max_error_rate: float = 0.05,
    min_items_for_rate: int = 50,
) -> str:
    """Run `answer_fn` over items, appending one JSON line per item to `out_path`.

    answer_fn(item) must return {"text": raw model output, ...extra fields, e.g. n_frames, timestamps}.
    Items already present in `out_path` (by qid, without error) are skipped, so runs can resume.
    """
    run_mcq_multi(items, {"_": answer_fn}, {"_": out_path}, max_consecutive_errors=max_consecutive_errors,
                  max_error_rate=max_error_rate, min_items_for_rate=min_items_for_rate)
    return out_path


def run_mcq_multi(
    items: Iterable[MCQItem],
    answer_fns: dict[str, Callable[[MCQItem], dict]],
    out_paths: dict[str, str],
    *,
    max_consecutive_errors: int = 5,
    max_error_rate: float = 0.05,
    min_items_for_rate: int = 50,
    on_item: Callable[[int, MCQItem], None] | None = None,
) -> dict[str, str]:
    """Item-major version of `run_mcq`: every condition answers item i before any condition sees
    item i+1, so per-video work (decoding, embedding) can be shared across conditions. Each condition
    writes, resumes and enforces error limits on its own file exactly as `run_mcq` does."""
    recs = {k: _Recorder(out_paths[k], max_consecutive_errors, max_error_rate, min_items_for_rate) for k in answer_fns}
    try:
        for i, item in enumerate(items):
            for k, fn in answer_fns.items():
                recs[k].run(item, fn)
            if on_item:
                on_item(i, item)
    finally:
        for r in recs.values():
            r.close()
    return out_paths


def _jsonable(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def score(path: str, *, by: str | None = None) -> dict:
    """Accuracy over items with a gold answer. Unparsable outputs and errors count as wrong but are
    reported separately so they cannot hide inside the accuracy number."""
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["qid"] not in recs or recs[r["qid"]].get("error") is not None:
                recs[r["qid"]] = r  # keep the latest successful attempt per qid
    rows = [r for r in recs.values() if r.get("gold")]

    def summarize(rs):
        n = len(rs)
        correct = sum(r.get("pred") == r["gold"] for r in rs)
        errors = sum(r.get("error") is not None for r in rs)
        unparsable = sum(r.get("error") is None and r.get("pred") is None for r in rs)
        return {"n": n, "accuracy": correct / n if n else float("nan"), "correct": correct,
                "errors": errors, "unparsable": unparsable,
                "pred_distribution": dict(Counter(r.get("pred") for r in rs))}

    result = {"overall": summarize(rows), "items_without_gold": len(recs) - len(rows)}
    if by:
        groups: dict = {}
        for r in rows:
            groups.setdefault(str((r.get("meta") or {}).get(by)), []).append(r)
        result[by] = {k: summarize(v) for k, v in sorted(groups.items())}
    return result
