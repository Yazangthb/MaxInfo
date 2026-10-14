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
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("error") is None:
                    done.add(rec["qid"])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    n_run = n_err = consecutive = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for item in items:
            if item.qid in done:
                continue
            rec = {"qid": item.qid, "gold": item.answer, "n_options": len(item.options), "meta": item.meta}
            try:
                out = answer_fn(item)
                rec.update({k: v for k, v in out.items() if k != "text"})
                rec["raw"] = out["text"]
                rec["pred"] = parse_choice(out["text"], len(item.options))
                rec["error"] = None
                consecutive = 0
            except Exception as e:  # recorded, never converted into an answer
                rec.update(raw=None, pred=None, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
                n_err += 1
                consecutive += 1
            n_run += 1
            f.write(json.dumps(rec, default=_jsonable) + "\n")
            f.flush()
            if consecutive >= max_consecutive_errors:
                raise RuntimeError(f"{consecutive} consecutive failures; last: {rec['error']}\n{rec['traceback']}")
            if n_run >= min_items_for_rate and n_err / n_run > max_error_rate:
                raise RuntimeError(f"error rate {n_err}/{n_run} exceeds {max_error_rate:.0%}; last: {rec['error']}")
    return out_path


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
