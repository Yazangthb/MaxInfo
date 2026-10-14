import json

import pytest

from maxinfo.eval import MCQItem, parse_choice, run_mcq, score


@pytest.mark.parametrize("text,n,expected", [
    ("C", 5, "C"),
    ("(D)", 5, "D"),
    ("B.", 4, "B"),
    ("A. The man opens the door", 4, "A"),
    ("Based on the video, the answer is C", 5, "C"),  # official parser: B
    ("Answer: D", 5, "D"),  # official parser: A
    ("Every step shows (A)", 5, "A"),  # official parser: E
    ("The correct option is E", 5, "E"),
    ("the answer is a man walking", 5, None),  # lowercase 'a' is not an option letter
    ("A man walks into the room.", 4, None),
    ("I cannot determine this.", 4, None),  # official Video-MME parser: C
    ("(A) or (C)", 4, None),  # ambiguous
    ("E", 4, None),  # out of range
    ("", 4, None),
    (None, 4, None),
])
def test_parse_choice(text, n, expected):
    assert parse_choice(text, n) == expected


def items(n):
    return [MCQItem(f"q{i}", "v.mp4", "q?", ["a", "b", "c", "d"], "B", {"group": i % 2}) for i in range(n)]


def test_run_and_score(tmp_path):
    out = tmp_path / "r.jsonl"
    answers = iter(["B", "the answer is B", "no idea", "A"])
    run_mcq(items(4), lambda it: {"text": next(answers), "n_frames": 8}, str(out))
    s = score(str(out), by="group")
    assert s["overall"] == {"n": 4, "accuracy": 0.5, "correct": 2, "errors": 0, "unparsable": 1,
                            "pred_distribution": {"B": 2, None: 1, "A": 1}}
    assert s["group"]["0"]["n"] == 2


def test_errors_are_recorded_and_abort(tmp_path):
    out = tmp_path / "r.jsonl"

    def boom(item):
        raise NameError("name 'torch' is not defined")

    with pytest.raises(RuntimeError, match="consecutive failures"):
        run_mcq(items(10), boom, str(out), max_consecutive_errors=3)
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == 3 and all(r["pred"] is None and "NameError" in r["error"] for r in recs)


def test_run_mcq_multi_item_major_and_resume(tmp_path):
    from maxinfo.eval import run_mcq_multi

    calls = []

    def fn(name):
        def answer(item):
            calls.append((item.qid, name))
            return {"text": "B" if name == "x" else "A"}
        return answer

    paths = {"x": str(tmp_path / "x.jsonl"), "y": str(tmp_path / "y.jsonl")}
    run_mcq_multi(items(2), {"x": fn("x"), "y": fn("y")}, paths)
    assert calls == [("q0", "x"), ("q0", "y"), ("q1", "x"), ("q1", "y")]
    assert score(paths["x"])["overall"]["accuracy"] == 1.0 and score(paths["y"])["overall"]["accuracy"] == 0.0
    calls.clear()
    run_mcq_multi(items(3), {"x": fn("x"), "y": fn("y")}, paths)  # resumes: only q2 is new
    assert calls == [("q2", "x"), ("q2", "y")]


def test_resume_retries_only_failed(tmp_path):
    out = tmp_path / "r.jsonl"
    calls = []

    def flaky(item):
        calls.append(item.qid)
        if item.qid == "q1" and calls.count("q1") == 1:
            raise RuntimeError("transient")
        return {"text": "B"}

    run_mcq(items(3), flaky, str(out))
    run_mcq(items(3), flaky, str(out))
    assert calls == ["q0", "q1", "q2", "q1"]
    assert score(str(out))["overall"]["accuracy"] == 1.0
