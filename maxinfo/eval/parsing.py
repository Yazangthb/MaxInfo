"""Multiple-choice answer extraction.

The official EgoSchema parser takes the first capital A-E anywhere in the output
("Based on the video, the answer is C" -> B; "Answer: D" -> A), and the Video-MME-style parser
maps anything it cannot parse to "C". This parser only accepts explicit answer forms and
returns None otherwise, so unparsable outputs are counted separately instead of silently scored.
"""

from __future__ import annotations

import re


def parse_choice(text: str | None, n_options: int) -> str | None:
    if not text:
        return None
    if not 1 <= n_options <= 10:
        raise ValueError("n_options must be in 1..10")
    valid = "ABCDEFGHIJ"[:n_options]
    L = f"[{valid}]"
    t = text.strip()
    if len(set(re.findall(rf"\(({L})\)", t))) > 1:
        return None  # several options named, e.g. "(A) or (C)"

    patterns = [
        rf"^\(?({L})\)?(?:[.:)\]]|\s*$)",  # "C", "(C)", "C.", "C) text", "C: text"
        # "the answer is (C)", "Answer: D", "correct option is C" — keywords any case, letter uppercase
        rf"\b(?i:answer|option|choice)\s*(?i:is)?\s*:?\s*(?i:option\s*)?\(?({L})\)?(?![A-Za-z])",
        rf"\(({L})\)",  # "... (C) ..."
    ]
    for p in patterns:
        hits = {m.group(1) for m in re.finditer(p, t)}
        if len(hits) == 1:
            return hits.pop()
        if len(hits) > 1:
            return None  # ambiguous, e.g. "(A) or (C)"
    return None
