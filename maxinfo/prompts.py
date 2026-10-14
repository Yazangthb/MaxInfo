"""Prompt helpers.

MaxInfo frames are irregularly spaced. LLaVA-Video was trained with a time instruction listing
frame timestamps; the official MaxInfo notebooks comment it out, so the model is given no way to
know that consecutive frames may be minutes apart. These helpers restore that information.
"""

from __future__ import annotations

from typing import Sequence

LETTERS = "ABCDEFGHIJ"


def time_instruction(duration: float, timestamps: Sequence[float]) -> str:
    """LLaVA-Video style time instruction, worded for non-uniform frames."""
    times = ",".join(f"{t:.2f}s" for t in timestamps)
    return (f"The video lasts for {duration:.2f} seconds, and {len(timestamps)} frames are selected from it. "
            f"These frames are located at {times}. Please answer the following questions related to this video.")


def mcq_prompt(question: str, options: Sequence[str], *, letters: str = LETTERS,
               instruction: str = "Respond with only the letter of the correct option.") -> str:
    """Multiple-choice prompt. `options` are bare option texts (no 'A.' prefix)."""
    lines = [question.strip()]
    lines += [f"({letters[i]}) {o.strip()}" for i, o in enumerate(options)]
    lines.append(instruction)
    return "\n".join(lines)
