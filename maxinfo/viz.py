from __future__ import annotations


def show_grid(frames, idx, title: str, cols: int = 8):
    """Display selected frames in a grid (matplotlib)."""

    import math
    import matplotlib.pyplot as plt

    n = len(frames)
    if n == 0:
        return

    rows = int(math.ceil(n / int(cols)))
    plt.figure(figsize=(cols * 2.25, rows * 2.25))
    for i, (im, j) in enumerate(zip(frames, idx), start=1):
        ax = plt.subplot(rows, cols, i)
        ax.imshow(im)
        ax.set_title(f"cand={j}", fontsize=9)
        ax.axis("off")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def show_timeline(uniform_idx, maxinfo_idx, total_candidates: int, title: str = "Selection timeline"):
    """Scatter plot of candidate indices selected by both methods."""

    import numpy as np
    import matplotlib.pyplot as plt

    u = np.asarray(uniform_idx, dtype=int)
    m = np.asarray(maxinfo_idx, dtype=int)
    plt.figure(figsize=(14, 2.4))
    plt.scatter(u, np.zeros_like(u), label="Uniform", alpha=0.85)
    plt.scatter(m, np.ones_like(m), label="MaxInfo", alpha=0.85)
    plt.yticks([0, 1], ["Uniform", "MaxInfo"])
    plt.xlim(0, max(0, int(total_candidates) - 1))
    plt.xlabel("Candidate frame index")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

