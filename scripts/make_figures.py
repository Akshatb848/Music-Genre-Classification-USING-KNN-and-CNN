"""Redraw the README charts from numbers printed in the saved notebook outputs.

The notebooks contain no plots, so these charts are drawn only from values the
notebooks printed (accuracy lines, classification report, Keras epoch log).
Nothing is re-trained.  Usage:  python scripts/make_figures.py
"""
import json
import pathlib
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#8a8985", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c9c8c3",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e7e6e2", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlesize": 13,
})


def stream_text(notebook, cell_index):
    nb = json.loads((ROOT / notebook).read_text(encoding="utf-8"))
    outs = nb["cells"][cell_index].get("outputs", [])
    text = "".join("".join(o.get("text", "")) for o in outs if o.get("output_type") == "stream")
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def keras_epochs(text):
    """(train_acc, val_acc) per epoch from a Keras progress log."""
    pairs = re.findall(r"- accuracy: ([0-9.]+) - loss: [0-9.e+-]+ - val_accuracy: ([0-9.]+)", text)
    return [(float(a), float(b)) for a, b in pairs]


def curve(ep, fname, title, caption, val_label, best=None):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    x = range(1, len(ep) + 1)
    ax.plot(x, [a for a, _ in ep], color=BLUE, lw=2, marker="o", ms=4, label="Training accuracy")
    ax.plot(x, [b for _, b in ep], color=ORANGE, lw=2, marker="o", ms=4, label=val_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, loc="left", pad=10)
    ax.legend(frameon=False, loc="lower right")
    if best:
        ax.annotate(best[2], xy=best[:2], xytext=(best[0] + 0.4, 0.45), fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    fig.text(0.01, -0.02, textwrap.fill(caption, 110), fontsize=8.5, color=INK2, va="top")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)
    return OUT / fname


def music_figures():
    knn = stream_text("KNN.ipynb", 7) + stream_text("KNN.ipynb", 8) + stream_text("KNN.ipynb", 9) + stream_text("KNN.ipynb", 14)
    pct = lambda pattern: float(re.search(pattern, knn).group(1))  # noqa: E731
    cnn_log = stream_text("CNN.ipynb", 3)
    models = [
        ("KNN, k=5, unscaled", pct(r"Accuracy of KNN model: ([0-9.]+)%")),
        ("KNN, k=5, standardized", pct(r"Accuracy of KNN model \(scaled\): ([0-9.]+)%")),
        ("PCA(10) + KNN, k=5", 100 * pct(r"Accuracy: ([0-9.]+)\n")),
        ("KNN, tuned (manhattan, k=7)", pct(r"Accuracy of Best KNN model: ([0-9.]+)%")),
        ("CNN, epoch 50", 100 * float(re.search(r"Test accuracy: ([0-9.]+)", cnn_log).group(1))),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ys = range(len(models))
    ax.barh(list(ys), [v for _, v in models], color=[GRAY, GRAY, GRAY, BLUE, ORANGE], height=0.6)
    for i, (_, v) in enumerate(models):
        ax.text(v + 0.8, i, f"{v:.2f}%", va="center", fontsize=10, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([n for n, _ in models])
    ax.axvline(10, color=INK2, ls="--", lw=1)
    ax.text(10.6, len(models) - 0.55, "chance (10%)", fontsize=9, color=INK2)
    ax.set_xlim(0, 65)
    ax.set_xlabel("Test accuracy (%), 200 held-out items")
    ax.grid(axis="y", visible=False)
    ax.set_title("GTZAN 10-genre test accuracy by model", loc="left")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "music_model_comparison.png", bbox_inches="tight")
    plt.close(fig)

    curve(keras_epochs(cnn_log), "music_cnn_training.png",
          "CNN on Mel spectrograms: train vs. test accuracy",
          "Plotted from the 50-epoch Keras log saved in CNN.ipynb "
          "(test split = 200 images, also used as validation_data)",
          "Test accuracy (200 images)")

    report = stream_text("KNN.ipynb", 9)
    rows = re.findall(r"^\s+([a-z]+)\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+\d+$", report, re.M)
    rows = sorted(((g, float(f)) for g, f in rows), key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.barh([g for g, _ in rows], [f for _, f in rows], color=BLUE, height=0.6)
    for i, (_, f) in enumerate(rows):
        ax.text(f + 0.01, i, f"{f:.2f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score on the test split")
    ax.grid(axis="y", visible=False)
    ax.set_title("Tuned KNN: per-genre F1 (from the saved classification report)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "music_knn_per_genre_f1.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    music_figures()
    print("figures written to", OUT)
