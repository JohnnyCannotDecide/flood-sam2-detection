import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio

matplotlib.use("Agg")


def main() -> None:
    files = [
        ("2024-07-27", r"d:\workPlace\graduate\d27.tif"),
        ("2024-07-28", r"d:\workPlace\graduate\d28.tif"),
        ("2024-07-29", r"d:\workPlace\graduate\d29.tif"),
    ]
    out_path = r"d:\workPlace\graduate\output\fig_sar_gray_histograms.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    histograms = []
    for label, fp in files:
        with rasterio.open(fp) as src:
            arr = src.read(1).astype(np.float32)
        gray = np.clip(arr, 0, 255).astype(np.uint8).reshape(-1)
        hist = np.bincount(gray, minlength=256).astype(np.int64)
        histograms.append((label, hist))

    all_counts = np.concatenate([h for _, h in histograms])
    robust_ymax = float(np.percentile(all_counts, 99.5)) * 1.15
    common_ymax = max(10.0, robust_ymax)
    clipped_any = any(float(h.max()) > common_ymax for _, h in histograms)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180, sharey=True)
    x = np.arange(256)
    for ax, (label, hist) in zip(axes, histograms):
        ax.bar(x, hist, width=1.0, color="#4C78A8", alpha=0.9)
        ax.set_title(label)
        ax.set_xlabel("Gray value (include NoData)")
        ax.set_ylabel("Pixel count")
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_xlim(0, 255)
        ax.set_ylim(0, common_ymax)

    if clipped_any:
        fig.text(0.5, 0.01, "Y-axis uses a shared robust upper bound (99.5th percentile) for visual comparability.",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
