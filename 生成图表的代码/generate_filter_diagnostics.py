import csv
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio

matplotlib.use("Agg")


def detect_nodata(data: np.ndarray) -> np.ndarray:
    h, w = data.shape
    zero = (data == 0).astype(np.uint8)
    ff_img = zero.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    for row in range(h):
        nz = np.where(zero[row] == 1)[0]
        if len(nz) and nz[-1] >= w - 200:
            cv2.floodFill(ff_img, ff_mask, (int(nz[-1]), row), newVal=2)
    for col in range(w):
        nz = np.where(zero[:10, col] == 1)[0]
        if len(nz):
            cv2.floodFill(ff_img, ff_mask, (col, int(nz[0])), newVal=2)
    return ff_img == 2


def enhance(arr: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    valid = arr[~nodata_mask]
    if valid.size == 0:
        return arr.astype(np.uint8)
    p2, p98 = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
    s = np.clip((arr.astype(np.float32) - p2) / max(p98 - p2, 1) * 255, 0, 255).astype(np.uint8)
    log_e = (np.log1p(s.astype(np.float32)) / np.log1p(255) * 255).astype(np.uint8)
    gamma_e = (np.power(s.astype(np.float32) / 255.0, 0.35) * 255).astype(np.uint8)
    out = ((log_e.astype(np.float32) + gamma_e.astype(np.float32)) / 2).astype(np.uint8)
    out[nodata_mask] = 0
    return out


def guided_filter(arr: np.ndarray, radius: int = 6, eps: float = 0.02) -> np.ndarray:
    g = arr.astype(np.float32) / 255.0

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, -1, (2 * radius + 1, 2 * radius + 1), normalize=True)

    m_i = box(g)
    var_i = box(g * g) - m_i**2
    a = var_i / (var_i + eps)
    b = m_i * (1 - a)
    return np.clip((box(a) * g + box(b)) * 255, 0, 255).astype(np.uint8)


def local_variance_mean(img: np.ndarray, valid_mask: np.ndarray, ksize: int = 5) -> float:
    f = img.astype(np.float32)
    mean = cv2.blur(f, (ksize, ksize))
    mean2 = cv2.blur(f * f, (ksize, ksize))
    var = np.maximum(mean2 - mean * mean, 0.0)
    vals = var[valid_mask]
    return float(vals.mean()) if vals.size else 0.0


def main() -> None:
    files = [
        ("2024-07-27", r"d:\workPlace\graduate\d27.tif"),
        ("2024-07-28", r"d:\workPlace\graduate\d28.tif"),
        ("2024-07-29", r"d:\workPlace\graduate\d29.tif"),
    ]
    out_dir = r"d:\workPlace\graduate\output"
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "fig_filter_diff_amplified.png")
    csv_path = os.path.join(out_dir, "table_filter_local_variance.csv")
    md_path = os.path.join(out_dir, "table_filter_local_variance.md")

    rows = []
    vis_rows = []
    gain = 4.0
    for date_str, fp in files:
        with rasterio.open(fp) as src:
            arr = src.read(1).astype(np.float32)
        nodata = detect_nodata(arr)
        enh = enhance(arr, nodata)
        enh_fill = enh.copy()
        enh_fill[nodata] = 128
        gf = guided_filter(enh_fill, radius=6, eps=0.02)
        gf[nodata] = 0

        diff = cv2.absdiff(gf, enh)
        diff_amp = np.clip(diff.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        diff_amp[nodata] = 0

        valid = ~nodata
        var_before = local_variance_mean(enh, valid, ksize=5)
        var_after = local_variance_mean(gf, valid, ksize=5)
        reduce_pct = (1.0 - var_after / max(var_before, 1e-9)) * 100.0

        rows.append((date_str, var_before, var_after, reduce_pct))
        vis_rows.append((enh, gf, diff_amp))

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), dpi=180)
    for i, (enh, gf, diff_amp) in enumerate(vis_rows):
        axes[i, 0].imshow(enh, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(gf, cmap="gray", vmin=0, vmax=255)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(diff_amp, cmap="magma", vmin=0, vmax=255)
        axes[i, 2].axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["影像日期", "局部方差均值(滤波前)", "局部方差均值(滤波后)", "方差下降比例(%)"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.2f}"])

    lines = [
        "# 局部方差对比表（5x5窗口）",
        "",
        "| 影像日期 | 局部方差均值(滤波前) | 局部方差均值(滤波后) | 方差下降比例(%) |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:.2f} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(fig_path)
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
