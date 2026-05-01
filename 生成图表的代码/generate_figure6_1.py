import os
import argparse

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


def load_nodata_from_cache(label: str, shape: tuple[int, int]) -> np.ndarray | None:
    cache_path = os.path.join(r"d:\workPlace\graduate\output", f"{label}_mask_cache.npz")
    if not os.path.exists(cache_path):
        return None
    try:
        z = np.load(cache_path)
        if "nodata_mask" not in z.files:
            return None
        nodata = z["nodata_mask"].astype(bool)
        if nodata.shape != shape:
            return None
        return nodata
    except Exception:
        return None


def percentile_stretch(arr: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    valid = arr[~nodata_mask]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    p2, p98 = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
    out = np.clip((arr.astype(np.float32) - p2) / max(p98 - p2, 1.0) * 255.0, 0, 255).astype(np.uint8)
    out[nodata_mask] = 0
    return out


def log_gamma_fusion(stretched: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    log_e = (np.log1p(stretched.astype(np.float32)) / np.log1p(255.0) * 255.0).astype(np.uint8)
    gamma_e = (np.power(stretched.astype(np.float32) / 255.0, 0.35) * 255.0).astype(np.uint8)
    out = ((log_e.astype(np.float32) + gamma_e.astype(np.float32)) / 2.0).astype(np.uint8)
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
    out = np.clip((box(a) * g + box(b)) * 255.0, 0, 255).astype(np.uint8)
    return out


def build_enhanced(arr: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    stretched = percentile_stretch(arr, nodata_mask)
    fused = log_gamma_fusion(stretched, nodata_mask)
    fill = fused.copy()
    fill[nodata_mask] = 128
    denoised = guided_filter(fill, radius=6, eps=0.02)
    denoised[nodata_mask] = 0
    return denoised


def normalize_raw(arr: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    valid = arr[~nodata_mask]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    p1, p99 = float(np.percentile(valid, 1)), float(np.percentile(valid, 99))
    out = np.clip((arr.astype(np.float32) - p1) / max(p99 - p1, 1.0) * 255.0, 0, 255).astype(np.uint8)
    out[nodata_mask] = 0
    return out


def resize_for_figure(img: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return img
    h, w = img.shape
    th, tw = max(1, h // scale), max(1, w // scale)
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成图6-1（3x4增强对比图）")
    parser.add_argument("--scale", type=int, default=8, help="绘图缩放倍数，越大越快")
    parser.add_argument(
        "--out-path",
        default=r"d:\workPlace\graduate\output\fig6_1_enhancement_compare_3x4.png",
        help="输出路径",
    )
    args = parser.parse_args()

    files = [
        r"d:\workPlace\graduate\d27.tif",
        r"d:\workPlace\graduate\d28.tif",
        r"d:\workPlace\graduate\d29.tif",
    ]
    out_path = args.out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    for fp in files:
        label = os.path.splitext(os.path.basename(fp))[0]
        print(f"[{label}] 读取影像...")
        with rasterio.open(fp) as src:
            arr = src.read(1).astype(np.float32)
        nodata = load_nodata_from_cache(label, arr.shape)
        if nodata is None:
            print(f"[{label}] 缓存未命中，执行 NoData 检测（较慢）...")
            nodata = detect_nodata(arr)
        else:
            print(f"[{label}] 复用缓存 NoData 掩膜")
        print(f"[{label}] 执行增强链...")
        raw = normalize_raw(arr, nodata)
        stretched = percentile_stretch(arr, nodata)
        fused = log_gamma_fusion(stretched, nodata)
        fill = fused.copy()
        fill[nodata] = 128
        denoised = guided_filter(fill, radius=6, eps=0.02)
        denoised[nodata] = 0
        rows.append(
            (
                resize_for_figure(raw, int(args.scale)),
                resize_for_figure(stretched, int(args.scale)),
                resize_for_figure(fused, int(args.scale)),
                resize_for_figure(denoised, int(args.scale)),
            )
        )

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12), dpi=150)
    for i in range(3):
        axes[i, 0].imshow(rows[i][0], cmap="gray", vmin=0, vmax=255)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(rows[i][1], cmap="gray", vmin=0, vmax=255)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(rows[i][2], cmap="gray", vmin=0, vmax=255)
        axes[i, 2].axis("off")
        axes[i, 3].imshow(rows[i][3], cmap="gray", vmin=0, vmax=255)
        axes[i, 3].axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
