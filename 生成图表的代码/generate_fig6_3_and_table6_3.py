import csv
import os
from typing import Dict, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio

matplotlib.use("Agg")

PIXEL_AREA_M2 = 25.0
MIN_WATER_AREA = 300
MORPH_CLOSE_K = 9
POST_DILATE_K = 7


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
    if len(valid) == 0:
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


def build_prob(arr: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    enh = enhance(arr, nodata_mask)
    enh_fill = enh.copy()
    enh_fill[nodata_mask] = 128
    gf = guided_filter(enh_fill)
    valid_vals = gf[~nodata_mask].flatten().astype(np.uint8)
    otsu, _ = cv2.threshold(valid_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = int(min(otsu * 0.6, 45))
    water_cand = (gf < thresh) & ~nodata_mask & (arr > 0)
    deep_water = (arr > 0) & (arr <= 8) & ~nodata_mask
    lr = cv2.dilate(gf, np.ones((11, 11), np.uint8)) - cv2.erode(gf, np.ones((11, 11), np.uint8))
    shadow = water_cand & (lr < 15) & ~deep_water
    water_final = (water_cand & ~shadow) | deep_water
    prob = np.zeros(arr.shape, np.float32)
    prob[water_final] = 1.0
    prob[shadow] = 0.2
    prob[deep_water] = 1.0
    prob[nodata_mask] = 0.0
    prob = cv2.GaussianBlur(prob, (7, 7), 2)
    prob[nodata_mask] = 0.0
    return prob


def postprocess_mask(raw: np.ndarray, nodata_mask: np.ndarray, post_dilate_k: int = POST_DILATE_K) -> np.ndarray:
    k = np.ones((MORPH_CLOSE_K, MORPH_CLOSE_K), np.uint8)
    closed = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, k)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if int(post_dilate_k) > 0:
        kd = np.ones((int(post_dilate_k), int(post_dilate_k)), np.uint8)
        opened = cv2.dilate(opened, kd, iterations=1)
    num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    final = np.zeros_like(opened)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_WATER_AREA:
            final[labels_map == i] = 1
    final = final.astype(bool) & ~nodata_mask
    return final


def load_from_cache(label: str, out_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    path = os.path.join(out_dir, f"{label}_mask_cache.npz")
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path)
        if "water_mask" not in z.files or "nodata_mask" not in z.files:
            return None
        water = z["water_mask"].astype(bool)
        nodata = z["nodata_mask"].astype(bool)
        prob = z["prob"].astype(np.float32) if "prob" in z.files else None
        return water, nodata, prob
    except Exception:
        return None


def render_mask(mask: np.ndarray, nodata: np.ndarray) -> np.ndarray:
    img = np.zeros(mask.shape, dtype=np.uint8)
    img[mask] = 255
    img[nodata] = 35
    return img


def calc_stats(mask: np.ndarray) -> Dict[str, float]:
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = int(len(contours))

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comp_count = int(max(0, num_labels - 1))
    max_comp_px = int(stats[1:, cv2.CC_STAT_AREA].max()) if comp_count > 0 else 0

    return {
        "water_area_km2": float(mask.sum() * PIXEL_AREA_M2 / 1e6),
        "contour_count": contour_count,
        "max_component_area_km2": float(max_comp_px * PIXEL_AREA_M2 / 1e6),
        "component_count": comp_count,
    }


def main() -> None:
    root = r"d:\workPlace\graduate"
    out_dir = os.path.join(root, "output")
    files = [
        ("2024-07-27", "d27", os.path.join(root, "d27.tif")),
        ("2024-07-28", "d28", os.path.join(root, "d28.tif")),
        ("2024-07-29", "d29", os.path.join(root, "d29.tif")),
    ]

    # 1) 图6-3：d28 形态学前后对比
    date_28, label_28, fp_28 = files[1]
    cache_28 = load_from_cache(label_28, out_dir)
    if cache_28 is not None:
        _, nodata_28, prob_28 = cache_28
        if prob_28 is None:
            with rasterio.open(fp_28) as src:
                arr_28 = src.read(1).astype(np.float32)
            prob_28 = build_prob(arr_28, nodata_28)
    else:
        with rasterio.open(fp_28) as src:
            arr_28 = src.read(1).astype(np.float32)
        nodata_28 = detect_nodata(arr_28)
        prob_28 = build_prob(arr_28, nodata_28)

    raw_28 = (prob_28 > 0.55) & ~nodata_28
    post_28 = postprocess_mask(raw_28, nodata_28, post_dilate_k=POST_DILATE_K)

    fig_path = os.path.join(out_dir, "fig6_3_d28_morph_compare.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=180)
    axes[0].imshow(render_mask(raw_28, nodata_28), cmap="gray", vmin=0, vmax=255)
    axes[0].axis("off")
    axes[1].imshow(render_mask(post_28, nodata_28), cmap="gray", vmin=0, vmax=255)
    axes[1].axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # 2) 表6-3：三景水体面积与连通域统计（使用缓存中的最终掩膜）
    rows = []
    for date_str, label, fp in files:
        cache = load_from_cache(label, out_dir)
        if cache is not None:
            water_mask, _, _ = cache
        else:
            with rasterio.open(fp) as src:
                arr = src.read(1).astype(np.float32)
            nodata = detect_nodata(arr)
            prob = build_prob(arr, nodata)
            raw = (prob > 0.55) & ~nodata
            water_mask = postprocess_mask(raw, nodata, post_dilate_k=POST_DILATE_K)
        s = calc_stats(water_mask)
        rows.append(
            (
                date_str,
                s["water_area_km2"],
                int(s["contour_count"]),
                s["max_component_area_km2"],
                int(s["component_count"]),
            )
        )

    csv_path = os.path.join(out_dir, "table6_3_water_stats.csv")
    md_path = os.path.join(out_dir, "table6_3_water_stats.md")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["影像日期", "检测水体面积（km²）", "水体轮廓数量", "最大连通域面积（km²）", "连通域总数"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", r[2], f"{r[3]:.4f}", r[4]])

    lines = [
        "# 表6-3 三景影像水体面积检测结果",
        "",
        "| 影像日期 | 检测水体面积（km²） | 水体轮廓数量 | 最大连通域面积（km²） | 连通域总数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]:.4f} | {r[2]} | {r[3]:.4f} | {r[4]} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("DONE")
    print(fig_path)
    print(csv_path)
    print(md_path)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
