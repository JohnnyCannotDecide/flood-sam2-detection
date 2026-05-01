"""
pipeline_showcase.py
功能：生成洪水检测流程的关键中间结果对比图与展示页面（用于答辩/汇报）。
"""
import argparse
import os
from typing import Dict, List, Tuple

import cv2
import matplotlib
import numpy as np
import rasterio
import torch

from flood_detection import (
    DEFAULT_PIXEL_AREA_M2,
    DEFAULT_POST_DILATE_K,
    MIN_WATER_AREA,
    MORPH_CLOSE_K,
    N_BG,
    N_FG,
    OVERLAP,
    SAM2_CONF_MIN_DEFAULT,
    SAM2_SCORE_MIN,
    UNCERTAIN_HIGH,
    UNCERTAIN_LOW,
    WINDOW_SIZE,
    build_water_prob,
    cos_win,
    detect_nodata,
    enhance,
    guided_filter,
    init_sam2_predictor,
    segment_image,
    sample_prob,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def normalize_to_uint8(arr: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    valid = arr[~nodata_mask]
    out = np.zeros_like(arr, dtype=np.uint8)
    if valid.size == 0:
        return out
    p2, p98 = np.percentile(valid, [2, 98])
    scale = np.clip((arr.astype(np.float32) - float(p2)) / max(float(p98 - p2), 1e-6), 0, 1)
    out = (scale * 255).astype(np.uint8)
    out[nodata_mask] = 0
    return out


def overlay_mask(
    gray: np.ndarray,
    mask: np.ndarray,
    nodata_mask: np.ndarray = None,
    color: Tuple[int, int, int] = (30, 120, 220),
    alpha: float = 0.75,
) -> np.ndarray:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if nodata_mask is not None:
        rgb[nodata_mask] = [20, 20, 20]
    idx = mask.astype(bool)
    rgb[idx] = (rgb[idx].astype(np.float32) * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return rgb


def save_pair(left: np.ndarray, right: np.ndarray, left_title: str, right_title: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(left, cmap="gray" if left.ndim == 2 else None)
    axes[0].set_title(left_title, fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(right, cmap="gray" if right.ndim == 2 else None)
    axes[1].set_title(right_title, fontsize=11)
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def build_prob_components(arr: np.ndarray, nodata_mask: np.ndarray) -> Dict[str, np.ndarray]:
    enh = enhance(arr, nodata_mask)
    enh_fill = enh.copy()
    enh_fill[nodata_mask] = 128
    gf = guided_filter(enh_fill)
    valid_vals = gf[~nodata_mask].astype(np.uint8)
    otsu, _ = cv2.threshold(valid_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = int(min(float(otsu) * 0.6, 45))
    water_cand = (gf < thresh) & ~nodata_mask & (arr > 0)
    deep_water = (arr > 0) & (arr <= 8) & ~nodata_mask
    local_range = cv2.dilate(gf, np.ones((11, 11), np.uint8)) - cv2.erode(gf, np.ones((11, 11), np.uint8))
    shadow = water_cand & (local_range < 15) & ~deep_water
    no_shadow = water_cand & ~shadow
    water_final = no_shadow | deep_water
    prob = np.zeros(arr.shape, np.float32)
    prob[water_final] = 1.0
    prob[shadow] = 0.2
    prob[deep_water] = 1.0
    prob[nodata_mask] = 0.0
    prob = cv2.GaussianBlur(prob, (7, 7), 2)
    prob[nodata_mask] = 0.0
    return {
        "enh": enh,
        "enh_fill": enh_fill,
        "gf": gf,
        "water_cand": water_cand,
        "deep_water": deep_water,
        "shadow": shadow,
        "no_shadow": no_shadow,
        "water_final": water_final,
        "prob": prob,
        "thresh": np.array([thresh], dtype=np.int32),
    }


def postprocess_mask(raw: np.ndarray, nodata_mask: np.ndarray, post_dilate_k: int) -> np.ndarray:
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
    final = final.astype(bool)
    final[nodata_mask] = False
    return final


def get_sliding_positions(h: int, w: int) -> Tuple[List[int], List[int]]:
    step = max(1, WINDOW_SIZE - OVERLAP)
    ys = list(range(0, max(h - WINDOW_SIZE + 1, 1), step))
    xs = list(range(0, max(w - WINDOW_SIZE + 1, 1), step))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    if ys[-1] + WINDOW_SIZE < h:
        ys.append(max(0, h - WINDOW_SIZE))
    if xs[-1] + WINDOW_SIZE < w:
        xs.append(max(0, w - WINDOW_SIZE))
    return ys, xs


def segment_and_trace(
    predictor,
    enh: np.ndarray,
    prob: np.ndarray,
    nodata_mask: np.ndarray,
) -> Dict[str, object]:
    h, w = enh.shape
    ys, xs = get_sliding_positions(h, w)
    vote = np.zeros((h, w), np.float32)
    wt = np.zeros((h, w), np.float32)
    fg_points: List[Tuple[int, int]] = []
    bg_points: List[Tuple[int, int]] = []
    win_full = cos_win(WINDOW_SIZE)
    windows = []
    sam2_ok = 0
    total = len(ys) * len(xs)

    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + WINDOW_SIZE, h), min(x0 + WINDOW_SIZE, w)
            pi = enh[y0:y1, x0:x1]
            pp = prob[y0:y1, x0:x1]
            pn = nodata_mask[y0:y1, x0:x1]
            windows.append((x0, y0, x1, y1))
            if pn.size == 0 or ((~pn).sum() / pn.size) < 0.05 or (pp > 0.5).sum() < 5:
                continue

            pv = pp.copy()
            pv[pn] = 0.0
            fg = sample_prob(pv, N_FG)
            bg = sample_prob(pv, N_BG, high=False)
            fg_points.extend([(x0 + int(x), y0 + int(y)) for x, y in fg])
            bg_points.extend([(x0 + int(x), y0 + int(y)) for x, y in bg])

            try:
                if predictor is not None:
                    predictor.set_image(cv2.cvtColor(pi, cv2.COLOR_GRAY2RGB))
                    masks, scores, _ = predictor.predict(
                        point_coords=np.array(fg + bg),
                        point_labels=np.array([1] * N_FG + [0] * N_BG),
                        multimask_output=True,
                    )
                    best = int(np.argmax(scores))
                    mask = masks[best].astype(bool)
                    score = float(scores[best])
                    if score >= SAM2_SCORE_MIN:
                        sam2_ok += 1
                    else:
                        mask = pv > 0.55
                        score = 0.3
                else:
                    mask = pv > 0.55
                    score = 0.3
            except Exception:
                mask = pv > 0.55
                score = 0.3

            mask[pn] = False
            ww = max(float(score), 0.3)
            win_crop = win_full[: (y1 - y0), : (x1 - x0)]
            vote[y0:y1, x0:x1] += mask.astype(np.float32) * win_crop * ww
            wt[y0:y1, x0:x1] += win_crop * ww

    raw = (vote / np.where(wt > 0, wt, 1e-6)) > 0.5
    raw[nodata_mask] = False
    return {
        "raw": raw,
        "windows": windows,
        "fg_points": fg_points,
        "bg_points": bg_points,
        "sam2_ok": sam2_ok,
        "total_windows": total,
    }


def segment_upgraded(
    predictor,
    arr: np.ndarray,
    nodata_mask: np.ndarray,
    label: str,
    out_dir: str,
    post_dilate_k: int,
    sam2_conf_min: float,
    enable_crf: bool,
    crf_iter: int,
) -> Dict[str, object]:
    prob_new, enh_new, prob_path, guidance, extra_paths = build_water_prob(
        arr, nodata_mask, label + "_upgraded", out_dir, return_guidance=True
    )
    with torch.inference_mode():
        final_mask, area_km2 = segment_image(
            predictor,
            arr,
            nodata_mask,
            prob_new,
            enh_new,
            label + "_upgraded",
            post_dilate_k=post_dilate_k,
            guidance=guidance,
            sam2_conf_min=sam2_conf_min,
            enable_crf=enable_crf,
            crf_iter=crf_iter,
        )
    return {
        "prob": prob_new,
        "enh": enh_new,
        "guidance": guidance,
        "prob_path": prob_path,
        "edge_path": extra_paths["edge_path"],
        "uncertain_path": extra_paths["uncertain_path"],
        "final_mask": final_mask,
        "area_km2": float(area_km2),
    }


def save_windows_visual(enh: np.ndarray, windows: List[Tuple[int, int, int, int]], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(enh, cmap="gray", vmin=0, vmax=255)
    for (x0, y0, x1, y1) in windows:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=0.45, edgecolor="#f59e0b", facecolor="none", alpha=0.35))
    ax.set_title(f"滑动窗口覆盖示意（窗口数: {len(windows)}）", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def save_prompt_visual(
    enh: np.ndarray,
    fg_points: List[Tuple[int, int]],
    bg_points: List[Tuple[int, int]],
    out_path: str,
    max_show: int = 1200,
) -> None:
    def subset(points: List[Tuple[int, int]]) -> np.ndarray:
        if not points:
            return np.zeros((0, 2), dtype=np.int32)
        arr = np.array(points, dtype=np.int32)
        if arr.shape[0] > max_show:
            idx = np.random.choice(arr.shape[0], max_show, replace=False)
            arr = arr[idx]
        return arr

    fg = subset(fg_points)
    bg = subset(bg_points)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(enh, cmap="gray", vmin=0, vmax=255)
    if fg.size:
        ax.scatter(fg[:, 0], fg[:, 1], s=8, c="#22c55e", label=f"前景提示点 ({len(fg_points)})", alpha=0.75)
    if bg.size:
        ax.scatter(bg[:, 0], bg[:, 1], s=8, c="#ef4444", label=f"背景提示点 ({len(bg_points)})", alpha=0.75)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("自动提示点位置示意（随机抽样显示）", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def save_prob_components_figure(components: Dict[str, np.ndarray], nodata_mask: np.ndarray, out_path: str) -> None:
    enh = components["enh"]
    water_cand = components["water_cand"]
    deep_water = components["deep_water"]
    shadow = components["shadow"]
    water_final = components["water_final"]
    prob = components["prob"]
    deep_gain = deep_water & ~(components["no_shadow"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes[0, 0].imshow(overlay_mask(enh, water_cand, nodata_mask, color=(30, 120, 220), alpha=0.7))
    axes[0, 0].set_title("概率候选水体（阈值后）", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(overlay_mask(enh, shadow, nodata_mask, color=(220, 60, 60), alpha=0.8))
    axes[0, 1].set_title("阴影抑制区域（被剔除）", fontsize=10)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(overlay_mask(enh, deep_gain, nodata_mask, color=(60, 200, 80), alpha=0.8))
    axes[0, 2].set_title("深水强制判决新增区域", fontsize=10)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(overlay_mask(enh, water_final, nodata_mask, color=(30, 120, 220), alpha=0.8))
    axes[1, 0].set_title("概率图最终水体", fontsize=10)
    axes[1, 0].axis("off")

    im = axes[1, 1].imshow(prob, cmap="Blues", vmin=0, vmax=1)
    axes[1, 1].set_title("平滑后水体概率图", fontsize=10)
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    txt = (
        f"阈值: {int(components['thresh'][0])}\n"
        f"候选水体像素: {int(water_cand.sum())}\n"
        f"阴影剔除像素: {int(shadow.sum())}\n"
        f"深水强制新增: {int(deep_gain.sum())}\n"
        f"最终水体像素: {int(water_final.sum())}"
    )
    axes[1, 2].axis("off")
    axes[1, 2].text(0.02, 0.95, txt, va="top", ha="left", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def save_key_compare_figure(
    enh: np.ndarray,
    prob_before: np.ndarray,
    prob_after: np.ndarray,
    uncertain_after: np.ndarray,
    edge_after: np.ndarray,
    mask_before: np.ndarray,
    mask_after: np.ndarray,
    nodata_mask: np.ndarray,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    im0 = axes[0, 0].imshow(prob_before, cmap="Blues", vmin=0, vmax=1)
    axes[0, 0].set_title("改造前：概率图", fontsize=11)
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(prob_after, cmap="Blues", vmin=0, vmax=1)
    axes[0, 1].set_title("改造后：概率图", fontsize=11)
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    unc_rgb = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
    unc_rgb[uncertain_after] = [245, 158, 11]
    unc_rgb[nodata_mask] = [20, 20, 20]
    axes[0, 2].imshow(unc_rgb)
    axes[0, 2].set_title("改造后：不确定区", fontsize=11)
    axes[0, 2].axis("off")

    edge_rgb = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
    edge_rgb[edge_after] = [255, 200, 60]
    edge_rgb[nodata_mask] = [20, 20, 20]
    axes[1, 0].imshow(edge_rgb)
    axes[1, 0].set_title("改造后：边界图", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay_mask(enh, mask_before, nodata_mask, color=(30, 120, 220), alpha=0.78))
    axes[1, 1].set_title("改造前：最终掩膜", fontsize=11)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(overlay_mask(enh, mask_after, nodata_mask, color=(30, 120, 220), alpha=0.78))
    axes[1, 2].set_title("改造后：最终掩膜", fontsize=11)
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def write_compare_summary(
    out_dir: str,
    label: str,
    area_before: float,
    area_after: float,
    uncertain_ratio: float,
    edge_ratio: float,
) -> str:
    delta = area_after - area_before
    lines = [
        f"# {label} 改造前后对比说明",
        "",
        "## 关键图说明",
        "- 改造前概率图：使用旧版概率图+全窗口SAM2策略，作为基线。",
        "- 改造后概率图：沿用概率图主干，但增加不确定区与边界图供SAM2提示。",
        "- 改造后不确定区：仅在`0.4~0.6`区间调用SAM2，减少高置信区域被误改。",
        "- 改造后边界图：以边界带负样本约束提示点，抑制边界处漏判/误判。",
        "- 最终掩膜对比：展示改造前后分割结果变化。",
        "",
        "## 数值摘要",
        f"- 改造前面积：`{area_before:.4f} km²`",
        f"- 改造后面积：`{area_after:.4f} km²`",
        f"- 面积差值（后-前）：`{delta:+.4f} km²`",
        f"- 不确定区占比：`{uncertain_ratio:.2%}`",
        f"- 边界像素占比：`{edge_ratio:.2%}`",
        "",
        "## 结论建议",
        "- 若改造后边界更平滑且噪点更少，说明“不确定区SAM2+结构化提示”有效。",
        "- 若面积变化过大，建议提高`--sam2-conf-min`或关闭`--enable-crf`再复核。",
    ]
    path = os.path.join(out_dir, "compare_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_showcase_html(out_dir: str, title: str, sections: List[Tuple[str, str]]) -> str:
    html_path = os.path.join(out_dir, "showcase_report.html")
    cards = []
    for name, file_name in sections:
        cards.append(
            f"""
            <section class="card">
              <h3>{name}</h3>
              <img src="{file_name}" alt="{name}" />
            </section>
            """
        )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", Arial, sans-serif;
      background: #f8fafc;
      color: #0f172a;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
    }}
    .muted {{
      color: #475569;
      margin-bottom: 20px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 14px;
      margin: 0 0 16px;
      box-shadow: 0 2px 8px rgba(15,23,42,0.06);
    }}
    .card h3 {{
      margin: 0 0 10px;
      font-size: 17px;
      color: #1d4ed8;
    }}
    img {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid #e2e8f0;
      display: block;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{title}</h1>
    <div class="muted">该页面用于展示流程关键中间结果与对比效果，适合项目答辩或汇报演示。</div>
    {''.join(cards)}
  </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="洪水检测流程可视化展示脚本")
    parser.add_argument("--input-file", required=True, help="输入单景SAR影像路径（tif/tiff）")
    parser.add_argument("--out-dir", default=os.path.join("output", "showcase"), help="展示输出目录")
    parser.add_argument("--label", default=None, help="影像标签，默认使用文件名")
    parser.add_argument("--disable-sam2", action="store_true", help="禁用SAM2，仅演示概率图流程")
    parser.add_argument("--post-dilate-k", type=int, default=DEFAULT_POST_DILATE_K, help="后处理膨胀核大小")
    parser.add_argument("--sam2-conf-min", type=float, default=SAM2_CONF_MIN_DEFAULT, help="SAM2高置信阈值")
    parser.add_argument("--enable-crf", action="store_true", help="启用CRF后处理")
    parser.add_argument("--crf-iter", type=int, default=5, help="CRF迭代次数")
    parser.add_argument("--pixel-area-m2", type=float, default=DEFAULT_PIXEL_AREA_M2, help="像素面积（m²）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（影响提示点采样）")
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    os.makedirs(args.out_dir, exist_ok=True)
    label = args.label or os.path.splitext(os.path.basename(args.input_file))[0]

    with rasterio.open(args.input_file) as src:
        arr = src.read(1).astype(np.float32)

    nodata = detect_nodata(arr, label)
    base_u8 = normalize_to_uint8(arr, nodata)
    components = build_prob_components(arr, nodata)
    enh = components["enh"]
    prob = components["prob"]

    # 1) NoData 检测效果
    nodata_vis = overlay_mask(base_u8, nodata, None, color=(255, 80, 80), alpha=0.8)
    p1 = os.path.join(args.out_dir, "01_nodata_detection.png")
    save_pair(base_u8, nodata_vis, "原始归一化", "NoData检测结果（红色）", p1)

    # 2) 亮度增强对比
    p2 = os.path.join(args.out_dir, "02_brightness_enhancement.png")
    save_pair(base_u8, enh, "增强前", "增强后", p2)

    # 3) 引导滤波去噪效果
    p3 = os.path.join(args.out_dir, "03_guided_filter_denoise.png")
    save_pair(components["enh_fill"], components["gf"], "引导滤波前（填充NoData）", "引导滤波后", p3)

    # 4) 概率图构建细节
    p4 = os.path.join(args.out_dir, "04_probability_components.png")
    save_prob_components_figure(components, nodata, p4)

    # 5) 阴影区域抑制效果
    p5 = os.path.join(args.out_dir, "05_shadow_suppression.png")
    before_shadow = overlay_mask(enh, components["water_cand"], nodata, color=(30, 120, 220), alpha=0.75)
    after_shadow = overlay_mask(enh, components["no_shadow"], nodata, color=(30, 120, 220), alpha=0.75)
    save_pair(before_shadow, after_shadow, "阴影抑制前（候选水体）", "阴影抑制后", p5)

    # 6) 深水强制判决效果
    p6 = os.path.join(args.out_dir, "06_deep_water_forced.png")
    no_deep = overlay_mask(enh, components["no_shadow"], nodata, color=(30, 120, 220), alpha=0.75)
    with_deep = overlay_mask(enh, components["water_final"], nodata, color=(30, 120, 220), alpha=0.75)
    save_pair(no_deep, with_deep, "不含深水强制判决", "加入深水强制判决", p6)

    # 概率图-only分割（用于对比）
    trace_prob = segment_and_trace(None, enh, prob, nodata)
    prob_raw = trace_prob["raw"]
    prob_post = postprocess_mask(prob_raw, nodata, max(0, int(args.post_dilate_k)))

    # 7) SAM2分割与概率图分割对比
    predictor = init_sam2_predictor(enable_sam2=not args.disable_sam2)
    sam2_enabled = predictor is not None
    if sam2_enabled:
        with torch.inference_mode():
            trace_sam2 = segment_and_trace(predictor, enh, prob, nodata)
    else:
        trace_sam2 = trace_prob
    sam2_raw = trace_sam2["raw"]
    sam2_post = postprocess_mask(sam2_raw, nodata, max(0, int(args.post_dilate_k)))

    pixel_area_km2 = float(args.pixel_area_m2) / 1e6
    area_prob = float(prob_post.sum()) * pixel_area_km2
    area_sam2 = float(sam2_post.sum()) * pixel_area_km2
    p7 = os.path.join(args.out_dir, "07_prob_vs_sam2.png")
    left = overlay_mask(enh, prob_post, nodata, color=(30, 120, 220), alpha=0.78)
    right = overlay_mask(enh, sam2_post, nodata, color=(30, 120, 220), alpha=0.78)
    right_title = f"SAM2融合分割后处理 (面积={area_sam2:.3f} km²)" if sam2_enabled else "SAM2不可用，回退概率图分割"
    save_pair(left, right, f"仅概率图分割后处理 (面积={area_prob:.3f} km²)", right_title, p7)

    # 8) 滑动窗口可视化
    p8 = os.path.join(args.out_dir, "08_sliding_windows.png")
    save_windows_visual(enh, trace_sam2["windows"], p8)

    # 9) 自动提示点位置可视化
    p9 = os.path.join(args.out_dir, "09_auto_prompts.png")
    save_prompt_visual(enh, trace_sam2["fg_points"], trace_sam2["bg_points"], p9)

    # 10) 形态学前后对比
    p10 = os.path.join(args.out_dir, "10_morphology_before_after.png")
    morph_before = overlay_mask(enh, sam2_raw, nodata, color=(30, 120, 220), alpha=0.75)
    morph_after = overlay_mask(enh, sam2_post, nodata, color=(30, 120, 220), alpha=0.75)
    save_pair(morph_before, morph_after, "后处理前（raw mask）", f"后处理后（dilate_k={max(0,int(args.post_dilate_k))}）", p10)

    # 11) 改造前后关键图对照（概率图/不确定区/边界图/最终掩膜）
    upgraded = segment_upgraded(
        predictor,
        arr,
        nodata,
        label,
        args.out_dir,
        max(0, int(args.post_dilate_k)),
        float(args.sam2_conf_min),
        bool(args.enable_crf),
        max(1, int(args.crf_iter)),
    )
    p11 = os.path.join(args.out_dir, "11_key_maps_before_after.png")
    save_key_compare_figure(
        enh=upgraded["enh"],
        prob_before=prob,
        prob_after=upgraded["prob"],
        uncertain_after=upgraded["guidance"]["uncertain_mask"],
        edge_after=upgraded["guidance"]["edge_map"],
        mask_before=sam2_post,
        mask_after=upgraded["final_mask"],
        nodata_mask=nodata,
        out_path=p11,
    )

    # 12) 最终掩膜前后差异
    p12 = os.path.join(args.out_dir, "12_final_mask_before_after.png")
    before_title = f"改造前最终掩膜 (面积={area_sam2:.3f} km²)"
    after_title = f"改造后最终掩膜 (面积={upgraded['area_km2']:.3f} km²)"
    save_pair(
        overlay_mask(enh, sam2_post, nodata, color=(30, 120, 220), alpha=0.78),
        overlay_mask(upgraded["enh"], upgraded["final_mask"], nodata, color=(30, 120, 220), alpha=0.78),
        before_title,
        after_title,
        p12,
    )

    summary_path = write_compare_summary(
        args.out_dir,
        label,
        area_before=area_sam2,
        area_after=upgraded["area_km2"],
        uncertain_ratio=float(upgraded["guidance"]["uncertain_mask"].mean()),
        edge_ratio=float(upgraded["guidance"]["edge_map"].mean()),
    )

    sections = [
        ("NoData 区检测结果", os.path.basename(p1)),
        ("亮度增强对比", os.path.basename(p2)),
        ("引导滤波去噪效果", os.path.basename(p3)),
        ("水体概率图构建与中间组件", os.path.basename(p4)),
        ("阴影抑制效果", os.path.basename(p5)),
        ("深水强制判决效果", os.path.basename(p6)),
        ("仅概率图 vs 概率图+SAM2 对比", os.path.basename(p7)),
        ("滑动窗口覆盖可视化", os.path.basename(p8)),
        ("自动提示点位置可视化", os.path.basename(p9)),
        ("形态学后处理前后对比", os.path.basename(p10)),
        ("改造前后关键图对照（概率/不确定区/边界/掩膜）", os.path.basename(p11)),
        ("改造前后最终掩膜对比", os.path.basename(p12)),
    ]
    html_path = write_showcase_html(args.out_dir, f"{label} 流程效果展示", sections)

    print("=" * 60)
    print(f"展示结果已生成: {args.out_dir}")
    print(f"汇总页面: {html_path}")
    print(f"对比说明: {summary_path}")
    print(f"SAM2状态: {'启用' if sam2_enabled else '不可用/已禁用(回退概率图)'}")
    print(f"SAM2窗口生效: {trace_sam2['sam2_ok']}/{trace_sam2['total_windows']} 块")
    print("=" * 60)


if __name__ == "__main__":
    main()
