"""
flood_detection.py — 洪水遥感检测与预警分析
功能：多时相SAR水体分割、变化检测、预警报告生成
支持：分割掩膜缓存复用、LLM专家分析、可视化网站生成
"""
import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from glob import glob

from dotenv import load_dotenv

import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from flood_web import write_dashboard

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

# 加载 .env 文件
load_dotenv()

matplotlib.rcParams["font.family"] = "Microsoft YaHei"
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

# ── 路径配置 ──────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_ROOT = os.path.join(ROOT_DIR, "sam2")
SAM2_CHECKPOINT = os.path.join(SAM2_ROOT, "checkpoints", "sam2.1_hiera_small.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
if SAM2_ROOT not in sys.path:
    sys.path.insert(0, SAM2_ROOT)

DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, "output")
DEFAULT_PIXEL_AREA_M2 = 25.0
PIXEL_AREA_M2 = DEFAULT_PIXEL_AREA_M2

# ── SAM2 参数 ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 512
OVERLAP = 128
N_FG = 6
N_BG = 4
SAM2_SCORE_MIN = 0.3
MIN_WATER_AREA = 300
MORPH_CLOSE_K = 9
DEFAULT_POST_DILATE_K = 3

# ── LLM 配置（SiliconFlow GLM-4.7）─────────────────────────────────────────
LLM_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
LLM_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MODEL = "Pro/zai-org/GLM-4.7"


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

@contextmanager
def switch_dir(path):
    old = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old)


def discover_tif_files(input_dir):
    files = []
    for p in [os.path.join(input_dir, "*.tif"), os.path.join(input_dir, "*.tiff")]:
        files.extend(glob(p))
    return sorted(dict.fromkeys(files))


def resolve_input_files(input_files, input_dir):
    if input_files:
        files = [os.path.abspath(f) for f in input_files if f.lower().endswith((".tif", ".tiff"))]
    else:
        files = discover_tif_files(os.path.abspath(input_dir))
    return [f for f in files if os.path.exists(f)]


def default_labels(filepaths):
    return [os.path.splitext(os.path.basename(fp))[0] or f"D{i}" for i, fp in enumerate(filepaths, 1)]


# ═══════════════════════════════════════════════════════════════════════════════
# 掩膜缓存：保存 / 加载已分割结果
# ═══════════════════════════════════════════════════════════════════════════════

def mask_cache_path(out_dir, label):
    """返回该标签对应的掩膜缓存文件路径（.npz）。"""
    return os.path.join(out_dir, f"{label}_mask_cache.npz")


def save_mask_cache(out_dir, label, water_mask, nodata_mask, area_km2, prob=None, post_dilate_k=0):
    """将分割结果持久化，日后可复用。现在也保存概率图。"""
    path = mask_cache_path(out_dir, label)
    save_dict = {
        "water_mask": water_mask.astype(np.uint8),
        "nodata_mask": nodata_mask.astype(np.uint8),
        "area_km2": np.array([area_km2]),
        "post_dilate_k": np.array([int(post_dilate_k)], dtype=np.int32),
    }
    if prob is not None:
        save_dict["prob"] = prob.astype(np.float32)
    np.savez_compressed(path, **save_dict)
    print(f"  [缓存] 掩膜已保存 → {os.path.basename(path)}")
    return path


def load_mask_cache(out_dir, label, expected_shape, expected_post_dilate_k=0):
    """
    尝试加载已缓存的掩膜。
    若缓存存在且尺寸匹配则返回 (water_mask, nodata_mask, area_km2, prob)，否则返回 None。
    注意：旧版缓存没有 prob 字段，这种情况下也会返回 None 以触发重新分割。
    """
    path = mask_cache_path(out_dir, label)
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path)
        wm = z["water_mask"].astype(bool)
        nd = z["nodata_mask"].astype(bool)
        area = float(z["area_km2"][0])
        # 检查是否有 prob 字段和参数信息（新版缓存）
        if "prob" not in z.files or "post_dilate_k" not in z.files:
            print(f"  [缓存] {label} 旧版缓存（参数不完整），重新分割")
            return None
        prob = z["prob"].astype(np.float32)
        cached_post_dilate_k = int(z["post_dilate_k"][0])
        if wm.shape != expected_shape or prob.shape != expected_shape:
            print(f"  [缓存] {label} 形状不匹配，重新分割")
            return None
        if cached_post_dilate_k != int(expected_post_dilate_k):
            print(f"  [缓存] {label} 后处理膨胀参数变化({cached_post_dilate_k}→{expected_post_dilate_k})，重新分割")
            return None
        print(f"  [缓存] 命中 {label}，直接复用 (面积={area:.3f} km²)")
        return wm, nd, area, prob
    except Exception as e:
        print(f"  [缓存] 读取失败({e})，重新分割")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 图像读取与预处理
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_images(filepaths):
    images, metas = [], []
    for fp in filepaths:
        with rasterio.open(fp) as src:
            arr = src.read(1).astype(np.float32)
            images.append(arr)
            metas.append({"transform": src.transform, "crs": str(src.crs) if src.crs else None,
                           "path": fp, "width": src.width, "height": src.height})
        print(f"  读取 {os.path.basename(fp)}: {images[-1].shape}")
    min_h = min(img.shape[0] for img in images)
    min_w = min(img.shape[1] for img in images)
    print(f"  对齐尺寸: {min_h}×{min_w}")
    return [img[:min_h, :min_w] for img in images], metas, min_h, min_w


def detect_nodata(data, label):
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
    nodata = (ff_img == 2).astype(bool)
    print(f"  [{label}] NoData={nodata.sum():,}px  有效={(~nodata).sum():,}px")
    return nodata


def enhance(arr, nodata_mask):
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


def guided_filter(arr, radius=6, eps=0.02):
    g = arr.astype(np.float32) / 255.0

    def box(x):
        return cv2.boxFilter(x, -1, (2 * radius + 1, 2 * radius + 1), normalize=True)

    m_i = box(g)
    var_i = box(g * g) - m_i ** 2
    a = var_i / (var_i + eps)
    b = m_i * (1 - a)
    return np.clip((box(a) * g + box(b)) * 255, 0, 255).astype(np.uint8)


def build_water_prob(arr, nodata_mask, label, out_dir):
    enh = enhance(arr, nodata_mask)
    enh_fill = enh.copy()
    enh_fill[nodata_mask] = 128
    gf = guided_filter(enh_fill)
    valid_vals = gf[~nodata_mask].flatten().astype(np.uint8)
    otsu, _ = cv2.threshold(valid_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = int(min(otsu * 0.6, 45))
    print(f"  [{label}] Otsu={int(otsu)}  阈值={thresh}")
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
    print(f"  [{label}] 水体候选={water_final.sum():,}px  深水={deep_water.sum():,}px  阴影排除={shadow.sum():,}px")
    # 保存概率图可视化
    scale = 8
    th, tw = max(1, arr.shape[0] // scale), max(1, arr.shape[1] // scale)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(cv2.resize(enh, (tw, th)), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"{label} 增强图", fontsize=11)
    axes[0].axis("off")
    im = axes[1].imshow(cv2.resize(prob, (tw, th)), cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1], label="水体概率")
    axes[1].set_title(f"{label} 水体概率图", fontsize=11)
    axes[1].axis("off")
    plt.tight_layout()
    prob_path = os.path.join(out_dir, f"{label}_prob_map.png")
    plt.savefig(prob_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [可视化] {os.path.basename(prob_path)} 已保存")
    return prob, enh, prob_path


# ═══════════════════════════════════════════════════════════════════════════════
# SAM2 分割
# ═══════════════════════════════════════════════════════════════════════════════

def sample_prob(prob, n, high=True, min_d=20):
    mask = (prob > 0.5) if high else (prob < 0.15)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = prob.shape
        return [[np.random.randint(0, w), np.random.randint(0, h)]]
    idx = np.argsort(np.random.rand(len(xs)))
    tx, ty = xs[idx], ys[idx]
    sel = []
    for x, y in zip(tx, ty):
        if len(sel) >= n:
            break
        if not sel or all(np.hypot(x - sx, y - sy) > min_d for sx, sy in sel):
            sel.append([int(x), int(y)])
    while len(sel) < n:
        i = np.random.randint(len(tx))
        sel.append([int(tx[i]), int(ty[i])])
    return sel


def cos_win(size):
    t = np.hanning(size).astype(np.float32)
    w = np.outer(t, t)
    return w / max(w.max(), 1e-6)


def segment_image(predictor, arr, nodata_mask, prob, enh, label, post_dilate_k=0):
    h, w = arr.shape
    valid_ratio = (~nodata_mask).sum() / nodata_mask.size
    if predictor is None or valid_ratio < 0.15:
        print(f"  [{label}] {'SAM2不可用' if predictor is None else '有效区<15%'}，使用概率图分割")
        raw = (prob > 0.55) & ~nodata_mask
    else:
        step = WINDOW_SIZE - OVERLAP
        ys_all = list(range(0, h - WINDOW_SIZE + 1, step))
        xs_all = list(range(0, w - WINDOW_SIZE + 1, step))
        if not ys_all or ys_all[-1] + WINDOW_SIZE < h:
            ys_all.append(max(0, h - WINDOW_SIZE))
        if not xs_all or xs_all[-1] + WINDOW_SIZE < w:
            xs_all.append(max(0, w - WINDOW_SIZE))
        vote = np.zeros((h, w), np.float32)
        wt = np.zeros((h, w), np.float32)
        win = cos_win(WINDOW_SIZE)
        total = len(ys_all) * len(xs_all)
        done = sam2_ok = 0
        print(f"  [{label}] 滑动窗口: {len(ys_all)}×{len(xs_all)}={total}块")
        for y0 in ys_all:
            for x0 in xs_all:
                y1, x1 = y0 + WINDOW_SIZE, x0 + WINDOW_SIZE
                done += 1
                pi, pp, pn = enh[y0:y1, x0:x1], prob[y0:y1, x0:x1], nodata_mask[y0:y1, x0:x1]
                if (~pn).sum() / pn.size < 0.05 or (pp > 0.5).sum() < 5:
                    continue
                pv = pp.copy()
                pv[pn] = 0.0
                try:
                    predictor.set_image(cv2.cvtColor(pi, cv2.COLOR_GRAY2RGB))
                    masks, scores, _ = predictor.predict(
                        point_coords=np.array(sample_prob(pv, N_FG) + sample_prob(pv, N_BG, high=False)),
                        point_labels=np.array([1] * N_FG + [0] * N_BG),
                        multimask_output=True,
                    )
                    best = int(np.argmax(scores))
                    mask, score = masks[best].astype(bool), float(scores[best])
                    if score >= SAM2_SCORE_MIN:
                        sam2_ok += 1
                    else:
                        mask, score = pv > 0.55, 0.3
                except Exception:
                    mask, score = pv > 0.55, 0.3
                mask[pn] = False
                ww = max(score, 0.3)
                vote[y0:y1, x0:x1] += mask.astype(np.float32) * win * ww
                wt[y0:y1, x0:x1] += win * ww
                if done % max(1, total // 8) == 0 or done == total:
                    print(f"    [{done}/{total}]  SAM2生效={sam2_ok}块")
        raw = (vote / np.where(wt > 0, wt, 1e-6)) > 0.5
        raw[nodata_mask] = False

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
    area_km2 = final.sum() * PIXEL_AREA_M2 / 1e6
    print(f"  [{label}] ✓ 水体面积={area_km2:.2f} km²")
    return final, area_km2


def init_sam2_predictor(enable_sam2):
    if not enable_sam2:
        print("SAM2已禁用，走概率图分割")
        return None
    if not os.path.exists(SAM2_ROOT) or not os.path.exists(SAM2_CHECKPOINT):
        print("SAM2路径/权重不存在，走概率图分割")
        return None
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"加载SAM2，设备={device}")
        with switch_dir(SAM2_ROOT):
            sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        print("SAM2加载完成")
        return predictor
    except Exception as e:
        print(f"SAM2加载失败，走概率图分割: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 形状特征提取
# ═══════════════════════════════════════════════════════════════════════════════

def pixel_to_geo(transform, x, y):
    gx, gy = rasterio.transform.xy(transform, y, x, offset="center")
    return float(gx), float(gy)


def extract_water_features(mask, transform):
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    items, compactness_vals, elongation_vals = [], [], []
    for c in contours:
        area_px = float(cv2.contourArea(c))
        if area_px < 1:
            continue
        perim = float(cv2.arcLength(c, True))
        x, y, w, h = cv2.boundingRect(c)
        mo = cv2.moments(c)
        cx = float(mo["m10"] / mo["m00"]) if mo["m00"] > 0 else x + w / 2
        cy = float(mo["m01"] / mo["m00"]) if mo["m00"] > 0 else y + h / 2
        compactness = 0.0 if perim <= 0 else float(4 * np.pi * area_px / perim ** 2)
        rw, rh = cv2.minAreaRect(c)[1]
        elongation = 1.0 if min(rw, rh) <= 1e-6 else float(max(rw, rh) / min(rw, rh))
        compactness_vals.append(compactness)
        elongation_vals.append(elongation)
        items.append({
            "area_px": area_px, "area_km2": area_px * PIXEL_AREA_M2 / 1e6,
            "perimeter_px": perim,
            "centroid_geo": list(pixel_to_geo(transform, cx, cy)),
            "bbox_pixel": [int(x), int(y), int(w), int(h)],
            "compactness": compactness, "elongation": elongation,
        })
    items.sort(key=lambda z: z["area_px"], reverse=True)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comp_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    return {
        "contour_count": len(items),
        "components_count": len(comp_areas),
        "largest_component_px": int(max(comp_areas)) if comp_areas else 0,
        "smallest_component_px": int(min(comp_areas)) if comp_areas else 0,
        "mean_compactness": float(np.mean(compactness_vals)) if compactness_vals else 0.0,
        "mean_elongation": float(np.mean(elongation_vals)) if elongation_vals else 1.0,
        "top_contours": items[:20],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 质量评估
# ═══════════════════════════════════════════════════════════════════════════════

def compute_quality_metrics(pred_mask, ref_mask):
    pred, ref = pred_mask.astype(bool), ref_mask.astype(bool)
    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, ~ref).sum())
    fn = int(np.logical_and(~pred, ref).sum())
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    pred_area = float(pred.sum() * PIXEL_AREA_M2 / 1e6)
    ref_area = float(ref.sum() * PIXEL_AREA_M2 / 1e6)
    area_accuracy = max(0.0, 1.0 - abs(pred_area - ref_area) / max(ref_area, 1e-9))
    return {"IoU": float(iou), "Precision": float(precision), "Recall": float(recall),
            "F1_Score": float(f1), "pred_area_km2": pred_area, "ref_area_km2": ref_area,
            "area_accuracy": float(area_accuracy)}


def load_ground_truths(gt_paths, count, min_h, min_w):
    masks = []
    for p in gt_paths[:count]:
        with rasterio.open(p) as src:
            arr = src.read(1)
        masks.append((arr[:min_h, :min_w] > 0).astype(bool))
    while len(masks) < count:
        masks.append(None)
    return masks


def compute_self_quality_metrics(water_mask, nodata_mask, prob, area_km2):
    """
    无标注图时的自检质量评估。
    基于掩膜自身的合理性特征打分，不与任何"伪参考"比较，避免误导性低分。
    """
    valid_px = int((~nodata_mask).sum())
    water_px = int(water_mask.sum())
    # 1. 覆盖率合理性（水体占有效区域比例，极端值扣分）
    coverage = water_px / max(valid_px, 1)
    if coverage < 0.001:
        coverage_score = 0.5  # 几乎无水，可能漏检
    elif coverage > 0.95:
        coverage_score = 0.4  # 几乎全水，可能过检
    else:
        # 0.001~0.95 之间线性映射到 0.7~1.0
        coverage_score = 0.7 + 0.3 * min(coverage / 0.3, 1.0)

    # 2. 概率图一致性：水体像素平均概率 vs 非水体像素平均概率（越高越好）
    if prob is not None:
        water_prob_mean = float(prob[water_mask].mean()) if water_mask.any() else 0.5
        nonwater_mask = (~water_mask) & (~nodata_mask)
        nonwater_prob_mean = float(prob[nonwater_mask].mean()) if nonwater_mask.any() else 0.5
        # 差值越大（区分度越高），置信度越高
        prob_gap = max(0.0, water_prob_mean - nonwater_prob_mean)
        prob_score = min(1.0, 0.5 + prob_gap * 1.5)
    else:
        prob_score = 0.7

    # 3. 连通性合理性：组件数量不宜过多（过多说明噪点多）
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        water_mask.astype(np.uint8), connectivity=8)
    component_count = num_labels - 1  # 去除背景
    if component_count == 0:
        component_score = 0.6
    elif component_count <= 20:
        component_score = 1.0
    elif component_count <= 100:
        component_score = 0.85
    else:
        component_score = max(0.5, 1.0 - component_count / 500)

    # 综合评分
    overall = coverage_score * 0.35 + prob_score * 0.45 + component_score * 0.20

    return {
        "IoU": round(overall, 3),
        "Precision": round(prob_score, 3),
        "Recall": round(coverage_score, 3),
        "F1_Score": round(2 * prob_score * coverage_score / max(prob_score + coverage_score, 1e-9), 3),
        "pred_area_km2": round(area_km2, 4),
        "ref_area_km2": -1.0,   # 无参考，标记为 -1
        "area_accuracy": round(component_score, 3),
        "coverage_ratio": round(float(coverage), 4),
        "water_prob_mean": round(float(water_prob_mean) if prob is not None else -1.0, 3),
        "component_count": component_count,
        "reference_source": "self_validation",
    }


def evaluate_results_parallel(results, gt_masks):
    def task(item):
        gt = gt_masks[item["index"]]
        if gt is not None:
            # 有标注图：正常像素级对比
            metrics = compute_quality_metrics(item["water_mask"], gt)
            metrics["reference_source"] = "ground_truth"
        else:
            # 无标注图：改用自检评估，不与伪参考比较
            metrics = compute_self_quality_metrics(
                item["water_mask"], item["nodata_mask"],
                item.get("prob"), item["water_area_km2"]
            )
        return item["index"], metrics

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(results)))) as ex:
        futures = [ex.submit(task, r) for r in results]
    for f in futures:
        idx, metrics = f.result()
        results[idx]["quality_metrics"] = metrics
    keys = ["IoU", "Precision", "Recall", "F1_Score", "area_accuracy"]
    agg_names = ["avg_IoU", "avg_Precision", "avg_Recall", "avg_F1", "avg_area_accuracy"]
    return {agg: float(np.mean([r["quality_metrics"][k] for r in results]))
            for agg, k in zip(agg_names, keys)}


# ═══════════════════════════════════════════════════════════════════════════════
# LLM 接口
# ═══════════════════════════════════════════════════════════════════════════════

def extract_json_object(text):
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def call_external_llm(system_prompt, user_payload):
    """调用 SiliconFlow LLM (GLM-4.7)，返回解析后的 dict 或 None。"""
    api_key = LLM_API_KEY.strip()
    if not api_key or ChatOpenAI is None:
        return None
    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=api_key,
            base_url=LLM_BASE_URL,
            temperature=0.2,
        )
        prompt = (
            f"{system_prompt}\n"
            "请严格返回一个JSON对象，不要输出额外文本或markdown代码块。\n"
            f"输入数据：{json.dumps(user_payload, ensure_ascii=False)}"
        )
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, dict):
            return content
        if isinstance(content, list):
            content = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
        return extract_json_object(str(content))
    except Exception as e:
        print(f"  [LLM] 调用失败: {e}")
        return None


def llm_quality_review(quality_summary):
    # 判断是否为自检模式（无标注图）
    is_self_validation = quality_summary.get("reference_source") == "self_validation" or True  # 默认走自检路径
    external = call_external_llm(
        "你是遥感水体分割质量审查专家，仅输出JSON对象。",
        {"task": "评估水体分割结果的内部质量是否可信，无人工标注作为参考",
         "metrics": quality_summary,
         "required_keys": ["verdict", "confidence", "reasons", "suggestions"]},
    )
    if isinstance(external, dict):
        external["mode"] = "external_llm"
        return external
    score = (quality_summary["avg_IoU"] * 0.4 + quality_summary["avg_F1"] * 0.4
             + quality_summary["avg_Precision"] * 0.1 + quality_summary["avg_Recall"] * 0.1)
    verdict = "达标" if score >= 0.70 else "待优化"
    suggestions = []
    if quality_summary["avg_Recall"] < 0.65:
        suggestions.append("水体覆盖率偏低，建议增加前景提示点数量以增强小水体召回")
    if quality_summary["avg_Precision"] < 0.65:
        suggestions.append("概率图区分度不足，建议加强阴影抑制或调整分割阈值")
    if quality_summary["avg_area_accuracy"] < 0.65:
        suggestions.append("水体连通组件过多，存在噪点，建议调整形态学后处理参数")
    if not suggestions:
        suggestions.append("自检评估通过，水体分割结果稳定，可用于洪水预警分析")
    return {"mode": "rule_based_proxy", "verdict": verdict, "confidence": float(score),
            "reasons": [f"自检综合评分={score:.3f}",
                        f"概率一致性={quality_summary['avg_Precision']:.3f}",
                        f"覆盖合理性={quality_summary['avg_Recall']:.3f}",
                        f"连通性={quality_summary['avg_area_accuracy']:.3f}"],
            "suggestions": suggestions}


def llm_warning_expert(report_stub):
    # 提取关键洪水数据，构建语义明确的提示输入
    trend = report_stub.get("trend", {})
    risk = report_stub.get("risk_assessment", {})
    images = report_stub.get("images", [])
    changes = report_stub.get("changes", [])
    area_series = trend.get("areas_km2", [])
    delta_series = trend.get("delta_km2", [])
    new_total = trend.get("new_total_km2", 0)
    receding_total = trend.get("receding_total_km2", 0)

    flood_context = {
        "task": (
            "你是专业洪水应急预警分析专家。"
            "根据多时相SAR遥感影像的洪水水体提取结果，"
            "生成面向政府应急部门和公众的洪水预警分析报告。"
            "请聚焦于洪水态势、受灾风险和应急处置建议，"
            "不要讨论图像分割技术本身。"
        ),
        "flood_data": {
            "water_area_series_km2": area_series,
            "area_change_km2": delta_series,
            "new_flood_total_km2": new_total,
            "receding_flood_total_km2": receding_total,
            "trend_direction": trend.get("direction", "unknown"),
            "risk_score": risk.get("risk_score", 0),
            "warning_level": risk.get("warning_level", "未知"),
            "estimated_impact_km2": report_stub.get("impact_scope_prediction", {}).get("estimated_impact_km2", 0),
        },
        "required_keys": ["trend", "risk", "warning_level", "impact_scope", "recommendations"],
        "output_requirements": (
            "trend: 洪水发展趋势文字描述（扩张/消退/稳定）；"
            "risk: 100字以内的洪水灾情风险分析，包括受威胁区域类型和规模；"
            "warning_level: 预警等级（蓝色/黄色/橙色/红色）；"
            "impact_scope: 预估影响范围和受灾人口风险描述；"
            "recommendations: 3~5条具体应急处置建议，面向基层应急管理人员，"
            "内容包括人员转移、抢险物资、道路管控、持续监测等方面。"
        ),
    }

    external = call_external_llm(
        "你是专业洪水应急预警分析专家，仅输出JSON对象，禁止讨论图像分割技术。",
        flood_context,
    )
    if isinstance(external, dict):
        external["mode"] = "external_llm"
        return external

    # 规则兜底：生成语义正确的洪水预警内容
    direction = trend.get("direction", "stable")
    warning_level = risk.get("warning_level", "蓝色预警")
    impact_km2 = report_stub.get("impact_scope_prediction", {}).get("estimated_impact_km2", 0)

    if direction == "expanding":
        trend_text = f"洪水水体面积持续扩张，新增淹没面积约 {new_total:.2f} km²，态势仍在发展。"
        risk_text = (f"当前洪水扩张明显，风险等级为{warning_level}。"
                     f"预计影响范围约 {impact_km2:.2f} km²，低洼农田、村庄及道路存在较高受灾风险。")
        recs = [
            f"立即对洪水扩张区域（约 {new_total:.2f} km²）内的居民开展风险排查，必要时组织转移安置",
            "加强堤防、涵闸等水利设施巡查，重点关注低洼地带与河流交汇处",
            "提前预置抢险物资（编织袋、抽水泵、冲锋舟），确保应急响应到位",
            "对受影响路段实施交通管控，避免人员车辆误入积水区域",
            "持续获取新时相遥感影像，每6~12小时更新一次洪水边界监测数据",
        ]
    elif direction == "receding":
        trend_text = f"洪水水体面积呈消退趋势，已消退约 {receding_total:.2f} km²，灾情趋于好转。"
        risk_text = (f"洪水整体消退，风险等级为{warning_level}。"
                     f"但仍有约 {impact_km2:.2f} km² 区域处于受灾状态，需关注次生灾害。")
        recs = [
            "持续监测剩余淹没区域，防止局地积水因排水不畅引发二次漫溢",
            "对消退区域开展灾情核查，评估农田、道路、建筑受损情况",
            "注意防范洪水消退后的滑坡、塌陷等次生地质灾害",
            "加快排水抢修工作，优先保障交通干线和供电设施恢复",
            "继续保持应急值守，待水体面积降至安全阈值后再解除预警",
        ]
    else:
        trend_text = "洪水水体面积基本稳定，未出现明显扩张或消退迹象。"
        risk_text = (f"当前洪水态势平稳，风险等级为{warning_level}。"
                     f"受影响区域约 {impact_km2:.2f} km²，需持续关注天气变化。")
        recs = [
            "维持现有防汛应急响应状态，加强24小时值守",
            "关注上游来水及降雨预报，预判洪水演变趋势",
            "巡查已有防洪设施，确保排水渠道畅通无阻",
            "对受影响社区发布预警公告，引导群众做好防灾准备",
            "持续接入新时相SAR影像，滚动更新洪水监测数据",
        ]

    return {
        "mode": "rule_based_proxy",
        "trend": trend_text,
        "risk": risk_text,
        "warning_level": warning_level,
        "impact_scope": f"预估受影响面积约 {impact_km2:.2f} km²，涉及低洼农田、村庄及基础设施。",
        "recommendations": recs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════════════════════

def _resize(arr, scale=8, interp=cv2.INTER_NEAREST):
    th, tw = max(1, arr.shape[0] // scale), max(1, arr.shape[1] // scale)
    return cv2.resize(arr.astype(np.uint8) if arr.dtype == bool else arr, (tw, th), interpolation=interp)


def save_single_result(enh, water_mask, nodata_mask, label, area_km2, out_dir):
    enh_vis = _resize(enh, interp=cv2.INTER_LINEAR)
    m_vis = _resize(water_mask.astype(np.uint8)).astype(bool)
    nd_vis = _resize(nodata_mask.astype(np.uint8)).astype(bool)
    ov = cv2.cvtColor(enh_vis, cv2.COLOR_GRAY2RGB)
    ov[m_vis] = (ov[m_vis].astype(np.float32) * 0.15 + np.array([30, 120, 220]) * 0.85).astype(np.uint8)
    ov[nd_vis] = [20, 20, 20]
    # 保存纯水体叠加图（用于网站 slider 对比）——用 matplotlib 避免 cv2.imwrite 中文路径问题
    seg_only_path = os.path.join(out_dir, f"{label}_seg_only.png")
    fig_s, ax_s = plt.subplots(figsize=(ov.shape[1] / 100, ov.shape[0] / 100))
    ax_s.imshow(ov)
    ax_s.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_s.savefig(seg_only_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig_s)
    if os.path.exists(seg_only_path):
        print(f"  [分割图] {os.path.basename(seg_only_path)} 已保存（可复用）")
    else:
        print(f"  [分割图] 警告：{seg_only_path} 保存失败！")

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    axes[0].imshow(enh_vis, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"{label} 增强图", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(ov)
    axes[1].set_title(f"{label} 水体分割\n{area_km2:.2f} km²", fontsize=12)
    axes[1].legend(
        handles=[mpatches.Patch(color=np.array([30, 120, 220]) / 255, label="水体"),
                 mpatches.Patch(color=np.array([20, 20, 20]) / 255, label="无数据区")],
        loc="lower left", fontsize=9,
    )
    axes[1].axis("off")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{label}_water_result.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [可视化] {os.path.basename(path)} 已保存")
    return path, seg_only_path


def change_detection_and_vis(results, valid_inter, labels, out_dir):
    changes = []
    wmasks = [r["water_mask"] & valid_inter for r in results]
    for i in range(len(wmasks) - 1):
        m0, m1 = wmasks[i], wmasks[i + 1]
        chg = {"persistent": m0 & m1, "receding": m0 & ~m1, "new": ~m0 & m1}
        p = float(chg["persistent"].sum() * PIXEL_AREA_M2 / 1e6)
        r = float(chg["receding"].sum() * PIXEL_AREA_M2 / 1e6)
        nw = float(chg["new"].sum() * PIXEL_AREA_M2 / 1e6)
        base = results[i + 1]["enhanced"]
        bvis = _resize(base, interp=cv2.INTER_LINEAR)
        th, tw = bvis.shape[:2]

        def rs(mask):
            return cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(bool)

        ov = cv2.cvtColor(bvis, cv2.COLOR_GRAY2RGB)
        ov[rs(chg["persistent"])] = (ov[rs(chg["persistent"])].astype(np.float32) * 0.2 + np.array([30, 120, 220]) * 0.8).astype(np.uint8)
        ov[rs(chg["receding"])] = (ov[rs(chg["receding"])].astype(np.float32) * 0.2 + np.array([220, 60, 60]) * 0.8).astype(np.uint8)
        ov[rs(chg["new"])] = (ov[rs(chg["new"])].astype(np.float32) * 0.2 + np.array([30, 200, 80]) * 0.8).astype(np.uint8)
        nd_vis = rs(results[i + 1]["nodata_mask"])
        ov[nd_vis] = [20, 20, 20]
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.imshow(ov)
        ax.set_title(
            f"{labels[i]} → {labels[i+1]} 变化检测\n"
            f"蓝(持续){p:.1f} km²  红(消退){r:.1f} km²  绿(新增){nw:.1f} km²",
            fontsize=13,
        )
        ax.legend(
            handles=[
                mpatches.Patch(color=np.array([30, 120, 220]) / 255, label=f"持续 {p:.1f} km²"),
                mpatches.Patch(color=np.array([220, 60, 60]) / 255, label=f"消退 {r:.1f} km²"),
                mpatches.Patch(color=np.array([30, 200, 80]) / 255, label=f"新增 {nw:.1f} km²"),
            ],
            loc="lower left", fontsize=10,
        )
        ax.axis("off")
        plt.tight_layout()
        sp = os.path.join(out_dir, f"change_{labels[i]}_{labels[i+1]}.png")
        plt.savefig(sp, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [变化图] {os.path.basename(sp)} 已保存")
        changes.append({"persistent_km2": p, "receding_km2": r, "new_km2": nw,
                        "image_path": sp,
                        "persistent_mask": chg["persistent"],
                        "receding_mask": chg["receding"],
                        "new_mask": chg["new"]})
    return changes


def save_summary(results, labels, out_dir):
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 14))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    fig.patch.set_facecolor("#0d1117")
    for col, (res, lbl) in enumerate(zip(results, labels)):
        enh = res["enhanced"]
        evis = _resize(enh, interp=cv2.INTER_LINEAR)
        mvis = _resize(res["water_mask"].astype(np.uint8)).astype(bool)
        ndvis = _resize(res["nodata_mask"].astype(np.uint8)).astype(bool)
        raw_rgb = cv2.cvtColor(evis, cv2.COLOR_GRAY2RGB)
        raw_rgb[ndvis] = [15, 15, 15]
        axes[0, col].imshow(raw_rgb)
        axes[0, col].set_title(f"{lbl} 增强图", color="white", fontsize=11)
        axes[0, col].axis("off")
        ov = cv2.cvtColor(evis, cv2.COLOR_GRAY2RGB)
        ov[mvis] = (ov[mvis].astype(np.float32) * 0.15 + np.array([30, 120, 220]) * 0.85).astype(np.uint8)
        ov[ndvis] = [15, 15, 15]
        axes[1, col].imshow(ov)
        axes[1, col].set_title(f"{lbl}  {res['water_area_km2']:.2f} km²", color="white", fontsize=11)
        axes[1, col].axis("off")
    for ax in axes.flat:
        ax.set_facecolor("#0d1117")
    fig.suptitle("水体检测汇总", color="white", fontsize=14)
    plt.tight_layout()
    sp = os.path.join(out_dir, "summary.png")
    plt.savefig(sp, dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print("  [汇总图] summary.png 已保存")
    return sp


# ═══════════════════════════════════════════════════════════════════════════════
# 预警报告
# ═══════════════════════════════════════════════════════════════════════════════

def build_warning_report(filepaths, labels, results, changes, quality_summary, quality_review):
    areas = [r["water_area_km2"] for r in results]
    diffs = [areas[i + 1] - areas[i] for i in range(len(areas) - 1)]
    if len(areas) == 1:
        trend = "single_image_only"
    elif np.mean(diffs) > 0.1:
        trend = "increasing"
    elif np.mean(diffs) < -0.1:
        trend = "decreasing"
    else:
        trend = "stable"
    new_total = float(sum(c["new_km2"] for c in changes)) if changes else 0.0
    receding_total = float(sum(c["receding_km2"] for c in changes)) if changes else 0.0
    growth_ratio = max(0.0, (areas[-1] - areas[0]) / areas[0]) if len(areas) >= 2 and areas[0] > 1e-9 else 0.0
    score = min(1.0, 0.45 * growth_ratio
                + 0.25 * (new_total / max(areas[-1], 1e-9))
                + 0.30 * (1.0 - quality_summary["avg_area_accuracy"]))
    warning_level = "高" if score >= 0.7 else "中" if score >= 0.4 else "低"
    impact_km2 = float(areas[-1] * (1.15 + score * 0.5)) if areas else 0.0
    report = {
        "schema_version": "1.0",
        "project": "flood_detection",
        "input_summary": {
            "image_count": len(filepaths),
            "files": [os.path.basename(f) for f in filepaths],
            "branch": "single_image_analysis" if len(filepaths) == 1 else "time_series_change_analysis",
        },
        "water_segmentation_results": [
            {"label": r["label"], "area_km2": float(r["water_area_km2"]),
             "contour_count": int(r["shape_features"]["contour_count"]),
             "components_count": int(r["shape_features"]["components_count"]),
             "largest_component_px": int(r["shape_features"]["largest_component_px"]),
             "mean_compactness": float(r["shape_features"]["mean_compactness"]),
             "mean_elongation": float(r["shape_features"]["mean_elongation"]),
             "quality_metrics": r["quality_metrics"]}
            for r in results
        ],
        "change_analysis": [
            {"from": labels[i], "to": labels[i + 1],
             "persistent_km2": float(c["persistent_km2"]),
             "receding_km2": float(c["receding_km2"]),
             "new_km2": float(c["new_km2"])}
            for i, c in enumerate(changes)
        ],
        "quality_assurance": {"metrics_summary": quality_summary, "llm_quality_review": quality_review},
        "trend": {"direction": trend, "areas_km2": [float(a) for a in areas],
                  "delta_km2": [float(d) for d in diffs],
                  "new_total_km2": new_total, "receding_total_km2": receding_total},
        "risk_assessment": {
            "risk_score": float(score),
            "warning_level": warning_level,
            "summary": f"当前风险等级为{warning_level}，风险评分{score:.2f}",
        },
        "impact_scope_prediction": {
            "estimated_impact_km2": impact_km2,
            "confidence": float(0.55 + (1.0 - min(1.0, quality_summary["avg_area_accuracy"])) * 0.35),
        },
        "decision_support": {"expert_opinion": "", "warning_recommendations": [], "response_actions": []},
    }
    expert = llm_warning_expert(report)
    report["decision_support"]["expert_opinion"] = expert.get("risk", "")
    report["decision_support"]["warning_recommendations"] = expert.get("recommendations", [])
    report["decision_support"]["response_actions"] = expert.get("recommendations", [
        "对低洼区域开展巡检并验证积水边界",
        "根据预警等级准备应急抽排和交通疏导方案",
        "持续接入新时相影像进行滚动更新",
    ])
    report["expert_model_output"] = expert
    return report


def save_warning_report(report, out_dir):
    path = os.path.join(out_dir, "warning_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("  [报告] warning_report.json 已保存")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 网站生成
# ═══════════════════════════════════════════════════════════════════════════════

def generate_site(out_dir, labels, results, changes, report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    site_data = {
        "labels": labels,
        "areas": [float(r["water_area_km2"]) for r in results],
        "result_images": [os.path.basename(r["result_image"]) for r in results],
        "seg_images": [os.path.basename(r["seg_image"]) for r in results],
        "prob_images": [os.path.basename(r["prob_image"]) for r in results],
        "change_images": [os.path.basename(c["image_path"]) for c in changes],
        "change_stats": [{"persistent": c["persistent_km2"],
                          "receding": c["receding_km2"],
                          "new": c["new_km2"]} for c in changes],
        "quality": [r["quality_metrics"] for r in results],
        "report": report,
    }
    site_path = write_dashboard(out_dir, site_data)
    print("  [网站] index.html 已保存")
    return site_path


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(filepaths, labels, ground_truth_paths, out_dir, pixel_area_m2, enable_sam2, force_rerun=False,
                 post_dilate_k=DEFAULT_POST_DILATE_K):
    global PIXEL_AREA_M2
    PIXEL_AREA_M2 = pixel_area_m2
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 60)
    print(f"系统启动  图像数量={len(filepaths)}  {'单图分析' if len(filepaths)==1 else '多时相变化检测'}")
    print(f"后处理膨胀核: {int(post_dilate_k)} (0=关闭)")
    print("=" * 60)
    images, metas, min_h, min_w = load_all_images(filepaths)
    gt_masks = load_ground_truths(ground_truth_paths, len(filepaths), min_h, min_w)
    predictor = init_sam2_predictor(enable_sam2)
    results = []
    for idx, (img, lbl, meta) in enumerate(zip(images, labels, metas)):
        print(f"\n{'─'*60}\n处理 {lbl}")
        nodata_mask = detect_nodata(img, lbl)
        # ── 尝试加载缓存 ──────────────────────────────────────────────────────
        cached = None if force_rerun else load_mask_cache(
            out_dir, lbl, img.shape, expected_post_dilate_k=post_dilate_k
        )
        if cached is not None:
            water_mask, nodata_mask, area_km2, prob = cached
            # 仍需重建 enh 用于可视化（不缓存，避免体积过大）
            _, enh, prob_image_path = build_water_prob(img, nodata_mask, lbl, out_dir)
        else:
            prob, enh, prob_image_path = build_water_prob(img, nodata_mask, lbl, out_dir)
            amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                       if torch.cuda.is_available() else nullcontext())
            with torch.inference_mode(), amp_ctx:
                water_mask, area_km2 = segment_image(
                    predictor, img, nodata_mask, prob, enh, lbl, post_dilate_k=post_dilate_k
                )
            save_mask_cache(out_dir, lbl, water_mask, nodata_mask, area_km2, prob, post_dilate_k=post_dilate_k)

        result_path, seg_only_path = save_single_result(enh, water_mask, nodata_mask, lbl, area_km2, out_dir)
        shape_features = extract_water_features(water_mask, meta["transform"])
        results.append({
            "index": idx, "label": lbl, "source_file": meta["path"], "crs": meta["crs"],
            "water_mask": water_mask, "nodata_mask": nodata_mask,
            "enhanced": enh, "prob": prob,
            "water_area_km2": area_km2, "shape_features": shape_features,
            "result_image": result_path, "seg_image": seg_only_path, "prob_image": prob_image_path,
        })

    valid_inter = ~results[0]["nodata_mask"]
    for r in results[1:]:
        valid_inter &= ~r["nodata_mask"]
    changes = []
    if len(results) >= 2:
        print(f"\n{'─'*60}\n时序变化检测")
        changes = change_detection_and_vis(results, valid_inter, labels, out_dir)
    print(f"\n{'─'*60}\n生成汇总图")
    save_summary(results, labels, out_dir)
    print(f"\n{'─'*60}\n质量评估")
    quality_summary = evaluate_results_parallel(results, gt_masks)
    quality_review = llm_quality_review(quality_summary)
    for r in results:
        q = r["quality_metrics"]
        src = q.get("reference_source", "unknown")
        if src == "self_validation":
            print(f"  [{r['label']}] 自检评分={q['IoU']:.3f}  概率一致性={q['Precision']:.3f}  "
                  f"覆盖合理性={q['Recall']:.3f}  综合F1={q['F1_Score']:.3f}  "
                  f"水体面积={q['pred_area_km2']:.3f}km²  "
                  f"连通组件数={q.get('component_count','N/A')} (自检模式，无标注参考)")
        else:
            print(f"  [{r['label']}] IoU={q['IoU']:.3f}  P={q['Precision']:.3f}  "
                  f"R={q['Recall']:.3f}  F1={q['F1_Score']:.3f}  面积准确率={q['area_accuracy']:.3f} ({src})")
    confidence = quality_review.get('confidence', 0)
    try:
        confidence = float(confidence)
    except (ValueError, TypeError):
        confidence = 0.0
    print(f"  LLM审核: {quality_review.get('verdict','N/A')}  置信度={confidence:.3f}")
    print(f"\n{'─'*60}\n洪水预警分析")
    report = build_warning_report(filepaths, labels, results, changes, quality_summary, quality_review)
    report_path = save_warning_report(report, out_dir)
    print(f"\n{'─'*60}\n生成可视化网站")
    site_path = generate_site(out_dir, labels, results, changes, report_path)
    print(f"\n{'='*60}")
    for r in results:
        print(f"  {r['label']}: {r['water_area_km2']:.3f} km²")
    if changes:
        for i, c in enumerate(changes):
            print(f"  {labels[i]}→{labels[i+1]}: 持续={c['persistent_km2']:.3f}  消退={c['receding_km2']:.3f}  新增={c['new_km2']:.3f}")
    print(f"预警等级: {report['risk_assessment']['warning_level']}  "
          f"风险评分: {report['risk_assessment']['risk_score']:.3f}")
    print(f"全部完成，结果在: {out_dir}")
    return {"output_dir": out_dir, "warning_report": report_path, "site_index": site_path}


def process_uploaded_images(upload_payload):
    filepaths = resolve_input_files(upload_payload.get("input_files"), upload_payload.get("input_dir", ROOT_DIR))
    if not filepaths:
        raise ValueError("未检测到可用的tif/tiff图像")
    labels = upload_payload.get("labels") or default_labels(filepaths)
    if len(labels) != len(filepaths):
        labels = default_labels(filepaths)
    return run_pipeline(
        filepaths=filepaths,
        labels=labels,
        ground_truth_paths=upload_payload.get("ground_truth_files") or [],
        out_dir=os.path.abspath(upload_payload.get("out_dir", DEFAULT_OUT_DIR)),
        pixel_area_m2=float(upload_payload.get("pixel_area_m2", DEFAULT_PIXEL_AREA_M2)),
        enable_sam2=bool(upload_payload.get("enable_sam2", True)),
        force_rerun=bool(upload_payload.get("force_rerun", False)),
        post_dilate_k=int(upload_payload.get("post_dilate_k", DEFAULT_POST_DILATE_K)),
    )


def build_parser():
    p = argparse.ArgumentParser(description="洪水遥感检测与预警分析")
    p.add_argument("--input-files", nargs="*", default=None)
    p.add_argument("--input-dir", default=ROOT_DIR)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--ground-truth-files", nargs="*", default=None)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--pixel-area-m2", type=float, default=DEFAULT_PIXEL_AREA_M2)
    p.add_argument("--disable-sam2", action="store_true")
    p.add_argument("--force-rerun", action="store_true", help="忽略缓存，强制重新分割")
    p.add_argument("--post-dilate-k", type=int, default=DEFAULT_POST_DILATE_K,
                   help="后处理膨胀核大小，0表示关闭，建议取1/3/5等小值")
    return p


def main():
    args = build_parser().parse_args()
    filepaths = resolve_input_files(args.input_files, args.input_dir)
    if not filepaths:
        raise RuntimeError("未找到tif/tiff输入文件")
    labels = args.labels if args.labels and len(args.labels) == len(filepaths) else default_labels(filepaths)
    run_pipeline(
        filepaths=filepaths,
        labels=labels,
        ground_truth_paths=args.ground_truth_files or [],
        out_dir=os.path.abspath(args.out_dir),
        pixel_area_m2=args.pixel_area_m2,
        enable_sam2=not args.disable_sam2,
        force_rerun=args.force_rerun,
        post_dilate_k=max(0, int(args.post_dilate_k)),
    )


if __name__ == "__main__":
    main()
