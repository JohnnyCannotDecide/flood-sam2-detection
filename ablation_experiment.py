"""
ablation_experiment.py
功能：无标注条件下的可执行消融实验（固定单景影像，如 d28.tif）。
改进点：
1) 结构化提示点采样（替代纯随机）
2) SAM2 高置信门控（低分回退概率图）
3) 滑窗边缘抑制（减弱拼接伪影）
4) 差异图与无标注自检指标输出
"""
import argparse
import csv
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import rasterio
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 独立常量，避免运行时强依赖 flood_detection 导致整脚本中断
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_ROOT = os.path.join(ROOT_DIR, "sam2")
SAM2_CHECKPOINT = os.path.join(SAM2_ROOT, "checkpoints", "sam2.1_hiera_small.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
if SAM2_ROOT not in sys.path:
    sys.path.insert(0, SAM2_ROOT)

DEFAULT_PIXEL_AREA_M2 = 25.0
WINDOW_SIZE = 512
OVERLAP = 128
N_FG = 6
N_BG = 4
MIN_WATER_AREA = 300
MORPH_CLOSE_K = 9
DEFAULT_POST_DILATE_K = 7
ENABLE_CRF_DEFAULT = True
CRF_ITER_DEFAULT = 5


def configure_matplotlib_fonts() -> None:
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


configure_matplotlib_fonts()


def should_skip_write(path: str, reuse_existing: bool) -> bool:
    if reuse_existing and os.path.exists(path):
        print(f"  [viz] 复用已有文件: {path}")
        return True
    return False


def copy_to_archive(src: str, archive_path: str, reuse_existing: bool) -> None:
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
    if should_skip_write(archive_path, reuse_existing):
        return
    shutil.copy2(src, archive_path)


def detect_nodata(data: np.ndarray, label: str) -> np.ndarray:
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

    def box(x):
        return cv2.boxFilter(x, -1, (2 * radius + 1, 2 * radius + 1), normalize=True)

    m_i = box(g)
    var_i = box(g * g) - m_i ** 2
    a = var_i / (var_i + eps)
    b = m_i * (1 - a)
    return np.clip((box(a) * g + box(b)) * 255, 0, 255).astype(np.uint8)


def sample_prob(prob: np.ndarray, n: int, high: bool = True, min_d: int = 20) -> List[List[int]]:
    ys, xs = np.where(prob > 0.5 if high else prob < 0.15)
    if len(xs) == 0:
        ys, xs = np.where(prob > 0 if high else prob <= 1)
    idx = np.argsort(np.random.rand(len(xs)))
    sel: List[List[int]] = []
    for i in idx:
        x, y = int(xs[i]), int(ys[i])
        if not sel or all(np.hypot(x - sx, y - sy) > min_d for sx, sy in sel):
            sel.append([x, y])
        if len(sel) >= n:
            break
    while len(sel) < n:
        sel.append([np.random.randint(0, prob.shape[1]), np.random.randint(0, prob.shape[0])])
    return sel


def cos_win(size: int) -> np.ndarray:
    h = np.hanning(size)
    w = np.outer(h, h).astype(np.float32)
    return w / max(float(w.max()), 1e-6)


def init_sam2_predictor(enable_sam2: bool):
    if not enable_sam2:
        return None
    if not os.path.exists(SAM2_ROOT) or not os.path.exists(SAM2_CHECKPOINT):
        print("SAM2路径/权重不存在，自动回退概率图")
        return None
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"加载SAM2，设备={device}")
        cwd = os.getcwd()
        try:
            os.chdir(SAM2_ROOT)
            model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
        finally:
            os.chdir(cwd)
        return SAM2ImagePredictor(model)
    except Exception as e:
        print(f"SAM2加载失败，自动回退概率图: {e}")
        return None


@dataclass
class ExperimentConfig:
    name: str
    use_sam2: bool
    sampler_mode: str
    sam2_conf_min: float
    edge_trim: int
    post_dilate_k: int


def normalize_to_u8(arr: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.uint8)
    valid = arr[~nodata_mask]
    if valid.size == 0:
        return out
    p2, p98 = np.percentile(valid, [2, 98])
    scale = np.clip((arr.astype(np.float32) - float(p2)) / max(float(p98 - p2), 1e-6), 0, 1)
    out = (scale * 255).astype(np.uint8)
    out[nodata_mask] = 0
    return out


def build_prob_components(
    arr: np.ndarray,
    nodata_mask: np.ndarray,
    gf_radius: int = 6,
    gf_eps: float = 0.02,
    deep_pct: float = 6.0,
    deep_fixed_thr: Optional[float] = 8.0,
    shadow_range_thr: int = 15,
) -> Dict[str, np.ndarray]:
    enh = enhance(arr, nodata_mask)
    enh_fill = enh.copy()
    enh_fill[nodata_mask] = 128
    gf = guided_filter(enh_fill, radius=gf_radius, eps=gf_eps)
    valid_vals = gf[~nodata_mask].astype(np.uint8)
    otsu, _ = cv2.threshold(valid_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(min(float(otsu) * 0.6, 45))
    water_cand = (gf < thr) & ~nodata_mask & (arr > 0)

    valid_arr = arr[(arr > 0) & (~nodata_mask)]
    if deep_fixed_thr is None:
        deep_thr = float(np.percentile(valid_arr, deep_pct)) if valid_arr.size else 8.0
    else:
        deep_thr = float(deep_fixed_thr)
    deep_water = (arr > 0) & (arr <= deep_thr) & ~nodata_mask

    local_range = cv2.dilate(gf, np.ones((11, 11), np.uint8)) - cv2.erode(gf, np.ones((11, 11), np.uint8))
    shadow = water_cand & (local_range < shadow_range_thr) & ~deep_water
    no_shadow = water_cand & ~shadow
    water_final = no_shadow | deep_water

    prob = np.zeros(arr.shape, np.float32)
    prob[water_final] = 1.0
    prob[shadow] = 0.2
    prob[deep_water] = 1.0
    prob[nodata_mask] = 0.0
    prob = cv2.GaussianBlur(prob, (7, 7), 2)
    prob[nodata_mask] = 0.0
    fg_mask = (prob >= 0.6) & ~nodata_mask
    bg_mask = (prob <= 0.4) & ~nodata_mask
    uncertain_mask = (prob > 0.4) & (prob < 0.6) & ~nodata_mask
    edge_map = cv2.Canny(gf, 50, 130) > 0
    edge_map[nodata_mask] = False

    return {
        "enh": enh,
        "gf": gf,
        "prob": prob,
        "fg_mask": fg_mask,
        "bg_mask": bg_mask,
        "uncertain_mask": uncertain_mask,
        "edge_map": edge_map,
        "water_cand": water_cand,
        "shadow": shadow,
        "deep_water": deep_water,
        "water_final": water_final,
    }


def structured_prompt_points(
    prob_window: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    edge_map: np.ndarray,
    n_fg: int,
    n_bg: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    h, w = prob_window.shape
    fg_u8 = fg_mask.astype(np.uint8)

    # 前景点：优先连通域中心与局部峰值
    fg_points: List[List[int]] = []
    num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
    order = sorted(range(1, num_labels), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
    for i in order:
        ys, xs = np.where(labels_map == i)
        if len(xs) == 0:
            continue
        px = int(np.median(xs))
        py = int(np.median(ys))
        fg_points.append([px, py])
        if len(fg_points) >= n_fg:
            break
    if len(fg_points) < n_fg:
        ys, xs = np.where(prob_window > 0.55)
        if len(xs) > 0:
            idx = np.argsort(prob_window[ys, xs])[::-1]
            for k in idx:
                fg_points.append([int(xs[k]), int(ys[k])])
                if len(fg_points) >= n_fg:
                    break
    if not fg_points:
        fg_points = [[w // 2, h // 2]]
    fg_points = fg_points[:n_fg]

    # 背景点：低概率区 + 外环
    bg_points: List[List[int]] = []
    edge_band = cv2.dilate(edge_map.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0
    ring = cv2.dilate(fg_u8, np.ones((15, 15), np.uint8), iterations=1) & (~fg_u8.astype(bool))
    ring = ring.astype(np.uint8)
    ys1, xs1 = np.where(((ring > 0) | edge_band) & (prob_window < 0.4))
    for x, y in zip(xs1[: n_bg // 2 + 1], ys1[: n_bg // 2 + 1]):
        bg_points.append([int(x), int(y)])
        if len(bg_points) >= n_bg:
            break
    if len(bg_points) < n_bg:
        ys, xs = np.where(bg_mask > 0)
        if len(xs) > 0:
            idx = np.random.permutation(len(xs))
            for i in idx:
                bg_points.append([int(xs[i]), int(ys[i])])
                if len(bg_points) >= n_bg:
                    break
    if not bg_points:
        bg_points = [[0, 0]]
    bg_points = bg_points[:n_bg]
    return fg_points, bg_points


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


def dense_crf_refine(image_u8: np.ndarray, bin_mask: np.ndarray, iterations: int = 5) -> np.ndarray:
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax

        h, w = bin_mask.shape
        img = cv2.cvtColor(image_u8.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        fg = np.clip(bin_mask.astype(np.float32), 1e-3, 1 - 1e-3)
        probs = np.stack([1.0 - fg, fg], axis=0)
        unary = unary_from_softmax(probs)
        unary = np.ascontiguousarray(unary)
        d = dcrf.DenseCRF2D(w, h, 2)
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=30, srgb=10, rgbim=img, compat=5)
        q = d.inference(max(1, int(iterations)))
        out = np.array(q)[1].reshape(h, w)
        return out > 0.5
    except Exception:
        return bin_mask


def segment_variant(
    predictor,
    enh: np.ndarray,
    prob: np.ndarray,
    nodata_mask: np.ndarray,
    guidance: Dict[str, np.ndarray],
    cfg: ExperimentConfig,
    strict_cv: bool = True,
    enable_crf: bool = ENABLE_CRF_DEFAULT,
    crf_iter: int = CRF_ITER_DEFAULT,
) -> Dict[str, object]:
    h, w = enh.shape
    ys, xs = get_sliding_positions(h, w)
    total = len(ys) * len(xs)
    progress_step = max(1, total // 20)  # 约每 5% 打印一次进度
    processed = 0

    vote = np.zeros((h, w), np.float32)
    wt = np.zeros((h, w), np.float32)
    fg_points: List[Tuple[int, int]] = []
    bg_points: List[Tuple[int, int]] = []
    windows: List[Tuple[int, int, int, int]] = []
    sam2_used = 0

    base_win = cos_win(WINDOW_SIZE)
    if cfg.edge_trim > 0:
        trim = min(cfg.edge_trim, WINDOW_SIZE // 4)
        edge = np.zeros_like(base_win)
        edge[trim : WINDOW_SIZE - trim, trim : WINDOW_SIZE - trim] = 1.0
        use_win = base_win * edge
    else:
        use_win = base_win

    print(f"  [{cfg.name}] 滑窗分割开始，总窗口={total}")
    for y0 in ys:
        for x0 in xs:
            processed += 1
            y1, x1 = min(y0 + WINDOW_SIZE, h), min(x0 + WINDOW_SIZE, w)
            windows.append((x0, y0, x1, y1))
            pi = enh[y0:y1, x0:x1]
            pp = prob[y0:y1, x0:x1]
            pn = nodata_mask[y0:y1, x0:x1]
            if processed == 1 or processed % progress_step == 0 or processed == total:
                pct = processed * 100.0 / max(total, 1)
                print(
                    f"  [{cfg.name}] 进度 {processed}/{total} ({pct:.1f}%) "
                    f"SAM2生效窗={sam2_used}",
                    flush=True,
                )
            if pn.size == 0 or ((~pn).sum() / pn.size) < 0.05 or (pp > 0.5).sum() < 5:
                continue
            pv = pp.copy()
            pv[pn] = 0.0
            fg_w = guidance["fg_mask"][y0:y1, x0:x1] & (~pn)
            bg_w = guidance["bg_mask"][y0:y1, x0:x1] & (~pn)
            uncertain_w = guidance["uncertain_mask"][y0:y1, x0:x1] & (~pn)
            edge_w = guidance["edge_map"][y0:y1, x0:x1] & (~pn)

            if cfg.sampler_mode == "structured":
                fg, bg = structured_prompt_points(pv, fg_w, bg_w, edge_w, N_FG, N_BG)
            else:
                fg = sample_prob(np.where(uncertain_w, pv, 0.0), N_FG)
                bg = sample_prob(np.where(bg_w | edge_w, pv, 0.0), N_BG, high=False)
            fg_points.extend([(x0 + int(x), y0 + int(y)) for x, y in fg])
            bg_points.extend([(x0 + int(x), y0 + int(y)) for x, y in bg])

            # 默认先用概率图
            mask = pv > 0.55
            score = 0.3
            if uncertain_w.sum() >= 8 and cfg.use_sam2 and predictor is not None:
                try:
                    predictor.set_image(cv2.cvtColor(pi, cv2.COLOR_GRAY2RGB))
                    masks, scores, _ = predictor.predict(
                        point_coords=np.array(fg + bg),
                        point_labels=np.array([1] * len(fg) + [0] * len(bg)),
                        multimask_output=True,
                    )
                    best = int(np.argmax(scores))
                    cand_mask = masks[best].astype(bool)
                    cand_score = float(scores[best])
                    if cand_score >= float(cfg.sam2_conf_min):
                        mask[uncertain_w] = cand_mask[uncertain_w]
                        score = cand_score
                        sam2_used += 1
                except Exception:
                    pass

            mask[pn] = False
            ww = max(float(score), 0.3)
            win = use_win[: y1 - y0, : x1 - x0]
            vote[y0:y1, x0:x1] += mask.astype(np.float32) * win * ww
            wt[y0:y1, x0:x1] += win * ww

    conf_map = vote / np.where(wt > 0, wt, 1e-6)
    raw = conf_map > 0.5
    raw[nodata_mask] = False

    # 严格控制变量模式：与主流程一致，增加概率一致性筛选以抑制噪点
    if bool(strict_cv):
        bad = raw & ((1.0 - np.abs(conf_map - prob)) < 0.25) & (prob < 0.75)
        raw[bad] = False

    final = postprocess_mask(raw, nodata_mask, cfg.post_dilate_k)
    if bool(strict_cv) and bool(enable_crf):
        final = dense_crf_refine(enh, final.astype(bool), iterations=int(crf_iter)).astype(bool)
        final = postprocess_mask(final, nodata_mask, cfg.post_dilate_k)
    print(f"  [{cfg.name}] 滑窗分割完成，SAM2生效窗={sam2_used}/{total}")
    return {
        "raw": raw,
        "final": final,
        "windows": windows,
        "fg_points": fg_points,
        "bg_points": bg_points,
        "sam2_used": sam2_used,
        "total_windows": total,
    }


def unsupervised_metrics(mask: np.ndarray, prob: np.ndarray, nodata_mask: np.ndarray, pixel_area_m2: float) -> Dict[str, float]:
    valid = ~nodata_mask
    water = mask.astype(bool)
    nonwater = valid & (~water)
    valid_px = int(valid.sum())
    water_px = int(water.sum())
    coverage = water_px / max(valid_px, 1)

    water_prob = float(prob[water].mean()) if water.any() else 0.0
    nonwater_prob = float(prob[nonwater].mean()) if nonwater.any() else 0.0
    prob_gap = max(0.0, water_prob - nonwater_prob)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(water.astype(np.uint8), connectivity=8)
    comp_count = int(num_labels - 1)
    largest_comp = int(stats[1:, cv2.CC_STAT_AREA].max()) if num_labels > 1 else 0

    # 边界复杂度（越高可能越噪）
    contours, _ = cv2.findContours(water.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    compact_proxy = float(water_px / max(perimeter, 1.0))

    area_km2 = float(water_px * pixel_area_m2 / 1e6)
    score = 0.40 * min(1.0, prob_gap * 2.2) + 0.35 * (1.0 - min(1.0, abs(coverage - 0.2) / 0.2)) + 0.25 * min(
        1.0, compact_proxy / 3.0
    )
    return {
        "area_km2": area_km2,
        "coverage_ratio": float(coverage),
        "water_prob_mean": water_prob,
        "nonwater_prob_mean": nonwater_prob,
        "prob_gap": prob_gap,
        "component_count": float(comp_count),
        "largest_component_px": float(largest_comp),
        "boundary_compact_proxy": compact_proxy,
        "self_score": float(max(0.0, min(1.0, score))),
    }


def overlay_mask(gray: np.ndarray, mask: np.ndarray, nodata_mask: np.ndarray = None, color=(30, 120, 220), alpha=0.78) -> np.ndarray:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if nodata_mask is not None:
        rgb[nodata_mask] = [20, 20, 20]
    idx = mask.astype(bool)
    rgb[idx] = (rgb[idx].astype(np.float32) * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return rgb


def build_visual_base(gray: np.ndarray, nodata: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    out = clahe.apply(gray.astype(np.uint8))
    out = np.clip(np.power(out.astype(np.float32) / 255.0, 0.85) * 255, 0, 255).astype(np.uint8)
    out[nodata] = 0
    return out


def save_mask_view(
    gray: np.ndarray, mask: np.ndarray, nodata: np.ndarray, title: str, out_path: str, reuse_existing: bool = False
) -> None:
    if should_skip_write(out_path, reuse_existing):
        return
    img = overlay_mask(gray, mask, nodata)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(title, fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def save_diff_view(
    gray: np.ndarray,
    base_mask: np.ndarray,
    other_mask: np.ndarray,
    nodata: np.ndarray,
    title: str,
    out_path: str,
    reuse_existing: bool = False,
) -> None:
    if should_skip_write(out_path, reuse_existing):
        return
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb = np.clip(rgb.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    if nodata is not None:
        rgb[nodata] = [20, 20, 20]
    add = other_mask & (~base_mask)
    remove = base_mask & (~other_mask)
    both = base_mask & other_mask
    rgb[both] = [65, 105, 225]
    rgb[add] = [40, 245, 95]
    rgb[remove] = [255, 80, 80]
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title(title, fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def save_diff_only_view(
    base_mask: np.ndarray, other_mask: np.ndarray, nodata: np.ndarray, out_path: str, reuse_existing: bool = False
) -> None:
    if should_skip_write(out_path, reuse_existing):
        return
    canvas = np.zeros((*base_mask.shape, 3), dtype=np.uint8)
    if nodata is not None:
        canvas[nodata] = [20, 20, 20]
    add = other_mask & (~base_mask)
    remove = base_mask & (~other_mask)
    both = base_mask & other_mask
    canvas[both] = [65, 105, 225]
    canvas[add] = [40, 245, 95]
    canvas[remove] = [255, 80, 80]
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def save_binary_mask_view(mask: np.ndarray, nodata: np.ndarray, out_path: str, reuse_existing: bool = False) -> None:
    if should_skip_write(out_path, reuse_existing):
        return
    canvas = np.zeros(mask.shape, dtype=np.uint8)
    canvas[mask.astype(bool)] = 255
    if nodata is not None:
        canvas[nodata] = 35
    cv2.imwrite(out_path, canvas)


def save_prompt_view(
    gray: np.ndarray, fg_points: List[Tuple[int, int]], bg_points: List[Tuple[int, int]], out_path: str, reuse_existing: bool = False
) -> None:
    if should_skip_write(out_path, reuse_existing):
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    if fg_points:
        arr_fg = np.array(fg_points, dtype=np.int32)
        if arr_fg.shape[0] > 1000:
            arr_fg = arr_fg[np.random.choice(arr_fg.shape[0], 1000, replace=False)]
        ax.scatter(arr_fg[:, 0], arr_fg[:, 1], s=7, c="#22c55e", alpha=0.7, label=f"FG({len(fg_points)})")
    if bg_points:
        arr_bg = np.array(bg_points, dtype=np.int32)
        if arr_bg.shape[0] > 1000:
            arr_bg = arr_bg[np.random.choice(arr_bg.shape[0], 1000, replace=False)]
        ax.scatter(arr_bg[:, 0], arr_bg[:, 1], s=7, c="#ef4444", alpha=0.7, label=f"BG({len(bg_points)})")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Prompt Points (sampled)", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def save_key_compare_figure(
    enh: np.ndarray,
    prob: np.ndarray,
    uncertain_mask: np.ndarray,
    edge_map: np.ndarray,
    base_mask: np.ndarray,
    best_mask: np.ndarray,
    nodata: np.ndarray,
    out_path: str,
    reuse_existing: bool = False,
) -> None:
    if should_skip_write(out_path, reuse_existing):
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    im = axes[0, 0].imshow(prob, cmap="Blues", vmin=0, vmax=1)
    axes[0, 0].set_title("Water Probability", fontsize=11)
    axes[0, 0].axis("off")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    unc = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
    unc[uncertain_mask] = [245, 158, 11]
    unc[nodata] = [20, 20, 20]
    axes[0, 1].imshow(unc)
    axes[0, 1].set_title("Uncertain Region (0.4~0.6)", fontsize=11)
    axes[0, 1].axis("off")

    edge = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
    edge[edge_map] = [255, 200, 60]
    edge[nodata] = [20, 20, 20]
    axes[1, 0].imshow(edge)
    axes[1, 0].set_title("Edge Map (Canny)", fontsize=11)
    axes[1, 0].axis("off")

    diff = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
    diff[nodata] = [20, 20, 20]
    both = base_mask & best_mask
    add = best_mask & (~base_mask)
    remove = base_mask & (~best_mask)
    diff[both] = (diff[both].astype(np.float32) * 0.35 + np.array([30, 120, 220]) * 0.65).astype(np.uint8)
    diff[add] = [60, 200, 80]
    diff[remove] = [220, 60, 60]
    axes[1, 1].imshow(diff)
    axes[1, 1].set_title("Final Mask Delta (Green:+, Red:-)", fontsize=11)
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def write_compare_summary(
    out_dir: str,
    label: str,
    base_name: str,
    best_name: str,
    area_base: float,
    area_best: float,
    uncertain_ratio: float,
    edge_ratio: float,
) -> str:
    delta = area_best - area_base
    lines = [
        f"# {label} 合并运行对比说明",
        "",
        f"- 对比基线：`{base_name}`",
        f"- 对比改造：`{best_name}`",
        "",
        "## 关键图含义",
        "- 概率图：传统概率主干输出。",
        "- 不确定区：仅这部分交给SAM2细化。",
        "- 边界图：作为背景/边界负样本约束。",
        "- 最终掩膜对比：绿色代表改造后新增，红色代表改造后减少。",
        "",
        "## 数值摘要",
        f"- 基线面积：`{area_base:.4f} km²`",
        f"- 改造面积：`{area_best:.4f} km²`",
        f"- 面积变化（改造-基线）：`{delta:+.4f} km²`",
        f"- 不确定区占比：`{uncertain_ratio:.2%}`",
        f"- 边界像素占比：`{edge_ratio:.2%}`",
    ]
    path = os.path.join(out_dir, "compare_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_figure_guide(out_dir: str, reuse_existing: bool = False) -> str:
    path = os.path.join(out_dir, "figure_guide.md")
    if should_skip_write(path, reuse_existing):
        return path
    lines = [
        "# 图像说明与判读建议",
        "",
        "## 关键图含义",
        "- `probability_map.png`：水体概率图，值越高越可能是水体。",
        "- `edge_map.png`：边缘图（Canny），用于刻画纹理/边界，帮助限制误分割。",
        "- `uncertain_map.png`：蓝=前景高置信，灰=背景高置信，橙=不确定区（0.4~0.6）。",
        "- `guided_filter_residual.png`：引导滤波残差，亮区表示局部变化更明显。",
        "",
        "## 对比图颜色约定",
        "- 蓝色：基线与当前实验共同识别为水体。",
        "- 绿色：相较基线新增水体（other - base）。",
        "- 红色：相较基线减少水体（base - other）。",
        "- 黑/深灰：非水体或 NoData。",
        "",
        "## 判读建议",
        "- 先看 `*_diff_only.png`，快速定位真正变化区域。",
        "- 再看 `*_diff_vs_prob_only.png`，确认变化是否贴合底图结构。",
        "- 最后看 `*_binary.png`，评估连通性与碎片噪声。",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_csv(rows: List[Dict[str, object]], path: str) -> None:
    keys = [
        "name",
        "use_sam2",
        "sam2_used",
        "total_windows",
        "area_km2",
        "coverage_ratio",
        "water_prob_mean",
        "nonwater_prob_mean",
        "prob_gap",
        "component_count",
        "largest_component_px",
        "boundary_compact_proxy",
        "self_score",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})


def write_markdown(rows: List[Dict[str, object]], md_path: str) -> None:
    rows_sorted = sorted(rows, key=lambda x: float(x["self_score"]), reverse=True)
    lines = [
        "# 无标注消融实验结果",
        "",
        "排序依据：`self_score`（综合概率一致性、覆盖合理性、边界紧凑度）",
        "",
        "| 实验 | SAM2生效窗 | 面积(km²) | prob_gap | 连通域数 | self_score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r['name']} | {int(r['sam2_used'])}/{int(r['total_windows'])} | "
            f"{float(r['area_km2']):.4f} | {float(r['prob_gap']):.4f} | "
            f"{int(r['component_count'])} | {float(r['self_score']):.4f} |"
        )
    lines.append("")
    lines.append("建议优先关注：`self_score`高且`component_count`不过高的配置。")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_html_report(out_dir: str, title: str, rows: List[Dict[str, object]]) -> str:
    html_path = os.path.join(out_dir, "experiment_report.html")
    trs = []
    rows_sorted = sorted(rows, key=lambda x: float(x["self_score"]), reverse=True)
    for r in rows_sorted:
        name = str(r["name"])
        mask_img = f"{name}_mask.png"
        diff_img = f"{name}_diff_vs_prob_only.png"
        diff_only_img = f"{name}_diff_only.png"
        binary_img = f"{name}_binary.png"
        trs.append(
            f"<tr><td>{name}</td><td>{int(r['sam2_used'])}/{int(r['total_windows'])}</td>"
            f"<td>{float(r['area_km2']):.4f}</td><td>{float(r['prob_gap']):.4f}</td>"
            f"<td>{int(r['component_count'])}</td><td>{float(r['self_score']):.4f}</td></tr>"
            f"<tr><td colspan='6'>"
            f"<div class='imgs'><img src='{mask_img}'/><img src='{diff_img}'/><img src='{diff_only_img}'/><img src='{binary_img}'/></div>"
            f"</td></tr>"
        )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>
    body{{font-family:"Microsoft YaHei",Arial,sans-serif;background:#f8fafc;color:#0f172a;margin:0}}
    .wrap{{max-width:1280px;margin:0 auto;padding:20px}}
    h1{{margin:0 0 10px}}
    .muted{{color:#475569;margin-bottom:14px}}
    table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e2e8f0}}
    th,td{{border:1px solid #e2e8f0;padding:8px;text-align:left}}
    th{{background:#eff6ff;color:#1d4ed8}}
    .imgs{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
    .imgs img{{width:100%;border:1px solid #e2e8f0;border-radius:8px}}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{title}</h1>
    <div class="muted">无标注自检消融：Prob-only / SAM2随机点 / SAM2结构化点 / SAM2结构化+门控。</div>
    <table>
      <thead><tr><th>实验</th><th>SAM2生效窗</th><th>面积(km²)</th><th>prob_gap</th><th>连通域数</th><th>self_score</th></tr></thead>
      <tbody>{''.join(trs)}</tbody>
    </table>
  </div>
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="无标注消融实验脚本（建议固定 d28.tif）")
    parser.add_argument("--input-file", required=True, help="输入单景影像路径（如 d28.tif）")
    parser.add_argument("--out-dir", default=os.path.join("output", "ablation_d28"), help="输出目录")
    parser.add_argument("--pixel-area-m2", type=float, default=DEFAULT_PIXEL_AREA_M2, help="像素面积(m²)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--best-exp-name", default="sam2_structured_gated", help="用于改造前后对比的目标实验名")
    parser.add_argument("--archive-dir", default=os.path.join("output", "common_archive"), help="通用图存档目录")
    parser.add_argument("--reuse-figures", dest="reuse_figures", action="store_true", default=True, help="复用已存在图像")
    parser.add_argument("--no-reuse-figures", dest="reuse_figures", action="store_false", help="强制重绘所有图像")
    parser.add_argument("--strict-cv", dest="strict_cv", action="store_true", default=True, help="严格控制变量：公共前后处理对齐 flood_detection")
    parser.add_argument("--loose-cv", dest="strict_cv", action="store_false", help="关闭严格控制变量，使用原有较自由配置")
    parser.add_argument("--enable-crf", dest="enable_crf", action="store_true", default=ENABLE_CRF_DEFAULT, help="启用CRF细化（strict-cv下生效）")
    parser.add_argument("--disable-crf", dest="enable_crf", action="store_false", help="禁用CRF细化")
    parser.add_argument("--crf-iter", type=int, default=CRF_ITER_DEFAULT, help="CRF迭代次数（strict-cv下生效）")
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/6] 读取输入影像...")
    with rasterio.open(args.input_file) as src:
        arr = src.read(1).astype(np.float32)
    label = os.path.splitext(os.path.basename(args.input_file))[0]
    print(f"[1/6] 完成：{label}，尺寸={arr.shape[1]}x{arr.shape[0]}")

    print("[2/6] 构建 NoData 与概率图组件...")
    nodata = detect_nodata(arr, label)
    if bool(args.strict_cv):
        print("[2/6] 严格控制变量模式：前处理参数对齐 flood_detection")
        comp = build_prob_components(
            arr,
            nodata,
            gf_radius=6,
            gf_eps=0.02,
            deep_fixed_thr=8.0,
            shadow_range_thr=15,
        )
        post_dilate_k = DEFAULT_POST_DILATE_K
    else:
        comp = build_prob_components(
            arr,
            nodata,
            gf_radius=8,
            gf_eps=0.01,
            deep_pct=6.0,
            deep_fixed_thr=None,
            shadow_range_thr=15,
        )
        post_dilate_k = 3
    enh = comp["enh"]
    prob = comp["prob"]
    base_gray = build_visual_base(enh, nodata)
    print("[2/6] 完成")

    print("[3/6] 初始化 SAM2（不可用会自动回退概率图）...")
    predictor = init_sam2_predictor(enable_sam2=True)
    print("[3/6] 完成")

    configs = [
        ExperimentConfig("prob_only", False, "random", 0.99, 0, post_dilate_k),
        ExperimentConfig("sam2_random", True, "random", 0.30, 0, post_dilate_k),
        ExperimentConfig("sam2_structured", True, "structured", 0.45, 24, post_dilate_k),
        ExperimentConfig("sam2_structured_gated", True, "structured", 0.60, 32, post_dilate_k),
    ]

    print(f"[4/6] 运行消融实验，共 {len(configs)} 组...")
    all_rows: List[Dict[str, object]] = []
    results: Dict[str, Dict[str, object]] = {}
    for idx, cfg in enumerate(configs, start=1):
        print(f"[4/6] ({idx}/{len(configs)}) 开始：{cfg.name}")
        trace = segment_variant(
            predictor,
            enh,
            prob,
            nodata,
            comp,
            cfg,
            strict_cv=bool(args.strict_cv),
            enable_crf=bool(args.enable_crf),
            crf_iter=int(args.crf_iter),
        )
        metrics = unsupervised_metrics(trace["final"], prob, nodata, float(args.pixel_area_m2))
        row = {
            "name": cfg.name,
            "use_sam2": int(cfg.use_sam2),
            "sam2_used": int(trace["sam2_used"]),
            "total_windows": int(trace["total_windows"]),
            **metrics,
        }
        all_rows.append(row)
        results[cfg.name] = trace
        print(
            f"[4/6] ({idx}/{len(configs)}) 完成：{cfg.name} "
            f"self_score={float(metrics['self_score']):.4f}"
        )

    print("[5/6] 生成可视化图与报告文件...")
    base_mask = results["prob_only"]["final"]
    for cfg in configs:
        name = cfg.name
        trace = results[name]
        save_mask_view(
            base_gray,
            trace["final"],
            nodata,
            f"{name} Mask Overlay",
            os.path.join(args.out_dir, f"{name}_mask.png"),
            reuse_existing=bool(args.reuse_figures),
        )
        save_diff_view(
            base_gray,
            base_mask,
            trace["final"],
            nodata,
            f"{name} vs prob_only (G:+, R:-)",
            os.path.join(args.out_dir, f"{name}_diff_vs_prob_only.png"),
            reuse_existing=bool(args.reuse_figures),
        )
        save_diff_only_view(
            base_mask,
            trace["final"],
            nodata,
            os.path.join(args.out_dir, f"{name}_diff_only.png"),
            reuse_existing=bool(args.reuse_figures),
        )
        save_binary_mask_view(
            trace["final"],
            nodata,
            os.path.join(args.out_dir, f"{name}_binary.png"),
            reuse_existing=bool(args.reuse_figures),
        )
        save_prompt_view(
            base_gray,
            trace["fg_points"],
            trace["bg_points"],
            os.path.join(args.out_dir, f"{name}_prompts.png"),
            reuse_existing=bool(args.reuse_figures),
        )

    archive_label_dir = os.path.join(args.archive_dir, label)
    os.makedirs(archive_label_dir, exist_ok=True)

    # 额外输出：概率图中间效果，便于解释深水与阴影机制
    p_prob = os.path.join(args.out_dir, "probability_map.png")
    if not should_skip_write(p_prob, bool(args.reuse_figures)):
        plt.figure(figsize=(7, 7))
        plt.imshow(comp["prob"], cmap="Blues", vmin=0, vmax=1)
        plt.title("Water Probability Map", fontsize=11)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(p_prob, dpi=130, bbox_inches="tight")
        plt.close()
    copy_to_archive(p_prob, os.path.join(archive_label_dir, "probability_map.png"), bool(args.reuse_figures))

    p_res = os.path.join(args.out_dir, "guided_filter_residual.png")
    if not should_skip_write(p_res, bool(args.reuse_figures)):
        plt.figure(figsize=(7, 7))
        diff = cv2.absdiff(comp["gf"], enh)
        plt.imshow(diff, cmap="magma")
        plt.title("Guided Filter Residual |gf-enh|", fontsize=11)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(p_res, dpi=130, bbox_inches="tight")
        plt.close()
    copy_to_archive(p_res, os.path.join(archive_label_dir, "guided_filter_residual.png"), bool(args.reuse_figures))

    p_edge = os.path.join(args.out_dir, "edge_map.png")
    if not should_skip_write(p_edge, bool(args.reuse_figures)):
        edge_vis = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)
        edge_vis[comp["edge_map"]] = [255, 200, 60]
        edge_vis[nodata] = [20, 20, 20]
        cv2.imwrite(p_edge, cv2.cvtColor(edge_vis, cv2.COLOR_RGB2BGR))
    copy_to_archive(p_edge, os.path.join(archive_label_dir, "edge_map.png"), bool(args.reuse_figures))

    p_unc = os.path.join(args.out_dir, "uncertain_map.png")
    if not should_skip_write(p_unc, bool(args.reuse_figures)):
        unc_vis = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)
        unc_vis[comp["fg_mask"]] = [37, 99, 235]
        unc_vis[comp["bg_mask"]] = [100, 116, 139]
        unc_vis[comp["uncertain_mask"]] = [245, 158, 11]
        unc_vis[nodata] = [20, 20, 20]
        cv2.imwrite(p_unc, cv2.cvtColor(unc_vis, cv2.COLOR_RGB2BGR))
    copy_to_archive(p_unc, os.path.join(archive_label_dir, "uncertain_map.png"), bool(args.reuse_figures))

    csv_path = os.path.join(args.out_dir, "experiment_metrics.csv")
    md_path = os.path.join(args.out_dir, "experiment_summary.md")
    html_path = write_html_report(args.out_dir, f"{label} 无标注消融实验报告", all_rows)
    write_csv(all_rows, csv_path)
    write_markdown(all_rows, md_path)

    base_name = "prob_only"
    best_name = str(args.best_exp_name)
    if best_name not in results:
        best_name = "sam2_structured_gated" if "sam2_structured_gated" in results else list(results.keys())[-1]
    base_mask = results[base_name]["final"]
    best_mask = results[best_name]["final"]
    p_key = os.path.join(args.out_dir, "key_maps_before_after.png")
    save_key_compare_figure(
        base_gray,
        prob,
        comp["uncertain_mask"],
        comp["edge_map"],
        base_mask,
        best_mask,
        nodata,
        p_key,
        reuse_existing=bool(args.reuse_figures),
    )
    area_base = float(base_mask.sum() * args.pixel_area_m2 / 1e6)
    area_best = float(best_mask.sum() * args.pixel_area_m2 / 1e6)
    compare_md = write_compare_summary(
        args.out_dir,
        label,
        base_name,
        best_name,
        area_base,
        area_best,
        float(comp["uncertain_mask"].mean()),
        float(comp["edge_map"].mean()),
    )
    figure_guide = write_figure_guide(args.out_dir, reuse_existing=bool(args.reuse_figures))
    copy_to_archive(figure_guide, os.path.join(archive_label_dir, "figure_guide.md"), bool(args.reuse_figures))
    print("[5/6] 完成")

    print("[6/6] 汇总输出路径")
    print("=" * 60)
    print(f"实验完成：{args.out_dir}")
    print(f"指标表：{csv_path}")
    print(f"结论摘要：{md_path}")
    print(f"可视化报告：{html_path}")
    print(f"关键图对比：{p_key}")
    print(f"改造前后说明：{compare_md}")
    print(f"图像说明：{figure_guide}")
    print(f"通用图存档：{archive_label_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
