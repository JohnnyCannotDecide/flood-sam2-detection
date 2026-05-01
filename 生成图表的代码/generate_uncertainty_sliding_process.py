import argparse
import os
import traceback
from typing import List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from flood_detection import (
    MIN_WATER_AREA,
    MORPH_CLOSE_K,
    N_BG,
    N_FG,
    SAM2_CONF_MIN_DEFAULT,
    build_adaptive_windows,
    cos_win,
    detect_nodata,
    enhance,
    guided_filter,
    init_sam2_predictor,
    sample_structured_points,
)

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def postprocess_mask(raw: np.ndarray, nodata_mask: np.ndarray, post_dilate_k: int = 7, lite_mode: bool = True) -> np.ndarray:
    k = np.ones((MORPH_CLOSE_K, MORPH_CLOSE_K), np.uint8)
    closed = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, k)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if int(post_dilate_k) > 0:
        kd = np.ones((int(post_dilate_k), int(post_dilate_k)), np.uint8)
        opened = cv2.dilate(opened, kd, iterations=1)
    if lite_mode:
        final = opened.astype(bool)
        final[nodata_mask] = False
        return final
    num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    final = np.zeros_like(opened)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_WATER_AREA:
            final[labels_map == i] = 1
    final = final.astype(bool) & ~nodata_mask
    return final


def build_prob_and_guidance(arr: np.ndarray, nodata: np.ndarray):
    enh = enhance(arr, nodata)
    enh_fill = enh.copy()
    enh_fill[nodata] = 128
    gf = guided_filter(enh_fill)
    valid_vals = gf[~nodata].flatten().astype(np.uint8)
    otsu, _ = cv2.threshold(valid_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = int(min(otsu * 0.6, 45))
    water_cand = (gf < thresh) & ~nodata & (arr > 0)
    deep_water = (arr > 0) & (arr <= 8) & ~nodata
    lr = cv2.dilate(gf, np.ones((11, 11), np.uint8)) - cv2.erode(gf, np.ones((11, 11), np.uint8))
    shadow = water_cand & (lr < 15) & ~deep_water
    water_final = (water_cand & ~shadow) | deep_water
    prob = np.zeros(arr.shape, np.float32)
    prob[water_final] = 1.0
    prob[shadow] = 0.2
    prob[deep_water] = 1.0
    prob[nodata] = 0.0
    prob = cv2.GaussianBlur(prob, (7, 7), 2)
    prob = prob.astype(np.float16)
    prob[nodata] = 0.0
    guidance = {
        "fg_mask": (prob >= 0.6) & ~nodata,
        "bg_mask": (prob <= 0.4) & ~nodata,
        "uncertain_mask": (prob > 0.4) & (prob < 0.6) & ~nodata,
        "edge_map": (cv2.Canny(gf, 50, 130) > 0) & ~nodata,
    }
    return prob, enh, guidance


def overlay_mask(gray: np.ndarray, mask: np.ndarray, nodata_mask: np.ndarray, color=(30, 120, 220), alpha=0.78) -> np.ndarray:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb[nodata_mask] = [20, 20, 20]
    idx = mask.astype(bool)
    rgb[idx] = (rgb[idx].astype(np.float32) * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return rgb


def save_windows_vis(
    enh: np.ndarray,
    nodata: np.ndarray,
    windows: List[Tuple[int, int, int, int, int]],
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    bg = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
    bg[nodata] = [20, 20, 20]
    ax.imshow(bg)
    for x0, y0, x1, y1, ws in windows:
        color = "#f59e0b" if ws >= 640 else "#06b6d4"
        lw = 0.4 if ws >= 640 else 0.25
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=lw, alpha=0.45)
        ax.add_patch(rect)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_prompt_vis(
    enh: np.ndarray,
    nodata: np.ndarray,
    fg_points: List[Tuple[int, int]],
    bg_points: List[Tuple[int, int]],
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    bg = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
    bg[nodata] = [20, 20, 20]
    ax.imshow(bg)
    if fg_points:
        fg = np.array(fg_points, dtype=np.int32)
        if fg.shape[0] > 2000:
            fg = fg[np.random.choice(fg.shape[0], 2000, replace=False)]
        ax.scatter(fg[:, 0], fg[:, 1], s=5, c="#22c55e", alpha=0.65, label=f"前景点({len(fg_points)})")
    if bg_points:
        bgp = np.array(bg_points, dtype=np.int32)
        if bgp.shape[0] > 2000:
            bgp = bgp[np.random.choice(bgp.shape[0], 2000, replace=False)]
        ax.scatter(bgp[:, 0], bgp[:, 1], s=5, c="#ef4444", alpha=0.65, label=f"背景点({len(bg_points)})")
    ax.legend(loc="lower right", fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_windows_vis_cv(
    enh: np.ndarray, nodata: np.ndarray, windows: List[Tuple[int, int, int, int, int]], out_path: str
) -> None:
    img = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
    img[nodata] = [20, 20, 20]
    for x0, y0, x1, y1, ws in windows:
        color = (11, 182, 211) if ws < 640 else (11, 158, 245)  # BGR
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), color, 1)
    cv2.imwrite(out_path, img)


def save_prompt_vis_cv(
    enh: np.ndarray, nodata: np.ndarray, fg_points: List[Tuple[int, int]], bg_points: List[Tuple[int, int]], out_path: str
) -> None:
    img = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
    img[nodata] = [20, 20, 20]
    for x, y in fg_points:
        cv2.circle(img, (x, y), 1, (34, 197, 94), -1)
    for x, y in bg_points:
        cv2.circle(img, (x, y), 1, (68, 68, 239), -1)
    cv2.imwrite(out_path, img)


def main() -> None:
    parser = argparse.ArgumentParser(description="不确定性驱动滑窗分割过程可视化")
    parser.add_argument("--input-file", default=r"d:\workPlace\graduate\d28.tif")
    parser.add_argument("--out-dir", default=r"d:\workPlace\graduate\output\uncertainty_sliding_vis")
    parser.add_argument("--sam2-conf-min", type=float, default=SAM2_CONF_MIN_DEFAULT)
    parser.add_argument("--disable-sam2", action="store_true")
    parser.add_argument("--viz-scale", type=int, default=4, help="可视化降采样倍数，>1可显著提速")
    parser.add_argument("--keep-key-only", action="store_true", help="仅保留02~05关键图，并删除重复中间图")
    parser.add_argument("--lite-post", action="store_true", default=True, help="可视化使用轻量后处理，降低大图内存占用")
    parser.add_argument("--window-subsample", type=int, default=1, help="窗口子采样步长，1表示全窗口")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "_run_log.txt")
    with open(log_path, "w", encoding="utf-8") as log:
        def mark(msg: str) -> None:
            log.write(msg + "\n")
            log.flush()

        try:
            mark("start")
            with open(os.path.join(args.out_dir, "_run_marker.txt"), "w", encoding="utf-8") as f:
                f.write("started")
            label = os.path.splitext(os.path.basename(args.input_file))[0]
            mark(f"label={label}")

            with rasterio.open(args.input_file) as src:
                arr = src.read(1).astype(np.float32)
            mark(f"read_arr_shape={arr.shape}")
            if int(args.viz_scale) > 1:
                h, w = arr.shape
                th, tw = max(1, h // int(args.viz_scale)), max(1, w // int(args.viz_scale))
                arr = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA)
                mark(f"resized_shape={arr.shape}")

            nodata = detect_nodata(arr, label)
            mark("nodata_done")
            prob, enh, guidance = build_prob_and_guidance(arr, nodata)
            mark("prob_and_guidance_done")
            uncertain = guidance["uncertain_mask"]
            fg_mask = guidance["fg_mask"]
            bg_mask = guidance["bg_mask"]
            edge_map = guidance["edge_map"]

            predictor = init_sam2_predictor(enable_sam2=not args.disable_sam2)
            windows = build_adaptive_windows(arr.shape[0], arr.shape[1], uncertain)
            mark(f"windows={len(windows)}")

            vote = np.zeros(arr.shape, np.float16)
            wt = np.zeros(arr.shape, np.float16)
            win_count = np.zeros(arr.shape, np.uint16)
            fg_points: List[Tuple[int, int]] = []
            bg_points: List[Tuple[int, int]] = []
            sam2_ok = 0

            for wi, (x0, y0, x1, y1, win_size) in enumerate(windows):
                if int(args.window_subsample) > 1 and (wi % int(args.window_subsample) != 0):
                    continue
                pi = enh[y0:y1, x0:x1]
                pp = prob[y0:y1, x0:x1]
                pn = nodata[y0:y1, x0:x1]
                roi = uncertain[y0:y1, x0:x1] & (~pn)
                if (~pn).sum() / max(pn.size, 1) < 0.05:
                    continue
                pv = pp.copy()
                pv[pn] = 0.0
                base_mask = pv > 0.55
                mask = base_mask.copy()
                score = 0.3

                if roi.sum() >= 8:
                    fg_w = fg_mask[y0:y1, x0:x1]
                    bg_w = bg_mask[y0:y1, x0:x1]
                    edge_w = edge_map[y0:y1, x0:x1]
                    fg, bg = sample_structured_points(fg_w, bg_w, edge_w, n_fg=N_FG, n_bg=N_BG)
                    fg_points.extend([(x0 + int(x), y0 + int(y)) for x, y in fg])
                    bg_points.extend([(x0 + int(x), y0 + int(y)) for x, y in bg])

                    if predictor is not None:
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
                            if cand_score >= float(args.sam2_conf_min):
                                mask[roi] = cand_mask[roi]
                                score = cand_score
                                sam2_ok += 1
                        except Exception:
                            pass

                mask[pn] = False
                win = cos_win(win_size)[: y1 - y0, : x1 - x0]
                ww = max(score, 0.3)
                vote[y0:y1, x0:x1] += (mask.astype(np.float16) * win.astype(np.float16) * np.float16(ww))
                wt[y0:y1, x0:x1] += (win.astype(np.float16) * np.float16(ww))
                win_count[y0:y1, x0:x1] += np.uint16(1)

            mark("voting_done")
            conf_map = vote.astype(np.float32) / np.where(wt > 0, wt, np.float16(1e-3)).astype(np.float32)
            raw = conf_map > 0.5
            raw[nodata] = False
            final = postprocess_mask(raw, nodata, post_dilate_k=7, lite_mode=bool(args.lite_post))
            mark("postprocess_done")

            # 1) 不确定区
            p1 = os.path.join(args.out_dir, f"{label}_01_uncertain_map.png")
            unc_vis = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
            unc_vis[nodata] = [20, 20, 20]
            unc_vis[uncertain] = [245, 158, 11]
            cv2.imwrite(p1, cv2.cvtColor(unc_vis, cv2.COLOR_RGB2BGR))

            # 2) 扫窗轨迹
            p2 = os.path.join(args.out_dir, f"{label}_02_adaptive_windows.png")
            if bool(args.keep_key_only) and int(args.viz_scale) == 1:
                save_windows_vis_cv(enh, nodata, windows, p2)
            else:
                save_windows_vis(enh, nodata, windows, p2)

            # 3) 正负提示点
            p3 = os.path.join(args.out_dir, f"{label}_03_prompt_points.png")
            if bool(args.keep_key_only) and int(args.viz_scale) == 1:
                save_prompt_vis_cv(enh, nodata, fg_points, bg_points, p3)
            else:
                save_prompt_vis(enh, nodata, fg_points, bg_points, p3)

            # 4) 投票与置信
            p4 = os.path.join(args.out_dir, f"{label}_04_voting_confidence.png")
            if bool(args.keep_key_only) and int(args.viz_scale) == 1:
                c0 = cv2.normalize(win_count.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                c0 = cv2.applyColorMap(c0, cv2.COLORMAP_VIRIDIS)
                c1 = cv2.normalize(conf_map.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                c1 = cv2.applyColorMap(c1, cv2.COLORMAP_MAGMA)
                c2 = (raw.astype(np.uint8) * 255)
                c2 = cv2.cvtColor(c2, cv2.COLOR_GRAY2BGR)
                # 全分辨率下避免大图拼接导致内存峰值过高，改为保存核心投票置信图
                cv2.imwrite(p4, c1)
                cv2.imwrite(os.path.join(args.out_dir, f"{label}_04a_window_count.png"), c0)
                cv2.imwrite(os.path.join(args.out_dir, f"{label}_04b_raw_mask.png"), c2)
            else:
                fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
                im0 = axes[0].imshow(win_count, cmap="viridis")
                axes[0].set_title("窗口覆盖次数")
                axes[0].axis("off")
                plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

                im1 = axes[1].imshow(conf_map, cmap="magma", vmin=0, vmax=1)
                axes[1].set_title("投票后置信图")
                axes[1].axis("off")
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                axes[2].imshow(raw, cmap="gray")
                axes[2].set_title("投票阈值后原始掩膜")
                axes[2].axis("off")
                plt.tight_layout()
                plt.savefig(p4, bbox_inches="tight")
                plt.close(fig)

            # 5) 最终掩膜
            p5 = os.path.join(args.out_dir, f"{label}_05_final_mask_compare.png")
            if bool(args.keep_key_only) and int(args.viz_scale) == 1:
                left = overlay_mask(enh, raw, nodata, color=(30, 120, 220), alpha=0.75)
                right = overlay_mask(enh, final, nodata, color=(30, 120, 220), alpha=0.75)
                cv2.imwrite(p5, cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(args.out_dir, f"{label}_05a_raw_overlay.png"), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
            else:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
                axes[0].imshow(overlay_mask(enh, raw, nodata, color=(30, 120, 220), alpha=0.75))
                axes[0].set_title("后处理前")
                axes[0].axis("off")
                axes[1].imshow(overlay_mask(enh, final, nodata, color=(30, 120, 220), alpha=0.75))
                axes[1].set_title("后处理后")
                axes[1].axis("off")
                plt.tight_layout()
                plt.savefig(p5, bbox_inches="tight")
                plt.close(fig)
            mark("figures_saved")

            print("=" * 60)
            print(f"输出目录: {args.out_dir}")
            print(f"窗口总数: {len(windows)}")
            print(f"SAM2生效窗口: {sam2_ok}")
            print(f"提示点数量: FG={len(fg_points)} BG={len(bg_points)}")
            print("已生成: ")
            print(p1)
            print(p2)
            print(p3)
            print(p4)
            print(p5)
            print("=" * 60)

            if bool(args.keep_key_only):
                to_remove = [
                    os.path.join(args.out_dir, f"{label}_01_uncertain_map.png"),
                    os.path.join(args.out_dir, f"{label}_prob_map.png"),
                    os.path.join(args.out_dir, f"{label}_edge_map.png"),
                    os.path.join(args.out_dir, f"{label}_uncertain_map.png"),
                ]
                for rp in to_remove:
                    if os.path.exists(rp):
                        os.remove(rp)
                mark("keep_key_only_cleanup_done")
            mark("done")
        except Exception:
            mark("exception")
            mark(traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
