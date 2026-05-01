import argparse
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

UNCERTAIN_LOW = 0.4
UNCERTAIN_HIGH = 0.6


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


def build_prob_components(arr: np.ndarray, nodata_mask: np.ndarray):
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
    no_shadow = water_cand & ~shadow
    water_final = no_shadow | deep_water
    prob = np.zeros(arr.shape, np.float32)
    prob[water_final] = 1.0
    prob[shadow] = 0.2
    prob[deep_water] = 1.0
    prob[nodata_mask] = 0.0
    prob = cv2.GaussianBlur(prob, (7, 7), 2)
    prob[nodata_mask] = 0.0
    fg_mask = (prob >= UNCERTAIN_HIGH) & ~nodata_mask
    bg_mask = (prob <= UNCERTAIN_LOW) & ~nodata_mask
    uncertain_mask = (prob > UNCERTAIN_LOW) & (prob < UNCERTAIN_HIGH) & ~nodata_mask
    return {
        "enh": enh,
        "gf": gf,
        "otsu": int(otsu),
        "thresh": thresh,
        "water_cand": water_cand,
        "shadow": shadow,
        "deep_water": deep_water,
        "water_final": water_final,
        "prob": prob,
        "fg_mask": fg_mask,
        "bg_mask": bg_mask,
        "uncertain_mask": uncertain_mask,
        "nodata": nodata_mask,
    }


def overlay_mask(gray: np.ndarray, mask: np.ndarray, nodata: np.ndarray, color, alpha: float = 0.75) -> np.ndarray:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb[nodata] = [20, 20, 20]
    idx = mask.astype(bool)
    rgb[idx] = (rgb[idx].astype(np.float32) * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return rgb


def save_process_figure(label: str, comp: dict, out_path: str) -> None:
    enh = comp["enh"]
    nodata = comp["nodata"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))

    axes[0, 0].imshow(enh, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("1) 增强图")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(comp["gf"], cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("2) 引导滤波后")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(overlay_mask(enh, comp["water_cand"], nodata, color=(30, 120, 220), alpha=0.8))
    axes[0, 2].set_title(f"3) 候选水体 (Otsu={comp['otsu']}, 压缩阈值={comp['thresh']})")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(overlay_mask(enh, comp["shadow"], nodata, color=(220, 60, 60), alpha=0.85))
    axes[1, 0].set_title("4) 阴影排除区域")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay_mask(enh, comp["deep_water"], nodata, color=(60, 200, 80), alpha=0.85))
    axes[1, 1].set_title("5) 深水补偿区域")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(overlay_mask(enh, comp["water_final"], nodata, color=(30, 120, 220), alpha=0.8))
    axes[1, 2].set_title("6) 最终水体先验")
    axes[1, 2].axis("off")

    im = axes[2, 0].imshow(comp["prob"], cmap="Blues", vmin=0, vmax=1)
    axes[2, 0].set_title("7) 概率图（平滑后）")
    axes[2, 0].axis("off")
    plt.colorbar(im, ax=axes[2, 0], fraction=0.046, pad=0.04)

    axes[2, 1].imshow(overlay_mask(enh, comp["fg_mask"], nodata, color=(37, 99, 235), alpha=0.8))
    axes[2, 1].set_title("8) 高置信前景区 (p>=0.6)")
    axes[2, 1].axis("off")

    tri = cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)
    tri[nodata] = [20, 20, 20]
    tri[comp["bg_mask"]] = [100, 116, 139]      # 背景灰蓝
    tri[comp["uncertain_mask"]] = [245, 158, 11]  # 不确定橙
    tri[comp["fg_mask"]] = [37, 99, 235]        # 前景蓝
    axes[2, 2].imshow(tri)
    axes[2, 2].set_title("9) 三区划分（背景/不确定/前景）")
    axes[2, 2].axis("off")

    fig.suptitle(f"{label} 概率图构建与三区划分过程", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="概率图构建与三区划分过程可视化")
    parser.add_argument("--input-files", nargs="*", default=[
        r"d:\workPlace\graduate\d27.tif",
        r"d:\workPlace\graduate\d28.tif",
        r"d:\workPlace\graduate\d29.tif",
    ])
    parser.add_argument("--out-dir", default=r"d:\workPlace\graduate\output\prob_threezone_process")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for fp in args.input_files:
        label = os.path.splitext(os.path.basename(fp))[0]
        with rasterio.open(fp) as src:
            arr = src.read(1).astype(np.float32)
        nodata = detect_nodata(arr)
        comp = build_prob_components(arr, nodata)
        out_path = os.path.join(args.out_dir, f"{label}_prob_threezone_process.png")
        save_process_figure(label, comp, out_path)
        print(out_path)


if __name__ == "__main__":
    main()
