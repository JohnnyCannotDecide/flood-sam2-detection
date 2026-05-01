import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio

matplotlib.use("Agg")


def main() -> None:
    files = [
        r"d:\workPlace\graduate\d27.tif",
        r"d:\workPlace\graduate\d28.tif",
        r"d:\workPlace\graduate\d29.tif",
    ]
    out_path = r"d:\workPlace\graduate\output\fig_original_row.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    imgs = []
    for fp in files:
        with rasterio.open(fp) as src:
            arr = src.read(1).astype(np.float32)
        # 仅用于可视化显示，保持原始灰度取值范围（裁剪到8-bit显示域）
        img = np.clip(arr, 0, 255).astype(np.uint8)
        imgs.append(img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=180)
    for ax, img in zip(axes, imgs):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
