"""UNet++ inference time on BrEaST images: single forward pass and TTA + post-processing.

Excludes image loading/resizing and the host-to-GPU copy. The first 10 images are warm-up.
Usage:
    python unetpp/inference_time.py --breast_dir <BrEaST root> --checkpoint <ckpt.pth> --out_dir <folder>
"""
import os
import json
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import BUSDataset, build_model, list_breast_cases, tta_predict, post_process_mask

N_WARMUP = 10


def summarize(name, t):
    t = np.array(t) * 1000
    return {"name": name, "n_images": len(t), "mean_ms": float(t.mean()), "std_ms": float(t.std()),
            "median_ms": float(np.median(t)), "min_ms": float(t.min()), "max_ms": float(t.max())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--breast_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_area", type=int, default=100)
    args = p.parse_args()

    images_dir = os.path.join(args.breast_dir, "Images")
    masks_dir = os.path.join(args.breast_dir, "Masks")
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    imgs, groups = list_breast_cases(images_dir, masks_dir)
    model = build_model(encoder_weights=None).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    loader = DataLoader(BUSDataset(images_dir, masks_dir, imgs, groups, augment=False,
                                   is_breast=True, mask_groups=groups), batch_size=1, shuffle=False)

    t_forward, t_pipeline = [], []
    with torch.no_grad():
        for i, (image, _mask) in enumerate(loader):
            image = image.to(device)

            sync(); t0 = time.perf_counter()
            _ = model(image)
            sync(); t1 = time.perf_counter()

            sync(); t2 = time.perf_counter()
            out = tta_predict(model, image)
            _ = post_process_mask(out[0, 0].cpu().numpy(), min_area=args.min_area)
            sync(); t3 = time.perf_counter()

            if i >= N_WARMUP:
                t_forward.append(t1 - t0)
                t_pipeline.append(t3 - t2)

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    res = {"device": gpu, "torch": torch.__version__, "input": "1x1x256x256, batch size 1",
           "forward_only": summarize("single forward pass", t_forward),
           "tta_plus_postprocess": summarize("TTA (4 passes) + post-processing", t_pipeline)}

    print(f"Device: {gpu}")
    for k in ("forward_only", "tta_plus_postprocess"):
        r = res[k]
        print(f"{r['name']}: {r['mean_ms']:.1f} ± {r['std_ms']:.1f} ms/image "
              f"(median {r['median_ms']:.1f}, min {r['min_ms']:.1f}, max {r['max_ms']:.1f}, n={r['n_images']})")

    with open(os.path.join(args.out_dir, "unetpp_inference_time.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
