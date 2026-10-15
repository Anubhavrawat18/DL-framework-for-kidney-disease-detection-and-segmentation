"""
Stage 2c: decode every image once into a uint8 memmap at CACHE_PX x CACHE_PX RGB.

Repeatedly decoding 512x512 JPEGs was starving the GPU (0% utilisation while a
dozen dataloader workers burned CPU, each re-importing torch for ~750 MB of RAM).
With a memmap cache the loader needs no worker processes at all.

384 px leaves headroom for the 299 px models' resize+centre-crop eval transform.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
CACHE_PX = 384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--px", type=int, default=CACHE_PX)
    args = ap.parse_args()

    inv = pd.read_csv(os.path.join(args.cache, "inventory.csv"))
    paths = inv.path.tolist()
    n, px = len(paths), args.px
    mm_path = os.path.join(args.cache, f"images_{px}.dat")
    idx_path = os.path.join(args.cache, f"images_{px}_index.json")

    if os.path.exists(mm_path) and os.path.exists(idx_path):
        existing = json.load(open(idx_path))
        if existing.get("paths") == paths and existing.get("px") == px:
            print(f"cache already built: {mm_path} ({os.path.getsize(mm_path)/1e9:.2f} GB)")
            return

    print(f"decoding {n} images -> {px}x{px} uint8 RGB memmap "
          f"({n*px*px*3/1e9:.2f} GB) at {mm_path}")
    mm = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.uint8, shape=(n, px, px, 3))
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((px, px), Image.BICUBIC)
        mm[i] = np.asarray(im, dtype=np.uint8)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}", flush=True)
    mm.flush()
    del mm
    json.dump({"px": px, "paths": paths}, open(idx_path, "w"))
    print(f"done: {mm_path} ({os.path.getsize(mm_path)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
