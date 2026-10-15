"""
Stage 1: Leakage audit for the Kidney Dataset.

Fingerprints every image several independent ways so that leakage can be found
even when filenames hide it:

  md5_file   - md5 of the raw bytes            -> byte-identical duplicates
  md5_pixel  - md5 of the decoded grayscale px -> re-encoded / recompressed duplicates
  dhash      - 64-bit difference hash          -> near-duplicates (slice neighbours)
  phash      - 64-bit DCT perceptual hash      -> near-duplicates, robust to contrast
  thumb      - 32x32 normalised thumbnail      -> precision check on near-dup pairs

Writes kidney-dataset-comparison/cache/fingerprints.npz + inventory.csv
"""
import argparse, hashlib, os, sys
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

Image.MAX_IMAGE_PIXELS = None
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def dhash(gray32):
    """64-bit difference hash from a 9x8 resize."""
    small = np.asarray(Image.fromarray(gray32).resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.packbits(bits).view(np.uint64)[0]


def phash(gray32):
    """64-bit DCT perceptual hash from a 32x32 resize, low-frequency 8x8 block."""
    small = np.asarray(Image.fromarray(gray32).resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = dctn(small, norm="ortho")[:8, :8]
    flat = d.flatten()
    med = np.median(flat[1:])          # ignore DC term when thresholding
    bits = (flat > med)
    return np.packbits(bits).view(np.uint64)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root, out = args.data_root, args.out_dir
    os.makedirs(out, exist_ok=True)

    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    print(f"classes found: {classes}")

    rows, dh, ph, thumbs = [], [], [], []
    for ci, cls in enumerate(classes):
        cdir = os.path.join(root, cls)
        files = sorted(f for f in os.listdir(cdir) if os.path.splitext(f)[1].lower() in EXTS)
        print(f"  {cls:8s} {len(files):5d} files", flush=True)
        for fn in files:
            fp = os.path.join(cdir, fn)
            raw = open(fp, "rb").read()
            try:
                im = Image.open(fp)
                im.load()
            except Exception as e:
                print(f"    !! unreadable {fn}: {e}")
                continue
            w, h = im.size
            g = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.uint8)
            rows.append(dict(
                path=fp, fname=fn, cls=cls, label=ci, w=w, h=h, mode=im.mode,
                md5_file=hashlib.md5(raw).hexdigest(),
                md5_pixel=hashlib.md5(g.tobytes()).hexdigest(),
            ))
            dh.append(dhash(g)); ph.append(phash(g))
            t = g.astype(np.float32).flatten()
            t = (t - t.mean()) / (t.std() + 1e-6)          # contrast/brightness invariant
            thumbs.append(t / np.linalg.norm(t))           # unit norm -> dot product = correlation

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "inventory.csv"), index=False)
    np.savez_compressed(
        os.path.join(out, "fingerprints.npz"),
        dhash=np.asarray(dh, dtype=np.uint64),
        phash=np.asarray(ph, dtype=np.uint64),
        thumb=np.asarray(thumbs, dtype=np.float32),
    )
    print(f"\ninventory: {len(df)} images -> {out}")

    # ---- quick leakage report -------------------------------------------------
    print("\n" + "=" * 70)
    print("LEAKAGE AUDIT")
    print("=" * 70)
    print("\nper-class counts:")
    print(df.cls.value_counts().to_string())

    for key, desc in [("md5_file", "byte-identical"), ("md5_pixel", "pixel-identical (32x32)")]:
        vc = df[key].value_counts()
        dupgroups = vc[vc > 1]
        print(f"\n{desc} duplicate groups: {len(dupgroups)}  "
              f"covering {int(dupgroups.sum())} files  "
              f"({int(dupgroups.sum() - len(dupgroups))} redundant copies)")
        if len(dupgroups):
            cross = 0
            for k in dupgroups.index:
                if df.loc[df[key] == k, "cls"].nunique() > 1:
                    cross += 1
            print(f"  of which span MORE THAN ONE CLASS (label noise): {cross}")

    aug = df[df.fname.str.startswith("aug_")]
    print(f"\nfiles named aug_*: {len(aug)}")
    if len(aug):
        base = aug.fname.str.replace(r"^aug_\d+_", "", regex=True)
        originals = set(df[~df.fname.str.startswith("aug_")].fname)
        present = base.isin(originals).sum()
        print(f"  whose source image is ALSO in the dataset: {present} / {len(aug)}")
        print(f"  distinct source images augmented: {base.nunique()}")

    print(f"\nimage size variety: {df.groupby(['w','h']).size().sort_values(ascending=False).head(8).to_string()}")


if __name__ == "__main__":
    main()
