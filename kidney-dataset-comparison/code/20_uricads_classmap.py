"""
Stage 10a: try to establish the URI-CADS pathology-ID -> class-name mapping.

The project's own Phase0 notebook records this mapping as unconfirmed
("pathology_id is currently an opaque integer"), and the manifest only confirms
global_diagnosis (0 = healthy, 1 = pathological). Without the mapping there is
no shared 3-class label space between URI-CADS and the Kidney Dataset.

Cyst and stone are visually distinctive in B-mode ultrasound:
  cyst  - anechoic (near-black) round lesion, sharp margin, posterior enhancement
  stone - small hyperechoic (near-white) focus, often with posterior shadowing

So for each pathology id this renders a montage of its annotated lesion crops and
prints simple intensity statistics inside the box relative to the surrounding
kidney. A class whose crops are much DARKER than background is cyst-like; one
much BRIGHTER is stone/calcification-like.

This is evidence for a mapping, not proof of one. Nothing downstream should
assume the mapping without a human confirming these montages.
"""
import argparse, os, re
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

BOX = re.compile(r"Bounding box \d+:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")


def load_boxes(label_dir):
    rows = []
    for fn in sorted(os.listdir(label_dir)):
        stem = os.path.splitext(fn)[0]
        for ln in open(os.path.join(label_dir, fn)):
            m = BOX.match(ln.strip())
            if m:
                x, y, w, h, pid = (int(v) for v in m.groups())
                rows.append(dict(case=stem, x=x, y=y, w=w, h=h, pid=pid))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbdd", required=True, help="bbdd_kidney dir with images/ and labels/")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-class", type=int, default=10)
    args = ap.parse_args()

    img_dir = os.path.join(args.bbdd, "images")
    df = load_boxes(os.path.join(args.bbdd, "labels"))
    print(f"{len(df)} lesion boxes over {df.case.nunique()} cases")
    print(df.pid.value_counts().sort_index().to_string())

    stats, rng = [], np.random.default_rng(0)
    TH, PAD = 110, 24
    pids = sorted(df.pid.unique())
    canvas = Image.new("RGB", (args.per_class * (TH + 4) + 8,
                               len(pids) * (TH + PAD) + 8), (18, 18, 18))
    d = ImageDraw.Draw(canvas)

    for ri, pid in enumerate(pids):
        sel = df[df.pid == pid]
        take = sel.iloc[rng.choice(len(sel), min(args.per_class, len(sel)), replace=False)]
        y0 = 8 + ri * (TH + PAD)
        ratios = []
        crops = []
        for _, r in take.iterrows():
            fp = os.path.join(img_dir, f"{r.case}.jpg")
            if not os.path.exists(fp):
                continue
            im = Image.open(fp).convert("L")
            a = np.asarray(im, dtype=np.float32)
            x1, y1 = max(0, r.x), max(0, r.y)
            x2, y2 = min(a.shape[1], r.x + r.w), min(a.shape[0], r.y + r.h)
            if x2 <= x1 or y2 <= y1:
                continue
            inside = a[y1:y2, x1:x2]
            # surrounding ring, 2x the box, excluding the box itself
            mx, my = (x2 - x1) // 2, (y2 - y1) // 2
            ox1, oy1 = max(0, x1 - mx), max(0, y1 - my)
            ox2, oy2 = min(a.shape[1], x2 + mx), min(a.shape[0], y2 + my)
            outer = a[oy1:oy2, ox1:ox2].copy()
            oyy, oxx = y1 - oy1, x1 - ox1
            mask = np.ones_like(outer, dtype=bool)
            mask[oyy:oyy + (y2 - y1), oxx:oxx + (x2 - x1)] = False
            ring = outer[mask]
            if ring.size > 20:
                ratios.append(float(inside.mean() / (ring.mean() + 1e-6)))
            crops.append(np.asarray(Image.fromarray(inside.astype(np.uint8))
                                    .resize((TH, TH), Image.BILINEAR)))

        lab = (f"pathology id {pid}  (n={len(sel)} boxes)  "
               f"mean inside/surround brightness = {np.mean(ratios):.2f}" if ratios else
               f"pathology id {pid}  (n={len(sel)})")
        d.text((8, y0), lab, fill=(235, 235, 235))
        for ci, c in enumerate(crops):
            canvas.paste(Image.fromarray(c).convert("RGB"), (8 + ci * (TH + 4), y0 + 14))
        stats.append(dict(pid=pid, n_boxes=int(len(sel)),
                          n_images=int(sel.case.nunique()),
                          brightness_ratio=round(float(np.mean(ratios)), 3) if ratios else None))

    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, "uricads_pathology_ids.png")
    canvas.save(out_png)
    s = pd.DataFrame(stats)
    s.to_csv(os.path.join(args.out_dir, "uricads_pathology_ids.csv"), index=False)
    print(f"\nwrote {out_png}")
    print("\nbrightness of lesion vs its surroundings (<1 = darker/anechoic = cyst-like,")
    print(">1 = brighter/hyperechoic = stone-like):")
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
