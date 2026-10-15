import argparse, json, os
import numpy as np, pandas as pd
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--px", type=int, default=384)
a = ap.parse_args()

clean = pd.read_csv(os.path.join(a.cache, "split_clean.csv"))
idx = json.load(open(os.path.join(a.cache, f"images_{a.px}_index.json")))
row = {p: i for i, p in enumerate(idx["paths"])}
mm = np.load(os.path.join(a.cache, f"images_{a.px}.dat"), mmap_mode="r")
inv = pd.read_csv(os.path.join(a.cache, "inventory.csv"))
sz = dict(zip(inv.fname, zip(inv.w, inv.h)))

rng = np.random.default_rng(7)
TH, PAD = 150, 22
classes = sorted(clean.cls.unique())
canvas = Image.new("RGB", (8 * (TH + 4) + 8, len(classes) * (TH + PAD) + 8), (18, 18, 18))
d = ImageDraw.Draw(canvas)
for ri, c in enumerate(classes):
    sel = clean[clean.cls == c]
    pick = sel.iloc[rng.choice(len(sel), 8, replace=False)]
    y = 8 + ri * (TH + PAD)
    dims = [f"{sz[f][0]}x{sz[f][1]}" for f in pick.fname]
    d.text((8, y), f"{c}  (n={len(sel)})  sizes: {', '.join(dims[:5])}", fill=(235, 235, 235))
    for ci, p in enumerate(pick.path):
        canvas.paste(Image.fromarray(np.asarray(mm[row[p]])).resize((TH, TH)),
                     (8 + ci * (TH + 4), y + 14))
canvas.save(a.out)
print("wrote", a.out)

print("\nimage dimensions per class (raw files):")
m = inv.merge(clean[["fname"]], on="fname")
print(pd.crosstab(m.cls, m.w.astype(str) + "x" + m.h.astype(str)).to_string())
