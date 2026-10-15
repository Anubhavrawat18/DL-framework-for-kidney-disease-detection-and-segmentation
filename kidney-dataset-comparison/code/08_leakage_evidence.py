"""
Stage 6: visual evidence that the grouping is right.

Builds two contact sheets from the cached images so the grouping rule can be
eyeballed rather than taken on trust:

  evidence_groups.png   - rows are duplicate/near-duplicate groups the splitter
                          merged. If these rows do NOT look like the same image
                          repeated, the rule over-merges.
  evidence_distinct.png - rows pair each group's representative with its NEAREST
                          image from a DIFFERENT group. If these pairs DO look
                          identical, the rule under-merges and leakage remains.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
TH = 128   # thumbnail size in the contact sheet
PAD = 26


def sheet(rows, titles, out, px=TH):
    """rows: list of lists of HxWx3 uint8 arrays."""
    ncol = max(len(r) for r in rows)
    W = ncol * (px + 4) + 8
    H = len(rows) * (px + PAD) + 8
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    for ri, (row, title) in enumerate(zip(rows, titles)):
        y = 8 + ri * (px + PAD)
        d.text((8, y), title, fill=(235, 235, 235))
        for ci, arr in enumerate(row):
            im = Image.fromarray(arr).resize((px, px), Image.BILINEAR)
            canvas.paste(im, (8 + ci * (px + 4), y + 14))
    canvas.save(out)
    print(f"wrote {out}  ({W}x{H})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-px", type=int, default=384)
    ap.add_argument("--n-rows", type=int, default=8)
    args = ap.parse_args()

    inv = pd.read_csv(os.path.join(args.cache, "inventory.csv"))
    fp = np.load(os.path.join(args.cache, "fingerprints.npz"))
    ph, thumb = fp["phash"], fp["thumb"]
    idx = json.load(open(os.path.join(args.cache, f"images_{args.cache_px}_index.json")))
    row_of_path = {p: i for i, p in enumerate(idx["paths"])}
    mm = np.load(os.path.join(args.cache, f"images_{args.cache_px}.dat"), mmap_mode="r")

    clean = pd.read_csv(os.path.join(args.cache, "split_clean.csv"))
    gmap = dict(zip(clean.fname, clean.group))

    # rebuild full grouping from the saved leaky split (it carries every file's group)
    leaky = pd.read_csv(os.path.join(args.cache, "split_leaky.csv"))
    sizes = leaky.group.value_counts()
    big = [g for g in sizes.index if 3 <= sizes[g] <= 6][: args.n_rows]

    # ---- sheet 1: what was merged ------------------------------------------
    rows, titles = [], []
    for g in big:
        sel = leaky[leaky.group == g]
        imgs = [np.asarray(mm[row_of_path[p]]) for p in sel.path.tolist()[:6]]
        rows.append(imgs)
        cls = sel.cls.iloc[0]
        titles.append(f"MERGED group {g} ({cls}, n={len(sel)}): "
                      + ", ".join(sel.fname.tolist()[:3])[:88])
    if rows:
        sheet(rows, titles, os.path.join(args.out_dir, "evidence_groups.png"))

    # ---- sheet 2: nearest image from a DIFFERENT group ----------------------
    pos = np.array([row_of_path[p] for p in clean.path])
    inv_pos = {f: i for i, f in enumerate(inv.fname)}
    ci = np.array([inv_pos[f] for f in clean.fname])
    sub_ph, sub_th = ph[ci], thumb[ci]
    n = len(ci)
    rows2, titles2 = [], []
    # score every cross-group pair, keep the most similar ones
    best = []
    for i0 in range(0, n, 512):
        i1 = min(i0 + 512, n)
        x = np.bitwise_xor(sub_ph[i0:i1, None], sub_ph[None, :])
        d = POP[x.view(np.uint8).reshape(i1 - i0, n, 8)].sum(axis=2)
        corr = sub_th[i0:i1] @ sub_th.T
        for r in range(i1 - i0):
            gi = i0 + r
            corr[r, gi] = -1
            order = np.argsort(-corr[r])[:3]
            for j in order:
                if clean.group.iloc[gi] != clean.group.iloc[j]:
                    best.append((float(corr[r, j]), int(d[r, j]), gi, int(j)))
                    break
    best.sort(reverse=True)
    for c, dd, i, j in best[: args.n_rows]:
        rows2.append([np.asarray(mm[pos[i]]), np.asarray(mm[pos[j]])])
        titles2.append(f"KEPT APART  corr={c:.3f} ham={dd}  "
                       f"{clean.fname.iloc[i][:34]} [{clean.split.iloc[i]}]  vs  "
                       f"{clean.fname.iloc[j][:34]} [{clean.split.iloc[j]}]")
    if rows2:
        sheet(rows2, titles2, os.path.join(args.out_dir, "evidence_distinct.png"), px=180)
        print("\nmost-similar surviving cross-group pairs (should look clearly different):")
        for c, dd, i, j in best[: args.n_rows]:
            print(f"  corr={c:.3f} ham={dd:2d}  {clean.fname.iloc[i]} [{clean.split.iloc[i]}]"
                  f"  vs  {clean.fname.iloc[j]} [{clean.split.iloc[j]}]")


if __name__ == "__main__":
    main()
