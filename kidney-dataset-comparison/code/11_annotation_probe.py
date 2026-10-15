"""
Stage 8: burned-in annotation probe.

Inspecting the centre-cropped images revealed a third leak, worse than the other
two because cropping does NOT remove it: many `Cyst` frames carry burned-in
radiologist annotations - the literal word "CYST", "Rt Kidney", and caliper
crosshairs with dotted measurement lines - rendered at saturated intensity.

Scanner overlays are drawn at or near pure white with hard edges, unlike
ultrasound speckle. So a mask of saturated pixels isolates the annotation layer
almost perfectly. If a classifier on that mask alone separates the classes, the
label is readable from the overlay rather than from the kidney.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--px", type=int, default=384)
    ap.add_argument("--sat", type=int, default=250)
    args = ap.parse_args()

    clean = pd.read_csv(os.path.join(args.cache, "split_clean.csv"))
    idx = json.load(open(os.path.join(args.cache, f"images_{args.px}_index.json")))
    row_of = {p: i for i, p in enumerate(idx["paths"])}
    mm = np.load(os.path.join(args.cache, f"images_{args.px}.dat"), mmap_mode="r")

    tr = (clean.split == "train").values
    te = (clean.split == "test").values
    y = clean.label.values
    names = [clean[clean.label == l].cls.iloc[0] for l in sorted(clean.label.unique())]

    S = 48
    px = args.px
    masks_full = np.zeros((len(clean), S * S), dtype=np.float32)
    masks_crop = np.zeros((len(clean), S * S), dtype=np.float32)
    counts = np.zeros(len(clean), dtype=np.float32)
    lo = int(px * (1 - 0.55) / 2)

    for k, p in enumerate(clean.path):
        g = np.asarray(mm[row_of[p]]).mean(axis=2)
        sat = (g >= args.sat).astype(np.float32)
        counts[k] = sat.sum()
        step = px // S
        masks_full[k] = sat[::step, ::step][:S, :S].flatten()
        c = sat[lo:px - lo, lo:px - lo]
        cs = max(1, c.shape[0] // S)
        cc = c[::cs, ::cs][:S, :S]
        if cc.shape != (S, S):
            cc = np.pad(cc, ((0, S - cc.shape[0]), (0, S - cc.shape[1])))
        masks_crop[k] = cc.flatten()

    out = {"classes": names, "saturation_threshold": args.sat, "probes": []}

    print(f"saturated-pixel count per class (intensity >= {args.sat}, out of {px*px}):")
    df = pd.DataFrame({"cls": clean.cls.values, "n_sat": counts})
    print(df.groupby("cls").n_sat.describe()[["mean", "50%", "max"]].round(1).to_string())
    frac_any = df.assign(has=df.n_sat > 50).groupby("cls").has.mean()
    print(f"\nfraction of images with >50 saturated pixels (i.e. a visible overlay):")
    print(frac_any.round(3).to_string())
    out["frac_with_overlay"] = {k: round(float(v), 4) for k, v in frac_any.items()}

    def run(name, X):
        clf = (DecisionTreeClassifier(max_depth=4, random_state=0)
               if X.shape[1] == 1 else LogisticRegression(max_iter=3000))
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        a, f = accuracy_score(y[te], p), f1_score(y[te], p, average="macro")
        print(f"  {name:<52s} test acc {a:.4f}   macro-F1 {f:.4f}")
        out["probes"].append(dict(probe=name, test_accuracy=round(float(a), 4),
                                  f1_macro=round(float(f), 4)))

    print("\nprobes using ONLY the saturated-pixel (annotation) layer:")
    run("E. saturated-pixel COUNT only (1 feature)", counts.reshape(-1, 1))
    run("F. saturated-pixel mask, full frame", masks_full)
    run("G. saturated-pixel mask, 0.55 centre crop", masks_crop)

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    print("\nThese probes see NO anatomy at all - only where the scanner burned pixels\n"
          "to pure white. Accuracy here is label information printed into the image.")


if __name__ == "__main__":
    main()
