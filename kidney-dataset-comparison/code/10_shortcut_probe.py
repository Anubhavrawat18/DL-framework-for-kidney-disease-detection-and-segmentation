"""
Stage 7: acquisition-shortcut probes.

Duplicate leakage is not the only problem with this dataset. The three classes
appear to come from different acquisition sources, so a model can score highly
by reading scanner artefacts instead of kidney anatomy. Each probe below
deliberately destroys the anatomy and keeps only an artefact channel; whatever
accuracy survives is accuracy that owes nothing to pathology.

  A  file dimensions only (width, height)   - a 2-feature decision tree
  B  32x32 contrast-normalised thumbnail    - global layout, anatomy unreadable
  C  border ring only (centre 60% blacked)  - scanner chrome / bezel / captions
  D  8x8 thumbnail                          - essentially only framing geometry

All are fitted on the SAME leakage-free clean split used for the encoders, so
the numbers are directly comparable to the deep-model results.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score


def report(name, clf, Xtr, ytr, Xte, yte, names):
    clf.fit(Xtr, ytr)
    p = clf.predict(Xte)
    acc = accuracy_score(yte, p)
    f1 = f1_score(yte, p, average="macro")
    print(f"  {name:<46s} test acc {acc:.4f}   macro-F1 {f1:.4f}")
    return dict(probe=name, test_accuracy=round(float(acc), 4),
                f1_macro=round(float(f1), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--px", type=int, default=384)
    args = ap.parse_args()

    clean = pd.read_csv(os.path.join(args.cache, "split_clean.csv"))
    inv = pd.read_csv(os.path.join(args.cache, "inventory.csv"))
    fp = np.load(os.path.join(args.cache, "fingerprints.npz"))
    idx = json.load(open(os.path.join(args.cache, f"images_{args.px}_index.json")))
    row_of_path = {p: i for i, p in enumerate(idx["paths"])}
    mm = np.load(os.path.join(args.cache, f"images_{args.px}.dat"), mmap_mode="r")

    inv_pos = {f: i for i, f in enumerate(inv.fname)}
    fi = np.array([inv_pos[f] for f in clean.fname])
    tr = (clean.split == "train").values
    te = (clean.split == "test").values
    y = clean.label.values
    names = [clean[clean.label == l].cls.iloc[0] for l in sorted(clean.label.unique())]
    print(f"clean split: {tr.sum()} train / {te.sum()} test   classes {names}\n")

    out = {"classes": names, "n_train": int(tr.sum()), "n_test": int(te.sum())}
    results = []

    # ---- A: dimensions only -------------------------------------------------
    print("PROBE A - file dimensions only (width, height), 2-feature decision tree")
    wh = inv.iloc[fi][["w", "h"]].values.astype(np.float32)
    results.append(report("A. width+height only", DecisionTreeClassifier(max_depth=4, random_state=0),
                          wh[tr], y[tr], wh[te], y[te], names))
    print("     per-class dimension makeup:")
    tmp = pd.DataFrame({"cls": clean.cls.values, "dim": [f"{int(a)}x{int(b)}" for a, b in wh]})
    print(pd.crosstab(tmp.cls, tmp.dim).to_string().replace("\n", "\n     "))

    # ---- B: 32x32 normalised thumbnail -------------------------------------
    print("\nPROBE B - 32x32 contrast-normalised thumbnail, logistic regression")
    th = fp["thumb"][fi]
    results.append(report("B. 32x32 thumbnail", LogisticRegression(max_iter=3000, C=1.0),
                          th[tr], y[tr], th[te], y[te], names))

    # ---- C: border ring only ------------------------------------------------
    print("\nPROBE C - border ring only, centre 60% blacked out (scanner chrome)")
    S = 64
    lo, hi = int(S * 0.20), int(S * 0.80)
    feats = np.empty((len(fi), S * S), dtype=np.float32)
    for k, p in enumerate(clean.path):
        a = np.asarray(mm[row_of_path[p]])
        g = a.astype(np.float32).mean(axis=2)
        g = g[::(args.px // S), ::(args.px // S)][:S, :S]
        g[lo:hi, lo:hi] = 0.0                      # destroy the anatomy
        g = (g - g.mean()) / (g.std() + 1e-6)
        feats[k] = g.flatten()
    results.append(report("C. border ring only (centre removed)",
                          LogisticRegression(max_iter=3000, C=1.0),
                          feats[tr], y[tr], feats[te], y[te], names))

    # ---- D: 8x8 thumbnail ---------------------------------------------------
    print("\nPROBE D - 8x8 thumbnail (framing geometry only)")
    tiny = th.reshape(-1, 32, 32)[:, ::4, ::4].reshape(len(fi), -1)
    results.append(report("D. 8x8 thumbnail", LogisticRegression(max_iter=3000, C=1.0),
                          tiny[tr], y[tr], tiny[te], y[te], names))

    out["probes"] = results
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    print("\nAny accuracy here is won WITHOUT readable kidney anatomy. It is an upper "
          "bound on how much of the encoders' score could come from acquisition\n"
          "artefacts (source resolution, scanner chrome, framing) rather than pathology.")


if __name__ == "__main__":
    main()
