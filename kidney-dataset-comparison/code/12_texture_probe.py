"""
Stage 9: texture / source-signature probe.

Centre-cropping collapsed the LINEAR pixel shortcut (0.93 -> 0.61) yet the deep
encoders barely moved (DenseNet 0.9961 -> 0.9922). So the models are not reading
the chrome; they are reading something the crop keeps and a downsampled linear
probe cannot see. The obvious candidate is scanner texture: speckle statistics,
gain, gamma and native resolution differ between acquisition sources, and each
class here comes from a different source.

These probes keep ONLY texture statistics and destroy anatomy entirely:

  H  radial FFT power spectrum        - speckle frequency signature, no geometry
  I  intensity histogram              - gain / gamma signature, no geometry
  J  H + I combined
  K  patch-shuffled image             - 16x16 tiles randomly permuted inside the
                                        centre crop: local texture intact,
                                        anatomy destroyed

If these score far above chance, label and acquisition source are confounded and
no amount of cropping can separate them.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--px", type=int, default=384)
    ap.add_argument("--crop", type=float, default=0.55)
    args = ap.parse_args()

    clean = pd.read_csv(os.path.join(args.cache, "split_clean.csv"))
    idx = json.load(open(os.path.join(args.cache, f"images_{args.px}_index.json")))
    row_of = {p: i for i, p in enumerate(idx["paths"])}
    mm = np.load(os.path.join(args.cache, f"images_{args.px}.dat"), mmap_mode="r")

    tr = (clean.split == "train").values
    te = (clean.split == "test").values
    y = clean.label.values
    names = [clean[clean.label == l].cls.iloc[0] for l in sorted(clean.label.unique())]
    px = args.px
    lo = int(px * (1 - args.crop) / 2)
    n = len(clean)

    NB = 48                      # radial frequency bins
    spec = np.zeros((n, NB), dtype=np.float32)
    hist = np.zeros((n, 64), dtype=np.float32)
    shuf = np.zeros((n, 48 * 48), dtype=np.float32)

    side = px - 2 * lo
    yy, xx = np.mgrid[0:side, 0:side]
    cy = cx = side / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rbin = np.clip((r / r.max() * (NB - 1)).astype(int), 0, NB - 1)
    flat_rbin = rbin.ravel()
    counts = np.bincount(flat_rbin, minlength=NB).astype(np.float32)

    rng = np.random.default_rng(0)
    T = 16
    ntile = side // T
    perm = rng.permutation(ntile * ntile)

    for k, p in enumerate(clean.path):
        g = np.asarray(mm[row_of[p]]).mean(axis=2)[lo:px - lo, lo:px - lo]

        # H: radial power spectrum, normalised so overall brightness cancels
        F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
        ps = np.bincount(flat_rbin, weights=F.ravel(), minlength=NB) / counts
        ps = np.log1p(ps)
        spec[k] = (ps - ps.mean()) / (ps.std() + 1e-6)

        # I: intensity histogram
        h = np.histogram(g, bins=64, range=(0, 255), density=True)[0]
        hist[k] = h

        # K: patch-shuffled - anatomy destroyed, local texture intact
        tiles = (g[:ntile * T, :ntile * T]
                 .reshape(ntile, T, ntile, T).transpose(0, 2, 1, 3)
                 .reshape(-1, T, T))[perm]
        sh = (tiles.reshape(ntile, ntile, T, T).transpose(0, 2, 1, 3)
              .reshape(ntile * T, ntile * T))
        s = sh[::max(1, sh.shape[0] // 48), ::max(1, sh.shape[0] // 48)][:48, :48]
        if s.shape != (48, 48):
            s = np.pad(s, ((0, 48 - s.shape[0]), (0, 48 - s.shape[1])))
        s = (s - s.mean()) / (s.std() + 1e-6)
        shuf[k] = s.flatten()

    out = {"classes": names, "crop": args.crop, "probes": []}

    def run(name, X):
        clf = LogisticRegression(max_iter=4000)
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        a, f = accuracy_score(y[te], p), f1_score(y[te], p, average="macro")
        print(f"  {name:<56s} test acc {a:.4f}   macro-F1 {f:.4f}")
        out["probes"].append(dict(probe=name, test_accuracy=round(float(a), 4),
                                  f1_macro=round(float(f), 4)))

    base = max(np.bincount(y[te])) / te.sum()
    print(f"majority-class baseline: {base:.4f}")
    print(f"\nprobes on the {args.crop} centre crop, anatomy destroyed, texture kept:")
    run("H. radial FFT power spectrum (speckle signature)", spec)
    run("I. intensity histogram (gain/gamma signature)", hist)
    run("J. spectrum + histogram", np.hstack([spec, hist]))
    run("K. patch-shuffled image (16x16 tiles permuted)", shuf)

    out["majority_baseline"] = round(float(base), 4)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
