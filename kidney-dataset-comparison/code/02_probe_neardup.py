"""Stage 2a: probe near-duplicate thresholds so the grouping rule is data-driven,
not guessed. Reports, for a grid of (phash hamming, thumbnail correlation),
how many pairs link and how large the resulting connected components get."""
import argparse, os
import numpy as np, pandas as pd

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def hamming_rows(h, i0, i1):
    """hamming distance of h[i0:i1] against all of h -> (chunk, N) uint8"""
    x = np.bitwise_xor(h[i0:i1, None], h[None, :])
    b = x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8)
    return POP[b].sum(axis=2)

class UF:
    def __init__(s, n): s.p = list(range(n))
    def find(s, x):
        while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def union(s, a, b):
        ra, rb = s.find(a), s.find(b)
        if ra != rb: s.p[ra] = rb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(args.cache, "inventory.csv"))
    fp = np.load(os.path.join(args.cache, "fingerprints.npz"))
    ph, thumb = fp["phash"], fp["thumb"]
    N = len(df)
    print(f"{N} images\n")

    # nearest-neighbour profile: for each image, its best non-self phash distance
    best = np.full(N, 64, dtype=np.int16)
    CH = 512
    for i0 in range(0, N, CH):
        i1 = min(i0 + CH, N)
        d = hamming_rows(ph, i0, i1).astype(np.int16)
        for r in range(i1 - i0):
            d[r, i0 + r] = 64
        best[i0:i1] = d.min(axis=1)
    print("distribution of each image's nearest phash distance:")
    for q in [0, 1, 5, 10, 25, 50, 75, 90, 100]:
        print(f"  p{q:3d}: {np.percentile(best, q):.0f}")
    print(f"  images with an exact phash twin (d=0): {(best==0).sum()}")
    print(f"  images with d<=4: {(best<=4).sum()}   d<=6: {(best<=6).sum()}   d<=8: {(best<=8).sum()}")

    print("\n{:>6} {:>6} {:>10} {:>8} {:>10} {:>10}".format(
        "ham<=", "corr>=", "pairs", "groups", "largest", "in_groups>1"))
    for T in [4, 6, 8, 10]:
        for C in [0.90, 0.95, 0.98]:
            uf, pairs = UF(N), 0
            for i0 in range(0, N, CH):
                i1 = min(i0 + CH, N)
                d = hamming_rows(ph, i0, i1)
                cand = np.argwhere(d <= T)
                if len(cand) == 0: continue
                gi = cand[:, 0] + i0; gj = cand[:, 1]
                keep = gi < gj
                gi, gj = gi[keep], gj[keep]
                if len(gi) == 0: continue
                corr = np.einsum("ij,ij->i", thumb[gi], thumb[gj])
                ok = corr >= C
                gi, gj = gi[ok], gj[ok]
                pairs += len(gi)
                for a, b in zip(gi, gj): uf.union(int(a), int(b))
            roots = np.array([uf.find(i) for i in range(N)])
            _, counts = np.unique(roots, return_counts=True)
            print("{:>6} {:>6.2f} {:>10d} {:>8d} {:>10d} {:>10d}".format(
                T, C, pairs, len(counts), counts.max(), int(counts[counts > 1].sum())))

if __name__ == "__main__":
    main()
