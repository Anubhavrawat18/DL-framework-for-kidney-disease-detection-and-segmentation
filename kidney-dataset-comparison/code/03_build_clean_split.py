"""
Stage 2b: build a leakage-free split.

Groups are formed by union-find over three independent links:
  1. byte-identical md5            (exact duplicates)
  2. aug_<n>_<base>.jpg -> <base>  (pre-computed augmentations of a source image)
  3. phash hamming <= 6 AND 32x32 thumbnail correlation >= 0.95   (near-duplicate slices)

Then: one representative per group is kept (dedup), groups are split
stratified-by-class, and the split is INDEPENDENTLY re-verified at a looser
threshold (hamming <= 10, corr >= 0.90); any residual cross-split neighbour has
its held-out member dropped.

Also emits a deliberately leaky random split over the raw files, to reproduce the
near-100% accuracy seen with a naive split and quantify the inflation.
"""
import argparse, json, os, re
import numpy as np, pandas as pd

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
CH = 512


def hamming_rows(h, i0, i1):
    x = np.bitwise_xor(h[i0:i1, None], h[None, :])
    b = x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8)
    return POP[b].sum(axis=2)


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb
            return True
        return False


def neardup_pairs(ph, thumb, T, C):
    N = len(ph)
    out = []
    for i0 in range(0, N, CH):
        i1 = min(i0 + CH, N)
        d = hamming_rows(ph, i0, i1)
        cand = np.argwhere(d <= T)
        if not len(cand):
            continue
        gi, gj = cand[:, 0] + i0, cand[:, 1]
        keep = gi < gj
        gi, gj = gi[keep], gj[keep]
        if not len(gi):
            continue
        corr = np.einsum("ij,ij->i", thumb[gi], thumb[gj])
        ok = corr >= C
        out.append(np.stack([gi[ok], gj[ok]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=int)


def stratified_split(frame, seed, tr=0.70, va=0.10):
    rng = np.random.default_rng(seed)
    frame = frame.copy()
    frame["split"] = ""
    for _cls, idx in frame.groupby("cls").groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(tr * n))
        n_va = int(round(va * n))
        frame.loc[idx[:n_tr], "split"] = "train"
        frame.loc[idx[n_tr:n_tr + n_va], "split"] = "val"
        frame.loc[idx[n_tr + n_va:], "split"] = "test"
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ham", type=int, default=6)
    ap.add_argument("--corr", type=float, default=0.95)
    ap.add_argument("--ham-verify", type=int, default=10)
    ap.add_argument("--corr-verify", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-prefix", default="clean")
    ap.add_argument("--skip-leaky", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.cache, "inventory.csv")).reset_index(drop=True)
    fp = np.load(os.path.join(args.cache, "fingerprints.npz"))
    ph, thumb = fp["phash"], fp["thumb"]
    N = len(df)
    uf = UF(N)
    report = {"n_images_raw": N}

    # link 1: exact duplicates
    n1 = 0
    for _k, idx in df.groupby("md5_file").groups.items():
        idx = list(idx)
        for b in idx[1:]:
            n1 += uf.union(int(idx[0]), int(b))

    # link 2: augmentation -> source image
    name_to_i = {fn: int(i) for i, fn in zip(df.index, df.fname)}
    n2 = 0
    for i, fn in zip(df.index, df.fname):
        if fn.startswith("aug_"):
            j = name_to_i.get(re.sub(r"^aug_\d+_", "", fn))
            if j is not None:
                n2 += uf.union(int(i), j)

    # link 3: near-duplicates
    pairs = neardup_pairs(ph, thumb, args.ham, args.corr)
    n3 = sum(uf.union(int(a), int(b)) for a, b in pairs)

    df["group"] = [uf.find(i) for i in range(N)]
    ng = df.group.nunique()
    report.update(merges_exact_dup=n1, merges_aug_source=n2,
                  neardup_pairs=int(len(pairs)), merges_neardup=int(n3), n_groups=int(ng))
    print(f"links: exact={n1}  aug->source={n2}  near-dup={n3} (from {len(pairs)} pairs)")
    print(f"{N} files collapse to {ng} independent groups "
          f"({100 * (1 - ng / N):.1f}% of the dataset is duplicate/near-duplicate material)")

    gsz = df.group.value_counts()
    print(f"group size: mean {gsz.mean():.2f}  max {gsz.max()}  "
          f"singletons {(gsz == 1).sum()}  multi {(gsz > 1).sum()}")

    # cross-class groups = label noise, not merely leakage
    gcls = df.groupby("group").cls.nunique()
    bad = gcls[gcls > 1]
    report["cross_class_groups"] = int(len(bad))
    print(f"groups spanning >1 class (label noise / ambiguous): {len(bad)}")
    if len(bad):
        ex = df[df.group.isin(bad.index[:3])].sort_values("group")[["fname", "cls", "group"]]
        print(ex.to_string(index=False))

    # ---- dedup: one representative per group, preferring a non-augmented original
    df["is_aug"] = df.fname.str.startswith("aug_")
    rep = (df.sort_values(["group", "is_aug", "fname"])
             .groupby("group", as_index=False).first())
    rep = rep[~rep.group.isin(set(bad.index))].reset_index(drop=True)
    print(f"\ndeduplicated to {len(rep)} images (1 per group)")
    print(rep.cls.value_counts().to_string())

    sub = stratified_split(rep, args.seed).reset_index(drop=True)

    # ---- independent verification at a LOOSER threshold ----------------------
    pos = np.array([name_to_i[f] for f in sub.fname])
    vp = neardup_pairs(ph[pos], thumb[pos], args.ham_verify, args.corr_verify)
    sp = sub.split.values
    drop, cross = set(), 0
    for a, b in vp:
        if sp[a] != sp[b]:
            cross += 1
            # drop the held-out side so train never shrinks silently
            drop.add(int(b) if sp[b] != "train" else int(a))
    print(f"\nverification @ ham<={args.ham_verify}, corr>={args.corr_verify}: "
          f"{cross} cross-split neighbour pairs -> dropping {len(drop)} held-out images")
    report.update(verify_cross_split_pairs=int(cross), verify_dropped=int(len(drop)))
    sub = sub.drop(index=sorted(drop)).reset_index(drop=True)

    pos = np.array([name_to_i[f] for f in sub.fname])
    vp2 = neardup_pairs(ph[pos], thumb[pos], args.ham_verify, args.corr_verify)
    sp = sub.split.values
    resid = sum(1 for a, b in vp2 if sp[a] != sp[b])
    print(f"residual cross-split neighbour pairs after cleaning: {resid}")
    report["residual_cross_split_pairs"] = int(resid)

    print("\nFINAL CLEAN SPLIT")
    print(pd.crosstab(sub.cls, sub.split).to_string())
    sub[["path", "fname", "cls", "label", "group", "split"]].to_csv(
        os.path.join(args.out_dir, f"split_{args.out_prefix}.csv"), index=False)

    # ---- the naive split: plain random over all raw files --------------------
    if args.skip_leaky:
        json.dump(report, open(os.path.join(args.out_dir,
                  f"leakage_report_{args.out_prefix}.json"), "w"), indent=2)
        print(f"\nwrote split_{args.out_prefix}.csv (leaky split skipped)")
        return
    leak = stratified_split(df, args.seed)
    gs = leak.groupby("group").split.nunique()
    n_test = int((leak.split == "test").sum())
    contaminated = leak[leak.group.isin(gs[gs > 1].index) & (leak.split == "test")]
    print(f"\nLEAKY (naive random) split: {len(leak)} files, "
          f"{len(contaminated)}/{n_test} test images "
          f"({100 * len(contaminated) / n_test:.1f}%) have a duplicate or "
          f"near-duplicate sitting in train/val")
    report["leaky_test_contaminated_frac"] = float(len(contaminated) / n_test)
    leak[["path", "fname", "cls", "label", "group", "split"]].to_csv(
        os.path.join(args.out_dir, "split_leaky.csv"), index=False)

    json.dump(report, open(os.path.join(args.out_dir, f"leakage_report_{args.out_prefix}.json"), "w"), indent=2)
    print(f"\nwrote split_clean.csv, split_leaky.csv, leakage_report.json -> {args.out_dir}")


if __name__ == "__main__":
    main()
