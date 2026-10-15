"""
Stage 11a: build the 3-class cross-dataset experiment.

TRAIN on URI-CADS, TEST on the Kidney Dataset, with the class mapping the user
confirmed:
    normal  <- URI-CADS global_diagnosis == 0
    cyst    <- image contains a lesion box with pathology id 1
    stone   <- image contains a lesion box with pathology id 4
               (a grouped category that contains stones)

Images carrying both id 1 and id 4 (4 of them) are dropped, since the task is
single-label. Images whose only findings are other pathology ids (2, 3, 5-9) are
not part of any target class and are excluded from training.

Note: an image counted as cyst or stone may also carry other pathology ids - the
target finding is present, but it is not necessarily the only one. That is
label noise this experiment accepts in order to keep the classes large enough.

The Kidney Dataset side uses the deduplicated representatives, so duplicates
cannot inflate the external metrics.
"""
import argparse, json, os, re
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
BOX = re.compile(r"Bounding box \d+:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")
CLASSES = ["normal", "cyst", "stone"]           # label 0, 1, 2
CYST_PID, STONE_PID = 1, 4


def lesion_ids(label_dir):
    out = {}
    for fn in os.listdir(label_dir):
        s = set()
        for ln in open(os.path.join(label_dir, fn)):
            m = BOX.match(ln.strip())
            if m:
                s.add(int(m.group(5)))
        out[os.path.splitext(fn)[0]] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bbdd", required=True)
    ap.add_argument("--kidney-split", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--px", type=int, default=384)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ids = lesion_ids(os.path.join(args.bbdd, "labels"))
    m = pd.read_pickle(args.manifest)
    m["case_id"] = m.case_id.astype(str)
    m["path"] = [os.path.join(args.bbdd, "images", f"{c}.jpg") for c in m.case_id]
    m = m[m.path.map(os.path.exists)].copy()
    m["pids"] = m.case_id.map(lambda c: ids.get(c, set()))
    has_c = m.pids.map(lambda s: CYST_PID in s)
    has_s = m.pids.map(lambda s: STONE_PID in s)

    m["cls"] = None
    m.loc[m.global_diagnosis == 0, "cls"] = "normal"
    m.loc[has_c & ~has_s, "cls"] = "cyst"
    m.loc[has_s & ~has_c, "cls"] = "stone"
    dropped_both = int((has_c & has_s).sum())
    uri = m[m.cls.notna()].copy()
    uri["label"] = uri.cls.map({c: i for i, c in enumerate(CLASSES)})
    uri = uri[["path", "case_id", "cls", "label", "split"]].rename(columns={"case_id": "fname"})
    print(f"URI-CADS (train domain): dropped {dropped_both} images carrying both "
          f"pid {CYST_PID} and pid {STONE_PID}")
    print(pd.crosstab(uri.cls, uri.split).to_string())
    print(f"excluded (other-pathology-only cases): {len(m) - len(uri) - dropped_both}")

    k = pd.read_csv(args.kidney_split)
    k["cls"] = k.cls.str.lower().replace({"normal": "normal", "cyst": "cyst", "stone": "stone"})
    k["label"] = k.cls.map({c: i for i, c in enumerate(CLASSES)})
    assert k.label.notna().all(), "unmapped Kidney Dataset class"
    k["split"] = "ext_test"
    kid = k[["path", "fname", "cls", "label", "split"]]
    print("\nKidney Dataset (external test, deduplicated):")
    print(kid.cls.value_counts().to_string())
    print(f"total: {len(kid)}")

    both = pd.concat([uri, kid], ignore_index=True)
    both["label"] = both.label.astype(int)
    both.to_csv(os.path.join(args.out_dir, "split_cross3.csv"), index=False)

    paths = both.path.tolist()
    mm_path = os.path.join(args.out_dir, f"cross3_{args.px}.dat")
    idx_path = os.path.join(args.out_dir, f"cross3_{args.px}_index.json")
    if os.path.exists(idx_path) and json.load(open(idx_path)).get("paths") == paths:
        print(f"\ncache already built: {mm_path}")
        return
    print(f"\ndecoding {len(paths)} images -> {args.px}px cache")
    mmv = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.uint8,
                                    shape=(len(paths), args.px, args.px, 3))
    for i, p in enumerate(paths):
        mmv[i] = np.asarray(Image.open(p).convert("RGB")
                            .resize((args.px, args.px), Image.BICUBIC), dtype=np.uint8)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(paths)}", flush=True)
    mmv.flush(); del mmv
    json.dump({"px": args.px, "paths": paths}, open(idx_path, "w"))
    print(f"wrote {mm_path}")


if __name__ == "__main__":
    main()
