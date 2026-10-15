"""
Stage 10b: build the cross-dataset experiment.

TRAIN on URI-CADS, TEST on the Kidney Dataset. This is the one experiment that
the acquisition-source confound cannot help with: the model never sees a Kidney
Dataset scanner during training, so it cannot use that dataset's source
signature to separate its classes.

Label space: binary, normal vs pathological.
  URI-CADS       global_diagnosis   0 = healthy, 1 = pathological  (confirmed in
                                    the project's own Phase0 notebook)
  Kidney Dataset Normal -> 0,  Cyst and stone -> 1

Why binary and not Cyst/Normal/stone: the URI-CADS pathology-ID -> class-name
mapping is recorded as UNCONFIRMED in Phase0 ("pathology_id is currently an
opaque integer"). Inspecting the annotated lesion crops identifies id 3 as
simple cysts and id 7 as hydronephrosis, but "stone" is ambiguous across the
hyperechoic ids (1, 5, 9). Guessing would fabricate labels, so the comparison
is run on the only label space both datasets confirmably share.

The Kidney Dataset side uses the deduplicated representatives (one image per
duplicate/near-duplicate group), so duplicates cannot inflate the test metrics.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
PX = 384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bbdd", required=True)
    ap.add_argument("--kidney-split", required=True, help="split_clean.csv (deduplicated)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--px", type=int, default=PX)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- URI-CADS side -------------------------------------------------------
    m = pd.read_pickle(args.manifest)
    img_dir = os.path.join(args.bbdd, "images")
    m["path"] = [os.path.join(img_dir, f"{c}.jpg") for c in m.case_id]
    missing = (~m.path.map(os.path.exists)).sum()
    if missing:
        print(f"WARNING: {missing} URI-CADS images missing on disk, dropping")
        m = m[m.path.map(os.path.exists)]
    m["label"] = m.global_diagnosis.astype(int)
    m["cls"] = m.label.map({0: "normal", 1: "pathological"})
    uri = m[["path", "case_id", "cls", "label", "split"]].rename(columns={"case_id": "fname"})
    uri["fname"] = uri.fname.astype(str)
    print("URI-CADS (train domain):")
    print(pd.crosstab(uri.cls, uri.split).to_string())

    # ---- Kidney Dataset side (deduplicated) ---------------------------------
    k = pd.read_csv(args.kidney_split)
    k["label"] = (k.cls != "Normal").astype(int)       # Cyst + stone -> pathological
    k["orig_cls"] = k.cls
    k["cls"] = k.label.map({0: "normal", 1: "pathological"})
    k["split"] = "ext_test"
    kid = k[["path", "fname", "cls", "label", "split", "orig_cls"]]
    print("\nKidney Dataset (external test domain, deduplicated):")
    print(kid.groupby(["orig_cls", "cls"]).size().to_string())
    print(f"total external test images: {len(kid)}")

    uri["orig_cls"] = uri.cls
    both = pd.concat([uri, kid], ignore_index=True)
    both.to_csv(os.path.join(args.out_dir, "split_cross.csv"), index=False)

    # ---- decoded-image cache for everything ---------------------------------
    paths = both.path.tolist()
    mm_path = os.path.join(args.out_dir, f"cross_{args.px}.dat")
    idx_path = os.path.join(args.out_dir, f"cross_{args.px}_index.json")
    if os.path.exists(idx_path) and json.load(open(idx_path)).get("paths") == paths:
        print(f"\ncache already built: {mm_path}")
        return
    print(f"\ndecoding {len(paths)} images -> {args.px}px cache "
          f"({len(paths)*args.px*args.px*3/1e9:.2f} GB)")
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
