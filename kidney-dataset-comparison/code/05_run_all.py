"""
Stage 4: run every encoder, one subprocess each so VRAM is fully released
between models (necessary on a 4 GB RTX 3050).
"""
import argparse, os, subprocess, sys, time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # xet backend stalls here
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ENCODERS = ["DenseNet-201", "ViT-Base-16", "VGG19", "InceptionV3", "Xception"]
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="clean")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--cache-px", type=int, default=384)
    ap.add_argument("--center-crop", type=float, default=1.0)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip an encoder whose result JSON is already present")
    ap.add_argument("--encoders", nargs="*", default=ENCODERS)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    failures = []
    for enc in args.encoders:
        print("=" * 78, flush=True)
        print(f"=== {args.tag} | {enc}", flush=True)
        print("=" * 78, flush=True)
        done_fn = os.path.join(args.out_dir, f"{args.tag}__{enc}.json")
        if args.skip_existing and os.path.exists(done_fn):
            print(f"--- {enc} already done, skipping", flush=True)
            continue
        t0 = time.time()
        cmd = [args.python, "-I", os.path.join(HERE, "04_train_one.py"),
               "--encoder", enc, "--split-csv", args.split_csv,
               "--out-dir", args.out_dir, "--tag", args.tag,
               "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
               "--cache", args.cache, "--cache-px", str(args.cache_px),
               "--center-crop", str(args.center_crop)]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"!!! {enc} FAILED (exit {r.returncode})", flush=True)
            failures.append(enc)
        print(f"--- {enc} took {(time.time()-t0)/60:.1f} min\n", flush=True)

    if failures:
        print(f"FAILED encoders: {failures}")
        sys.exit(1)
    print("all encoders done")


if __name__ == "__main__":
    main()
