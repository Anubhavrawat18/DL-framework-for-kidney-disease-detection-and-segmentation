"""
Stage 0: pre-download every pretrained checkpoint, with timeouts and retries.

A HuggingFace download stalled mid-file during the first full run (byte count
frozen for minutes with no error), which blocked the whole pipeline behind it.
Fetching the weights up front - with a hard socket timeout so a stall raises
instead of hanging, and resume-on-retry - means a training run never waits on
the network.
"""
import os, sys, time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # xet backend stalls here
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")   # stall -> raise, don't hang
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import timm  # noqa: E402

ENCODERS = {
    "DenseNet-201":  "densenet201",
    "ViT-Base-16":   "vit_base_patch16_224",
    "VGG19":         "vgg19",
    "InceptionV3":   "inception_v3",
    "Xception":      "xception",
}
MAX_TRIES = 6


def main():
    failed = []
    for name, model in ENCODERS.items():
        for attempt in range(1, MAX_TRIES + 1):
            t0 = time.time()
            try:
                timm.create_model(model, pretrained=True, num_classes=3)
                print(f"OK   {name:<14s} ({model}) in {time.time()-t0:.0f}s", flush=True)
                break
            except Exception as e:
                print(f"RETRY {attempt}/{MAX_TRIES} {name}: {type(e).__name__}: "
                      f"{str(e)[:140]}", flush=True)
                time.sleep(min(5 * attempt, 30))
        else:
            print(f"FAIL {name} after {MAX_TRIES} attempts", flush=True)
            failed.append(name)

    if failed:
        print(f"\nCOULD NOT FETCH: {failed}")
        sys.exit(1)
    print("\nall checkpoints cached")


if __name__ == "__main__":
    main()
