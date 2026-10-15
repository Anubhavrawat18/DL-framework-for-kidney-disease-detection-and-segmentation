"""
Stage 3: fine-tune ONE encoder on a given split and dump metrics as JSON.

Run as a separate process per encoder so the 4 GB of VRAM on the RTX 3050 is
fully reclaimed between models.

Identical recipe for every encoder (no per-model tuning, so the comparison is
about the backbone):
  AdamW, backbone lr 1e-4 / head lr 1e-3, cosine schedule, weight decay 1e-4,
  class-weighted cross-entropy with 0.05 label smoothing, AMP, batch 16.
Input resolution and normalisation come from each model's own pretrained
data config (224 for DenseNet/VGG/ViT, 299 for Inception/Xception).
Best epoch is chosen on validation macro-F1; test metrics are reported from it.
"""
import argparse, json, os, random, time

# the hf-xet storage backend stalls mid-download on this network; the
# classic HTTP path downloads steadily, so force it.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support)
import timm
from timm.data import resolve_model_data_config, create_transform

ENCODERS = {
    "DenseNet-201":  "densenet201",
    "ViT-Base-16":   "vit_base_patch16_224",
    "VGG19":         "vgg19",
    "InceptionV3":   "inception_v3",
    "Xception":      "xception",
}


def seed_all(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class KidneyDS(Dataset):
    """Reads from the decoded-image memmap so no dataloader workers are needed.

    On Windows every worker respawns and re-imports torch (~750 MB each), which
    starved the GPU; with the memmap, transforms in the main process are cheap.
    """

    def __init__(self, frame, tf, mm, row_of_path, center_crop=1.0):
        self.rows = [row_of_path[p] for p in frame.path]
        self.labels = frame.label.tolist()
        self.tf = tf
        self.mm = mm
        self.cc = center_crop

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        a = np.asarray(self.mm[self.rows[i]])
        if self.cc < 1.0:
            # keep only the central fraction: strips the scanner chrome, depth
            # rulers and captions that let a linear probe hit 94% without anatomy
            px = a.shape[0]
            lo = int(px * (1 - self.cc) / 2)
            a = a[lo:px - lo, lo:px - lo]
        return self.tf(Image.fromarray(a)), self.labels[i]


@torch.no_grad()
def evaluate(model, loader, crit, dev):
    model.eval()
    loss_sum, n = 0.0, 0
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(x)
            loss = crit(out, y)
        loss_sum += loss.item() * y.size(0); n += y.size(0)
        ys.append(y.cpu().numpy()); ps.append(out.argmax(1).cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return loss_sum / n, accuracy_score(y, p), f1_score(y, p, average="macro"), y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(ENCODERS))
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="clean")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cache", required=True, help="dir holding images_<px>.dat")
    ap.add_argument("--cache-px", type=int, default=384)
    ap.add_argument("--center-crop", type=float, default=1.0,
                    help="keep only this central fraction of the frame (1.0 = whole image)")
    args = ap.parse_args()

    seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.split_csv)
    classes = sorted(df.cls.unique())
    cls_of_label = dict(zip(df.label, df.cls))
    names = [cls_of_label[i] for i in sorted(df.label.unique())]
    n_cls = len(names)

    model = timm.create_model(ENCODERS[args.encoder], pretrained=True, num_classes=n_cls).to(dev)
    dcfg = resolve_model_data_config(model)
    tf_train = create_transform(**dcfg, is_training=True, auto_augment=None,
                                scale=(0.70, 1.0), ratio=(0.9, 1.11),
                                hflip=0.5, vflip=0.0, color_jitter=0.1,
                                re_prob=0.0)
    tf_eval = create_transform(**dcfg, is_training=False)
    print(f"[{args.encoder}] input {dcfg['input_size']}  mean {tuple(round(m,3) for m in dcfg['mean'])}")

    idx = json.load(open(os.path.join(args.cache, f"images_{args.cache_px}_index.json")))
    row_of_path = {p: i for i, p in enumerate(idx["paths"])}
    mm = np.load(os.path.join(args.cache, f"images_{args.cache_px}.dat"), mmap_mode="r")

    parts = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    loaders = {}
    for s, frame in parts.items():
        ds = KidneyDS(frame, tf_train if s == "train" else tf_eval, mm, row_of_path,
                      args.center_crop)
        loaders[s] = DataLoader(ds, batch_size=args.batch_size, shuffle=(s == "train"),
                                num_workers=0, pin_memory=True, drop_last=False)
    # train-set accuracy must be measured without augmentation to be comparable
    loaders["train_eval"] = DataLoader(
        KidneyDS(parts["train"], tf_eval, mm, row_of_path, args.center_crop),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"[{args.encoder}] train {len(parts['train'])}  val {len(parts['val'])}  "
          f"test {len(parts['test'])}  classes {names}")

    cnt = parts["train"].label.value_counts().sort_index().values
    w = torch.tensor((cnt.sum() / (n_cls * cnt)), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

    head_names = [n for n, _ in model.named_parameters()
                  if n.startswith(tuple(["head", "classifier", "fc", "last_linear"]))]
    head_p = [p for n, p in model.named_parameters() if n in head_names]
    back_p = [p for n, p in model.named_parameters() if n not in head_names]
    opt = torch.optim.AdamW([{"params": back_p, "lr": 1e-4},
                             {"params": head_p, "lr": 1e-3}], weight_decay=1e-4)
    steps = args.epochs * len(loaders["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    hist, best = [], {"val_f1": -1}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        run, seen, correct = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            prev_scale = scaler.get_scale()
            scaler.step(opt); scaler.update()
            if scaler.get_scale() >= prev_scale:   # skip sched on a skipped step
                sched.step()
            run += loss.item() * y.size(0); seen += y.size(0)
            correct += (out.argmax(1) == y).sum().item()
        vl, va, vf1, _, _ = evaluate(model, loaders["val"], crit, dev)
        hist.append(dict(epoch=ep, train_loss=run / seen, train_acc_online=correct / seen,
                         val_loss=vl, val_acc=va, val_macro_f1=vf1))
        print(f"[{args.encoder}] ep{ep}/{args.epochs} "
              f"train_loss {run/seen:.4f} train_acc~{correct/seen:.4f} | "
              f"val_loss {vl:.4f} val_acc {va:.4f} val_f1 {vf1:.4f}", flush=True)
        if vf1 > best["val_f1"]:
            best = {"val_f1": vf1, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    train_minutes = (time.time() - t0) / 60
    model.load_state_dict(best["state"])
    print(f"[{args.encoder}] best epoch {best['epoch']} (val macro-F1 {best['val_f1']:.4f})")

    _, tr_acc, _, _, _ = evaluate(model, loaders["train_eval"], crit, dev)
    te_loss, te_acc, te_f1, y, p = evaluate(model, loaders["test"], crit, dev)
    pm, rm, fm, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)

    res = dict(
        encoder=args.encoder, timm_model=ENCODERS[args.encoder], tag=args.tag,
        input_size=list(dcfg["input_size"]), params_m=round(sum(q.numel() for q in model.parameters()) / 1e6, 2),
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        center_crop=args.center_crop,
        best_epoch=best["epoch"], train_minutes=round(train_minutes, 2),
        n_train=len(parts["train"]), n_val=len(parts["val"]), n_test=len(parts["test"]),
        train_accuracy=round(float(tr_acc), 4),
        test_accuracy=round(float(te_acc), 4),
        test_loss=round(float(te_loss), 4),
        precision_macro=round(float(pm), 4), recall_macro=round(float(rm), 4), f1_macro=round(float(fm), 4),
        precision_weighted=round(float(pw), 4), recall_weighted=round(float(rw), 4), f1_weighted=round(float(fw), 4),
        classes=names,
        confusion_matrix=confusion_matrix(y, p).tolist(),
        per_class=classification_report(y, p, target_names=names, output_dict=True, zero_division=0),
        history=hist,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    fn = os.path.join(args.out_dir, f"{args.tag}__{args.encoder}.json")
    json.dump(res, open(fn, "w"), indent=2)
    print(f"[{args.encoder}] TEST acc {te_acc:.4f}  loss {te_loss:.4f}  "
          f"macro P/R/F1 {pm:.4f}/{rm:.4f}/{fm:.4f}  (train acc {tr_acc:.4f})")
    print(f"[{args.encoder}] -> {fn}")


if __name__ == "__main__":
    main()
