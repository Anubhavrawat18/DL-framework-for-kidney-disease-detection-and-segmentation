"""
Stage 10c: train ONE encoder on URI-CADS, evaluate on URI-CADS held-out test
AND on the Kidney Dataset (external domain). Same recipe as the within-dataset
study so the numbers are comparable.

Reporting both matters: the URI-CADS test score is in-domain performance, the
Kidney Dataset score is out-of-domain. The gap between them is the part the
acquisition-source confound cannot manufacture.
"""
import argparse, json, os, random, time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)
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


class DS(Dataset):
    def __init__(self, frame, tf, mm, row_of):
        self.rows = [row_of[p] for p in frame.path]
        self.labels = frame.label.tolist()
        self.tf, self.mm = tf, mm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(np.asarray(self.mm[self.rows[i]]))), self.labels[i]


@torch.no_grad()
def evaluate(model, loader, crit, dev):
    model.eval()
    tot, n, ys, ps, pr = 0.0, 0, [], [], []
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(x)
            loss = crit(out, y)
        tot += loss.item() * y.size(0); n += y.size(0)
        ys.append(y.cpu().numpy()); ps.append(out.argmax(1).cpu().numpy())
        sm = torch.softmax(out.float(), 1)
        pr.append((sm[:, 1] if sm.shape[1] == 2 else sm).cpu().numpy())
    return tot / n, np.concatenate(ys), np.concatenate(ps), np.concatenate(pr)


def pack(loss, y, p, prob, names):
    pm, rm, fm, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
    d = dict(
        n=int(len(y)), loss=round(float(loss), 4),
        accuracy=round(float(accuracy_score(y, p)), 4),
        balanced_accuracy=round(float(balanced_accuracy_score(y, p)), 4),
        precision_macro=round(float(pm), 4), recall_macro=round(float(rm), 4),
        f1_macro=round(float(fm), 4),
        precision_weighted=round(float(pw), 4), recall_weighted=round(float(rw), 4),
        f1_weighted=round(float(fw), 4),
        confusion_matrix=confusion_matrix(y, p).tolist(),
        per_class=classification_report(y, p, target_names=names, output_dict=True,
                                        zero_division=0),
    )
    try:
        if prob.ndim == 1:
            d["roc_auc"] = round(float(roc_auc_score(y, prob)), 4)
        else:
            d["roc_auc_ovr_macro"] = round(
                float(roc_auc_score(y, prob, multi_class="ovr", average="macro")), 4)
    except ValueError:
        pass          # a class absent from this eval set makes AUC undefined
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(ENCODERS))
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="cross")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cache-px", type=int, default=384)
    ap.add_argument("--cache-prefix", default="cross")
    args = ap.parse_args()

    seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.split_csv)
    names = [df[df.label == l].cls.iloc[0] for l in sorted(df.label.unique())]
    n_cls = len(names)
    model = timm.create_model(ENCODERS[args.encoder], pretrained=True,
                              num_classes=n_cls).to(dev)
    dcfg = resolve_model_data_config(model)
    tf_tr = create_transform(**dcfg, is_training=True, auto_augment=None,
                             scale=(0.70, 1.0), ratio=(0.9, 1.11), hflip=0.5,
                             color_jitter=0.1, re_prob=0.0)
    tf_ev = create_transform(**dcfg, is_training=False)

    idx = json.load(open(os.path.join(args.cache, f"{args.cache_prefix}_{args.cache_px}_index.json")))
    row_of = {p: i for i, p in enumerate(idx["paths"])}
    mm = np.load(os.path.join(args.cache, f"{args.cache_prefix}_{args.cache_px}.dat"), mmap_mode="r")

    parts = {s: df[df.split == s].reset_index(drop=True)
             for s in ("train", "val", "test", "ext_test")}
    L = {}
    for s, fr in parts.items():
        L[s] = DataLoader(DS(fr, tf_tr if s == "train" else tf_ev, mm, row_of),
                          batch_size=args.batch_size, shuffle=(s == "train"),
                          num_workers=0, pin_memory=True)
    L["train_eval"] = DataLoader(DS(parts["train"], tf_ev, mm, row_of),
                                 batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, pin_memory=True)
    print(f"[{args.encoder}] URI-CADS train {len(parts['train'])} val {len(parts['val'])} "
          f"test {len(parts['test'])} | Kidney Dataset ext_test {len(parts['ext_test'])}")

    cnt = parts["train"].label.value_counts().sort_index().values
    w = torch.tensor(cnt.sum() / (n_cls * cnt), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    crit_eval = nn.CrossEntropyLoss(label_smoothing=0.05)   # unweighted on eval domains

    hn = tuple(["head", "classifier", "fc", "last_linear"])
    hp = [p for n, p in model.named_parameters() if n.startswith(hn)]
    bp = [p for n, p in model.named_parameters() if not n.startswith(hn)]
    opt = torch.optim.AdamW([{"params": bp, "lr": 1e-4}, {"params": hp, "lr": 1e-3}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[1e-4, 1e-3], total_steps=args.epochs * len(L["train"]), pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    hist, best = [], {"f1": -1}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        run, seen, corr = 0.0, 0, 0
        for x, y in L["train"]:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                out = model(x); loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            prev = scaler.get_scale()
            scaler.step(opt); scaler.update()
            if scaler.get_scale() >= prev:
                sched.step()
            run += loss.item() * y.size(0); seen += y.size(0)
            corr += (out.argmax(1) == y).sum().item()
        vl, vy, vp, _ = evaluate(model, L["val"], crit_eval, dev)
        vf1 = f1_score(vy, vp, average="macro")
        hist.append(dict(epoch=ep, train_loss=run / seen, train_acc_online=corr / seen,
                         val_loss=vl, val_macro_f1=vf1))
        print(f"[{args.encoder}] ep{ep}/{args.epochs} train_loss {run/seen:.4f} "
              f"train_acc~{corr/seen:.4f} | val_loss {vl:.4f} val_f1 {vf1:.4f}", flush=True)
        if vf1 > best["f1"]:
            best = {"f1": vf1, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    mins = (time.time() - t0) / 60
    model.load_state_dict(best["state"])

    _, ty, tp, _ = evaluate(model, L["train_eval"], crit_eval, dev)
    tr_acc = accuracy_score(ty, tp)
    res = dict(
        encoder=args.encoder, timm_model=ENCODERS[args.encoder], tag=args.tag,
        input_size=list(dcfg["input_size"]),
        params_m=round(sum(q.numel() for q in model.parameters()) / 1e6, 2),
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        best_epoch=best["epoch"], train_minutes=round(mins, 2),
        classes=names, train_accuracy=round(float(tr_acc), 4), history=hist,
    )
    for key, split in [("uricads_test", "test"), ("kidney_ext_test", "ext_test")]:
        loss, y, p, prob = evaluate(model, L[split], crit_eval, dev)
        res[key] = pack(loss, y, p, prob, names)
        r = res[key]
        print(f"[{args.encoder}] {key:16s} n={r['n']:5d} acc {r['accuracy']:.4f} "
              f"bal_acc {r['balanced_accuracy']:.4f} loss {r['loss']:.4f} "
              f"macro P/R/F1 {r['precision_macro']:.4f}/{r['recall_macro']:.4f}/"
              f"{r['f1_macro']:.4f} auc "
              f"{r.get('roc_auc', r.get('roc_auc_ovr_macro'))}")

    os.makedirs(args.out_dir, exist_ok=True)
    fn = os.path.join(args.out_dir, f"{args.tag}__{args.encoder}.json")
    json.dump(res, open(fn, "w"), indent=2)
    print(f"[{args.encoder}] -> {fn}")


if __name__ == "__main__":
    main()
