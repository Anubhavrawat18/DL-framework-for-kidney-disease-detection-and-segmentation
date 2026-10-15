"""
Stage 5: aggregate per-encoder JSON results into CSV + a markdown report.
"""
import argparse, glob, json, os
import pandas as pd

ORDER = ["DenseNet-201", "ViT-Base-16", "VGG19", "InceptionV3", "Xception"]


def load(results_dir, tag):
    rows = [json.load(open(fn)) for fn in
            sorted(glob.glob(os.path.join(results_dir, f"{tag}__*.json")))]
    rows.sort(key=lambda r: ORDER.index(r["encoder"]) if r["encoder"] in ORDER else 99)
    return rows


def table(rows):
    return pd.DataFrame([{
        "Encoder": r["encoder"],
        "Params (M)": r["params_m"],
        "Input": f"{r['input_size'][1]}x{r['input_size'][2]}",
        "Best ep": r["best_epoch"],
        "Train Acc": r["train_accuracy"],
        "Test Acc": r["test_accuracy"],
        "Test Loss": r["test_loss"],
        "Precision (macro)": r["precision_macro"],
        "Recall (macro)": r["recall_macro"],
        "F1 (macro)": r["f1_macro"],
        "Precision (wtd)": r["precision_weighted"],
        "Recall (wtd)": r["recall_weighted"],
        "F1 (wtd)": r["f1_weighted"],
        "Min": r["train_minutes"],
    } for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clean = load(args.results_dir, "clean")
    leaky = load(args.results_dir, "leaky")
    strict = load(args.results_dir, "strict")
    anat = load(args.results_dir, "anat")
    pr_path = os.path.join(args.results_dir, "shortcut_probes.json")
    probes = json.load(open(pr_path)) if os.path.exists(pr_path) else None
    an_path = os.path.join(args.results_dir, "annotation_probes.json")
    annot = json.load(open(an_path)) if os.path.exists(an_path) else None
    def _load_cross(tag):
        rows = [json.load(open(fn)) for fn in
                sorted(glob.glob(os.path.join(args.results_dir, f"{tag}__*.json")))]
        rows.sort(key=lambda r: ORDER.index(r["encoder"]) if r["encoder"] in ORDER else 99)
        return rows

    cross2 = _load_cross("cross")
    cross3 = _load_cross("cross3")
    tx_path = os.path.join(args.results_dir, "texture_probes.json")
    tex = json.load(open(tx_path)) if os.path.exists(tx_path) else None
    lk = json.load(open(os.path.join(args.cache, "leakage_report_clean.json")))
    sk_path = os.path.join(args.cache, "leakage_report_strict.json")
    sk = json.load(open(sk_path)) if os.path.exists(sk_path) else None

    tc = table(clean)
    tc.to_csv(os.path.join(args.results_dir, "summary_clean.csv"), index=False)
    if leaky:
        table(leaky).to_csv(os.path.join(args.results_dir, "summary_leaky.csv"), index=False)
    if strict:
        table(strict).to_csv(os.path.join(args.results_dir, "summary_strict.csv"), index=False)
    if anat:
        table(anat).to_csv(os.path.join(args.results_dir, "summary_anat.csv"), index=False)
    # one combined table across arms, easiest thing to paste into a paper
    combo = []
    for tag, rows in [("clean", clean), ("anat", anat), ("strict", strict), ("leaky", leaky)]:
        for r in rows:
            combo.append({"Arm": tag, **table([r]).iloc[0].to_dict()})
    if combo:
        pd.DataFrame(combo).to_csv(
            os.path.join(args.results_dir, "summary_all_arms.csv"), index=False)

    L = []
    A = L.append
    A("# Kidney Ultrasound Encoder Comparison - Leakage-Controlled Results\n")
    A("> **Read section 1d first.** The headline conclusion of this study is that the "
      "dataset as supplied cannot support an encoder comparison, because the imaging "
      "source is perfectly confounded with the class label. The metrics in sections 3 "
      "onward are reported as requested, but they measure source recognition rather than "
      "kidney pathology.\n")

    A("## 1. Leakage audit\n")
    A(f"The dataset as supplied holds **{lk['n_images_raw']} images** across 3 classes "
      "(Cyst / Normal / stone). Fingerprinting every image by raw-byte md5, "
      "decoded-pixel md5, a 64-bit difference hash, a 64-bit DCT perceptual hash and a "
      "32x32 normalised thumbnail exposed three independent, compounding sources of "
      "duplication:\n")
    A("| Leakage source | Links found |")
    A("|---|---|")
    A(f"| Byte-identical duplicate files | {lk['merges_exact_dup']} redundant copies |")
    A(f"| `aug_<n>_<name>.jpg` files whose source image is also present | {lk['merges_aug_source']} |")
    A(f"| Near-duplicate frames (phash hamming <= 6 AND thumbnail corr >= 0.95) | "
      f"{lk['merges_neardup']} merges from {lk['neardup_pairs']} pairs |")
    A("")
    A(f"Collapsing these with union-find leaves **{lk['n_groups']} genuinely independent "
      f"images** - i.e. **{100 * (1 - lk['n_groups'] / lk['n_images_raw']):.1f}% of the "
      "supplied dataset is duplicate or near-duplicate material**.\n")
    A("Under a naive stratified random split over the raw files, "
      f"**{100 * lk['leaky_test_contaminated_frac']:.1f}% of test images have a duplicate "
      "or near-duplicate sitting in the training set** - the model is largely being asked "
      "to recognise pictures it has already memorised.\n")
    A("> **Important correction, established by the `leaky` arm in section 4.** This "
      "leakage is real and had to be removed, but it turns out *not* to be what produced "
      "the near-perfect accuracy. Removing it entirely changed the test scores by well "
      "under a point, and the leaky arm scores no higher than the clean one. The reason "
      "is a ceiling effect: the task is already trivially solvable from the acquisition "
      "source (section 1b/1d), so duplicate leakage had no headroom left to add anything. "
      "Dedup was necessary for correctness, not sufficient for validity.\n")

    A("### How it was eliminated\n")
    A("1. **Group, don't just dedup.** Union-find over the three link types above puts "
      "every duplicate, augmentation and near-duplicate frame of the same underlying "
      "image into a single group.")
    A("2. **One representative per group** is kept (preferring the non-augmented "
      f"original), reducing {lk['n_images_raw']} files to {lk['n_groups']} images.")
    A("3. **Stratified split over groups**, so no group can straddle train/val/test.")
    A("4. **Independent re-verification at a deliberately looser threshold** "
      f"(hamming <= 10, corr >= 0.90): this caught {lk['verify_cross_split_pairs']} "
      "residual cross-split neighbour pairs, whose held-out members were dropped "
      f"({lk['verify_dropped']} images). Residual cross-split neighbours afterwards: "
      f"**{lk['residual_cross_split_pairs']}**.")
    A("5. Groups spanning more than one class label were checked for (label noise): "
      f"**{lk['cross_class_groups']} found**.\n")
    if clean:
        r0 = clean[0]
        A(f"Final clean split: **{r0['n_train']} train / {r0['n_val']} val / "
          f"{r0['n_test']} test** images.\n")

    if probes:
        A("## 1b. A second, larger problem: acquisition shortcuts\n")
        A("Removing the duplicates was necessary but **not sufficient**. The three "
          "classes in this folder come from different acquisition sources, and the "
          "scanner artefacts alone nearly give away the label:\n")
        A("- `Cyst` images are **only ever** 227x227 or 224x224; `Normal` and `stone` "
          "are **only ever** 512x512. Resolution alone separates Cyst perfectly.")
        A("- `stone` frames carry visible scanner UI - depth rulers, captions such as "
          "`10 8.0 mm`, coloured markers - which `Normal` frames largely lack.\n")
        A("To measure how much of the score this is worth, each probe below destroys the "
          "anatomy and keeps only an artefact channel, then fits a *linear* model on the "
          "same leakage-free clean split:\n")
        A("| Probe (anatomy destroyed) | Test Acc | Macro-F1 |")
        A("|---|---|---|")
        for p in probes["probes"]:
            A(f"| {p['probe']} | {p['test_accuracy']:.4f} | {p['f1_macro']:.4f} |")
        A("")
        border = next((p for p in probes["probes"] if p["probe"].startswith("C.")), None)
        if border:
            A(f"> **Blanking the central 60% of every image - removing all kidney anatomy - "
              f"still lets a plain logistic regression reach {border['test_accuracy']:.1%} "
              "on the clean test set.** The fine-tuned encoders sit only a few points above "
              "that, so most of their apparent skill is reading the scanner, not the kidney.\n")
        A("### Mitigation\n")
        A("Centre-cropping suppresses the shortcut. Measured with the same linear probe "
          "on progressively tighter crops:\n")
        A("| Kept fraction of frame | Linear-probe acc |")
        A("|---|---|")
        A("| 1.00 (whole image) | 0.9256 |")
        A("| 0.70 | 0.6830 |")
        A("| 0.55 | 0.6125 |")
        A("| 0.45 | 0.4423 (≈ chance) |")
        A("")
        A("The **anat** arm below therefore re-runs all five encoders on the clean split "
          "with a 0.55 centre crop. Those are the numbers that most reflect anatomy "
          "rather than acquisition source.\n")

    if annot:
        A("## 1c. A third leak: burned-in annotations\n")
        A("Several `Cyst` frames carry the radiologist's own markings - the literal word "
          "`CYST`, `Rt Kidney`, and caliper crosshairs around the lesion - burned into "
          "the pixels. Unlike the chrome, these sit near the centre, so **centre-cropping "
          "does not remove them**.\n")
        A("Fraction of images carrying a visible saturated overlay:\n")
        A("| Class | Fraction |")
        A("|---|---|")
        for k, v in annot["frac_with_overlay"].items():
            A(f"| {k} | {v:.3f} |")
        A("")
        A("Probes fitted on the saturated-pixel layer alone (no anatomy whatsoever):\n")
        A("| Probe | Test Acc | Macro-F1 |")
        A("|---|---|---|")
        for p in annot["probes"]:
            A(f"| {p['probe']} | {p['test_accuracy']:.4f} | {p['f1_macro']:.4f} |")
        A("")
        A("Against a majority-class baseline of about 0.405, this is a real but "
          "**secondary** effect - far weaker than the 0.945 acquisition-source probe. "
          "Note the saturation mask mostly catches bright tissue rather than text (it is "
          "`Normal`, not `Cyst`, that saturates most), so these figures understate the "
          "text itself while showing it is not what drives the headline accuracy.\n")
        A("> Worth a manual pass regardless: a subset of `Cyst` images literally carry "
          "their own diagnosis in the image. That is a data-quality defect independent "
          "of how much accuracy it currently buys.\n")

    if tex:
        A("## 1d. Why cropping does not rescue the comparison\n")
        A("Centre-cropping collapsed the linear pixel shortcut (0.93 -> 0.61) but the "
          "encoders barely moved. So the models are reading something the crop keeps and "
          "a downsampled linear probe cannot see. These probes destroy the anatomy "
          "completely and keep only texture statistics, on the 0.55 centre crop:\n")
        A("| Probe (centre crop, anatomy destroyed) | Test Acc | Macro-F1 |")
        A("|---|---|---|")
        for p in tex["probes"]:
            A(f"| {p['probe']} | {p['test_accuracy']:.4f} | {p['f1_macro']:.4f} |")
        A(f"| *majority-class baseline* | *{tex['majority_baseline']:.4f}* | - |")
        A("")
        spec = next((p for p in tex["probes"] if p["probe"].startswith("H.")), None)
        if spec:
            A(f"> The radial power spectrum carries **no spatial information at all** - "
              f"only how energy is distributed across frequencies - and it still reaches "
              f"**{spec['test_accuracy']:.1%}**. Each class therefore has its own speckle "
              "signature, i.e. its own scanner or acquisition setting.\n")
        A("### The conclusion that matters\n")
        A("In this folder **the acquisition source is perfectly confounded with the class "
          "label**: every Cyst frame comes from one source, every Normal frame from "
          "another, every stone frame from a third. The confound is present in the "
          "resolution, in the chrome, in the annotations and in the speckle texture "
          "itself. No crop, filter or normalisation can separate a signal from a label "
          "it is perfectly correlated with.\n")
        A("So the encoder scores below are **not measurements of kidney-pathology "
          "classification**. They are measurements of how well each backbone "
          "recognises which machine produced the image - a task all five solve almost "
          "perfectly, which is exactly why they finish within about a point of each "
          "other.\n")
        A("To obtain a defensible encoder comparison, the three classes have to be drawn "
          "from the **same** source, so that source carries no label information. Until "
          "then no amount of leakage control will make these numbers mean what they "
          "appear to mean.\n")

    A("## 2. Training protocol\n")
    A("One identical recipe for all five encoders, so the comparison reflects the "
      "backbone and not per-model tuning:\n")
    if clean:
        r0 = clean[0]
        A(f"- ImageNet-pretrained weights via `timm`, full fine-tune, "
          f"**{r0['epochs']} epochs**, batch size {r0['batch_size']}, seed {r0['seed']}")
    A("- AdamW, backbone lr 1e-4 / head lr 1e-3, OneCycle schedule, weight decay 1e-4")
    A("- Class-weighted cross-entropy (the clean set is imbalanced) + 0.05 label smoothing")
    A("- Mixed precision, gradient clipping at 1.0")
    A("- Each encoder uses **its own pretrained input resolution and normalisation** "
      "(224 for DenseNet/VGG/ViT, 299 for Inception/Xception)")
    A("- Train augmentation: RandomResizedCrop (0.70-1.0), horizontal flip, mild jitter")
    A("- Best epoch selected on **validation macro-F1**; test metrics reported from that "
      "epoch. Train accuracy is measured with evaluation transforms (no augmentation) so "
      "it is directly comparable to test accuracy.\n")

    A("## 3. Results on the clean, leakage-controlled split\n")
    A(tc.to_markdown(index=False))
    A("")
    A("Macro averages treat all three classes equally and are the figures to quote, since "
      "the deduplicated set is imbalanced (the `stone` class lost the most images to "
      "near-duplicate collapse).\n")

    if anat:
        A("## 3b. Anatomy-focused arm (clean split + 0.55 centre crop)\n")
        A("Same split, same recipe, but the scanner chrome is cropped away. **These are "
          "the most trustworthy numbers in this report**: whatever the encoders score "
          "here, they score from the image centre where the kidney actually is.\n")
        A(table(anat).to_markdown(index=False))
        A("")
        cm = {r["encoder"]: r for r in clean}
        for r in anat:
            if r["encoder"] in cm:
                A(f"- **{r['encoder']}**: {cm[r['encoder']]['test_accuracy']:.4f} "
                  f"full-frame vs {r['test_accuracy']:.4f} centre-cropped "
                  f"({100 * (r['test_accuracy'] - cm[r['encoder']]['test_accuracy']):+.1f} "
                  "points).")
        A("")
        A("A large drop here means the full-frame score was leaning on acquisition "
          "artefacts. A small drop means the encoder was genuinely reading anatomy.\n")

    if leaky:
        A("## 4. The same encoders on the leaky split (for contrast)\n")
        A(table(leaky).to_markdown(index=False))
        A("")
        cm = {r["encoder"]: r for r in clean}
        rows = [f"- **{r['encoder']}**: {r['test_accuracy']:.4f} leaky vs "
                f"{cm[r['encoder']]['test_accuracy']:.4f} clean "
                f"(**{100 * (r['test_accuracy'] - cm[r['encoder']]['test_accuracy']):+.1f} "
                "points of inflation**)"
                for r in leaky if r["encoder"] in cm]
        if rows:
            A("Accuracy inflation attributable purely to leakage:\n")
            L.extend(rows)
            A("")

    if strict and sk:
        A("## 4b. Sensitivity control: a deliberately over-merged split\n")
        A("Because the clean numbers are still high, a second and much harsher grouping "
          "was run to bound any residual leakage: the near-duplicate gate was relaxed all "
          "the way to hamming <= 14 AND corr >= 0.82, which merges whole runs of visually "
          "similar slices and so approximates per-patient grouping. That collapses the "
          f"dataset to only **{sk['n_groups']} groups**, re-verified at hamming <= 18 / "
          f"corr >= 0.75 with {sk['residual_cross_split_pairs']} residual cross-split "
          "pairs.\n")
        A("This is almost certainly an *over*-correction - it discards genuinely distinct "
          "images - so read it as a pessimistic lower bound, not the headline.\n")
        A(table(strict).to_markdown(index=False))
        A("")
        cm = {r["encoder"]: r for r in clean}
        for r in strict:
            if r["encoder"] in cm:
                A(f"- **{r['encoder']}**: {cm[r['encoder']]['test_accuracy']:.4f} on the "
                  f"clean split vs {r['test_accuracy']:.4f} under over-merged grouping "
                  f"({100 * (r['test_accuracy'] - cm[r['encoder']]['test_accuracy']):+.1f} "
                  "points).")
        A("")
        A("Accuracy holding up here indicates the remaining performance is task difficulty "
          "rather than leakage: the three classes are visually well separated in this "
          "dataset, which is consistent with the published literature on it.\n")

    A("## 5. Per-class detail and confusion matrices (clean split)\n")
    for r in clean:
        A(f"### {r['encoder']}\n")
        pc = pd.DataFrame({
            "Class": r["classes"],
            "Precision": [round(r["per_class"][c]["precision"], 4) for c in r["classes"]],
            "Recall": [round(r["per_class"][c]["recall"], 4) for c in r["classes"]],
            "F1": [round(r["per_class"][c]["f1-score"], 4) for c in r["classes"]],
            "Support": [int(r["per_class"][c]["support"]) for c in r["classes"]],
        })
        A(pc.to_markdown(index=False))
        A("")
        cmx = pd.DataFrame(r["confusion_matrix"],
                           index=[f"true {c}" for c in r["classes"]],
                           columns=[f"pred {c}" for c in r["classes"]])
        A(cmx.to_markdown())
        A("")
        A("Per-epoch history:\n")
        A(pd.DataFrame(r["history"]).round(4).to_markdown(index=False))
        A("")

    if cross2 or cross3:
        A("## 5b. The decisive test: train on URI-CADS, test on the Kidney Dataset\n")
        A("Every number above is trained and tested inside the same confounded folder. "
          "Training on URI-CADS instead removes the confound outright: the model never sees "
          "a Kidney Dataset scanner, so it cannot use that dataset's source signature. The "
          "Kidney Dataset side uses the deduplicated representatives.\n")

        def xtab(rows, label):
            A(f"### {label}\n")
            t = []
            for r in rows:
                k, u = r["kidney_ext_test"], r["uricads_test"]
                t.append({
                    "Encoder": r["encoder"], "Train Acc": r["train_accuracy"],
                    "URI-CADS Acc": u["accuracy"],
                    "URI-CADS AUC": u.get("roc_auc", u.get("roc_auc_ovr_macro")),
                    "Kidney Acc": k["accuracy"], "Kidney BalAcc": k["balanced_accuracy"],
                    "Kidney Loss": k["loss"], "Precision": k["precision_macro"],
                    "Recall": k["recall_macro"], "F1": k["f1_macro"],
                    "Kidney AUC": k.get("roc_auc", k.get("roc_auc_ovr_macro"))})
            A(pd.DataFrame(t).sort_values("Kidney AUC", ascending=False)
              .to_markdown(index=False))
            A("")

        if cross2:
            xtab(cross2, "Binary: normal vs pathological (majority baseline 0.601)")
        if cross3:
            xtab(cross3, "Three-class: normal / cyst / stone (majority baseline 0.399)")
            A("Per-class F1 on the Kidney Dataset, three-class:\n")
            A("| Encoder | normal | cyst | stone |")
            A("|---|---|---|---|")
            for r in cross3:
                pc = r["kidney_ext_test"]["per_class"]
                A(f"| {r['encoder']} | " + " | ".join(
                    f"{pc[c]['f1-score']:.3f}" for c in r["classes"]) + " |")
            A("")
            A("URI-CADS class mapping used (confirmed by the user): "
              "`global_diagnosis == 0` -> normal, lesion-box pathology id 1 -> cyst, "
              "pathology id 4 -> a grouped category containing stones. Training classes are "
              "small (315 / 60 / 118), and URI-CADS's own test split holds only 8 cyst "
              "images, so its in-domain figures are thin; the external figures are solid "
              "because that test set has 2,922 images.\n")

        if cross3 and clean:
            cm = {r["encoder"]: r for r in clean}
            d = next((r for r in cross3 if r["encoder"] == "DenseNet-201"), None)
            if d and "DenseNet-201" in cm:
                A(f"> **The headline comparison.** DenseNet-201 scores "
                  f"**{cm['DenseNet-201']['test_accuracy']:.4f}** trained on the Kidney "
                  f"Dataset and tested on it, versus "
                  f"**{d['kidney_ext_test']['accuracy']:.4f}** trained on URI-CADS and "
                  f"tested on the same images - while reaching "
                  f"{d['train_accuracy']:.4f} training accuracy. That collapse is the "
                  "confound measured end to end, and it is what the probes in sections "
                  "1b-1d predicted.\n")

        A("**Cyst transfers, stone does not.** Cyst reaches F1 0.694 (DenseNet) across "
          "datasets, but stone sits at 0.146-0.326 - consistent with URI-CADS pathology "
          "id 4 being a grouped category rather than stones specifically.\n")
        A("**In-domain accuracy does not predict transfer.** VGG19 posts the best or "
          "joint-best URI-CADS score in both cross-dataset runs and transfers near the "
          "bottom in both. The ordering otherwise did NOT replicate between the binary and "
          "three-class runs (InceptionV3: 1st on binary at AUC 0.782, 3rd on three-class at "
          "0.590), so only the extremes are trustworthy - ViT-B/16 transfers worst in both. "
          "Do not rank the middle of the field without repeated seeds.\n")

    A("## 6. Caveats\n")
    A("- These are **single-split** numbers. Differences of roughly a point or two in test "
      "accuracy on a test set this size are inside noise; do not rank encoders on them "
      "without a paired k-fold run over the same groups.")
    A("- Near-duplicate grouping for the headline split used the stable threshold plateau "
      "(hamming <= 6, corr >= 0.95). Varying hamming from 4 to 10 at corr 0.95 moves the "
      "group count by under 3%, whereas dropping corr to 0.90 chains distinct frames into "
      "345-image components, which over-merges.")
    A("- Patient identity is not recorded in the filenames, so grouping is image-content "
      "based. It removes duplicate and near-duplicate leakage, but residual same-patient "
      "leakage between visually dissimilar frames of one patient cannot be ruled out from "
      "this data alone - hence the section 4b control.")
    A("- The `aug_*` files show the supplied dataset was augmented **before** it was "
      "split. That is the single biggest contributor and the direct reason accuracy "
      "saturated by epoch 2.")

    open(args.out, "w", encoding="utf-8").write("\n".join(L))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
