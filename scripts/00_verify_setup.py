"""Phase 0 (local): verify GPU, data integrity, manifest, and measure real throughput.

Run this before any training. It answers the questions that decide whether the rest of the
pipeline is feasible on this machine: does CUDA work, does a batch fit in 4 GB, and how long
does one epoch actually take.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import (DATASET_DIR, IMG_SIZE, LOGS_DIR, MANIFEST_PATH,
                        PROJECT_ROOT)
from src.data import load_manifest, split_datasets
from src.engine import get_device, make_loader, run_epoch, set_seed
from src.model import MultiTaskNet, MultiTaskNetV2


def section(title):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70, flush=True)


def main():
    report = {}

    section('1. ENVIRONMENT')
    print(f'Python           : {sys.version.split()[0]}')
    print(f'PyTorch          : {torch.__version__}')
    print(f'CUDA available   : {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'GPU              : {props.name}')
        print(f'VRAM             : {props.total_memory / 1e9:.2f} GB')
        print(f'Compute capability: {props.major}.{props.minor}')
        report['gpu'] = props.name
        report['vram_gb'] = round(props.total_memory / 1e9, 2)
    else:
        raise SystemExit('CUDA unavailable - stopping (training on CPU is not viable here).')
    report['torch'] = torch.__version__

    section('2. DATASET INTEGRITY')
    for sub in ('images', 'labels', 'masks'):
        n = len(os.listdir(os.path.join(DATASET_DIR, sub)))
        print(f'{sub:8s}: {n} files')
        report[f'n_{sub}'] = n

    section('3. MANIFEST')
    manifest = load_manifest()
    print(f'Rows             : {len(manifest)}')
    print(f'Columns          : {list(manifest.columns)}')
    print(f'\nSplit sizes:\n{manifest["split"].value_counts().to_string()}')
    print(f'\nGlobal diagnosis:\n{manifest["global_diagnosis"].value_counts().to_string()}')
    missing_img = (~manifest['image_path'].apply(os.path.exists)).sum()
    missing_mask = (~manifest['mask_path'].apply(os.path.exists)).sum()
    print(f'\nMissing images after path remap : {missing_img}')
    print(f'Missing masks after path remap  : {missing_mask}')
    if missing_img or missing_mask:
        raise SystemExit('Manifest paths do not resolve locally - fix before training.')
    report['splits'] = manifest['split'].value_counts().to_dict()

    section('4. DATASET SAMPLE')
    set_seed()
    train_ds, val_ds, test_ds = split_datasets(manifest)
    sample = train_ds[0]
    for k, v in sample.items():
        print(f'{k:20s}: {tuple(v.shape) if hasattr(v, "shape") else v}')
    lesion_pos = sum(1 for i in range(len(test_ds)) if test_ds[i]['has_lesion'].item() > 0) \
        if os.environ.get('COUNT_LESIONS') else None
    if lesion_pos is not None:
        print(f'test images containing >=1 lesion: {lesion_pos}/{len(test_ds)}')

    section('5. GPU FORWARD/BACKWARD FIT TEST (batch size 4 @ 512x512)')
    device = get_device()
    for name, cls in (('MultiTaskNet (Phase 2/3)', MultiTaskNet),
                      ('MultiTaskNetV2 (Phase 4/5/6)', MultiTaskNetV2)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = cls().to(device)
        x = torch.randn(4, 3, *IMG_SIZE, device=device)
        with torch.autocast('cuda'):
            out = model(x)
            loss = (out['kidney_seg'] if isinstance(out, dict) else out[2]).mean()
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1e9
        n_params = sum(p.numel() for p in model.parameters())
        print(f'{name:30s} OK | params={n_params/1e6:.2f}M | peak VRAM={peak:.2f} GB')
        report[f'peak_vram_{cls.__name__}'] = round(peak, 2)
        report[f'params_{cls.__name__}'] = n_params
        del model, x, out, loss
        torch.cuda.empty_cache()

    section('6. THROUGHPUT BENCHMARK (40 training samples, AMP on)')
    set_seed()
    model = MultiTaskNetV2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    subset = torch.utils.data.Subset(train_ds, range(40))
    loader = make_loader(subset, 4, False, num_workers=0)

    run_epoch(model, loader, device, optimizer=optimizer, scaler=scaler)  # warm-up
    torch.cuda.synchronize()
    t0 = time.time()
    run_epoch(model, loader, device, optimizer=optimizer, scaler=scaler)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    per_sample = elapsed / 40
    n_train = int((manifest['split'] == 'train').sum())
    n_val = int((manifest['split'] == 'val').sum())
    est_epoch = per_sample * n_train + per_sample * 0.4 * n_val
    print(f'40 samples in {elapsed:.1f}s -> {per_sample:.3f} s/sample')
    print(f'Estimated full epoch (train {n_train} + val {n_val}): {est_epoch/60:.1f} min')
    print(f'Estimated 15-epoch run: {est_epoch*15/3600:.2f} hours')
    report['s_per_sample'] = round(per_sample, 4)
    report['est_epoch_min'] = round(est_epoch / 60, 2)
    report['est_15epoch_hours'] = round(est_epoch * 15 / 3600, 2)

    with open(os.path.join(LOGS_DIR, 'phase0_local_setup.json'), 'w') as fh:
        json.dump(report, fh, indent=2)
    section('DONE - report saved to results/logs/phase0_local_setup.json')


if __name__ == '__main__':
    main()
