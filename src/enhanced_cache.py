"""Precompute enhanced images with a process pool and serve them from RAM.

Non-local-means denoising costs ~0.4-0.6 s per image, which would dominate every epoch if it
ran inside __getitem__. Enhancement is deterministic given (image, params), so it is computed
once per configuration across a pool of workers and reused for every epoch of that run.
"""
import multiprocessing as mp
import os

import numpy as np
from PIL import Image

from .config import IMG_SIZE
from .data import (URICADSMultiTaskDataset, enhance_ultrasound)


def _enhance_one(args):
    path, size, params, fast = args
    image = Image.open(path).convert('RGB').resize((size[1], size[0]), Image.BILINEAR)
    return enhance_ultrasound(np.array(image).astype(np.uint8), params, fast_denoise=fast)


class EnhancementPool:
    """A reusable worker pool.

    Phase 6 enhances the dataset once per candidate configuration, hundreds of times over. On
    Windows each pool creation respawns workers that re-import the whole module chain, costing
    roughly 15 s every time. Holding one pool open for the whole search removes that.
    """

    def __init__(self, workers=None):
        self.workers = workers or max(1, min(12, (os.cpu_count() or 4) - 2))
        self._pool = None

    @property
    def pool(self):
        if self._pool is None:
            self._pool = mp.Pool(self.workers)
        return self._pool

    def map(self, jobs):
        return self.pool.map(_enhance_one, jobs, chunksize=8)

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def precompute_enhanced(paths, params, img_size=IMG_SIZE, fast_denoise=False, workers=None,
                        desc='', pool=None):
    """Returns a uint8 array (N, H, W, 3) of enhanced images."""
    jobs = [(p, img_size, params, fast_denoise) for p in paths]
    if pool is not None:
        out = pool.map(jobs)
        n_workers = pool.workers
    else:
        n_workers = workers or max(1, min(12, (os.cpu_count() or 4) - 2))
        if n_workers == 1:
            out = [_enhance_one(j) for j in jobs]
        else:
            with mp.Pool(n_workers) as p:
                out = p.map(_enhance_one, jobs, chunksize=8)
    if desc:
        print(f'  enhanced {len(out)} images ({desc}) with {n_workers} workers', flush=True)
    return np.stack(out)


class PrecomputedEnhancedDataset(URICADSMultiTaskDataset):
    """Identical to the base dataset, but reads images from a precomputed enhanced array."""

    supports_workers = False  # see make_loader: the in-RAM image array cannot be pickled

    def __init__(self, manifest_df, images, img_size=IMG_SIZE, **kwargs):
        super().__init__(manifest_df, img_size=img_size, enhancement=None, **kwargs)
        self.images = images

    def _load_image(self, idx, row, target_h, target_w):
        # Serving the image through this hook (rather than overriding __getitem__) keeps the
        # base class in charge of augmentation, so masks and image stay in register.
        return self.images[idx]


def build_enhanced_splits(manifest, params, splits=('train', 'val', 'test'),
                          img_size=IMG_SIZE, fast_denoise=False, workers=None, pool=None,
                          quiet=False, augment_train=False, augment_params=None):
    """Enhance and wrap the requested splits in one pass each."""
    datasets = {}
    for split in splits:
        sub = manifest[manifest['split'] == split]
        images = precompute_enhanced(sub['image_path'].tolist(), params, img_size,
                                     fast_denoise=fast_denoise, workers=workers,
                                     desc='' if quiet else split, pool=pool)
        datasets[split] = PrecomputedEnhancedDataset(
            sub, images, img_size=img_size,
            augment=augment_train and split == 'train', augment_params=augment_params)
    return datasets
