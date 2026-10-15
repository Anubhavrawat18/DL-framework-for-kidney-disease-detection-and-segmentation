"""Dataset, manifest loading, and the ultrasound enhancement layer.

Ported from the Colab notebooks (Phases 1-5) with two local-specific changes:
  * manifest image/mask paths are remapped from Colab paths to this machine's paths
  * enhancement parameters are a runtime argument, so Phase 6 can optimize them
"""
import os

import cv2
import numpy as np
import pandas as pd
import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import (IMAGES_DIR, IMG_SIZE, MANIFEST_PATH, MASKS_DIR,
                     NUM_LOCAL_PATHOLOGY_IDS)


def load_manifest(path=MANIFEST_PATH):
    """Load the saved manifest and repoint its Colab paths at the local dataset copy."""
    manifest = pd.read_pickle(path)
    manifest['image_path'] = manifest['image_path'].apply(
        lambda p: os.path.join(IMAGES_DIR, os.path.basename(p)))
    manifest['mask_path'] = manifest['mask_path'].apply(
        lambda p: os.path.join(MASKS_DIR, os.path.basename(p)))
    return manifest


def load_kidney_mask(mat_path):
    return np.asarray(scipy.io.loadmat(mat_path)['mask']).astype(np.uint8)


def parse_local_lesions(extra_lines):
    """'Bounding box 1: 326 196 48 32 5' -> x_min y_min width height pathology_id."""
    lesions = []
    for line in extra_lines:
        if not line.startswith('Bounding box'):
            continue
        _, values = line.split(':', 1)
        nums = [int(v) for v in values.split()]
        if len(nums) != 5:
            continue
        x, y, w, h, pathology_id = nums
        lesions.append({'bbox': (x, y, x + w, y + h), 'pathology_id': pathology_id})
    return lesions


def bbox_to_coarse_mask(bbox, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    x_min, y_min, x_max, y_max = [int(round(v)) for v in bbox]
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(width, x_max), min(height, y_max)
    if x_max > x_min and y_max > y_min:
        mask[y_min:y_max, x_min:x_max] = 1
    return mask


DEFAULT_ENHANCEMENT = {
    'clahe_clip_limit': 2.0,
    'clahe_tile_grid_size': 8,
    'denoise_h': 10,
    'gamma': 1.2,
}


def enhance_ultrasound(image_rgb, params=None, fast_denoise=False):
    """CLAHE on luminance -> denoise -> gamma. uint8 HxWx3 in, uint8 HxWx3 out.

    fast_denoise swaps non-local-means for a bilateral filter: visually similar but ~50x
    cheaper, which is what makes the Phase 6 search loop tractable.
    """
    params = params or DEFAULT_ENHANCEMENT
    tile = int(params['clahe_tile_grid_size'])

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(params['clahe_clip_limit']),
                            tileGridSize=(tile, tile))
    l_channel = clahe.apply(l_channel)
    image_rgb = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2RGB)

    h = float(params['denoise_h'])
    if h > 0:
        if fast_denoise:
            image_rgb = cv2.bilateralFilter(image_rgb, d=5, sigmaColor=h * 2.5, sigmaSpace=h * 2.5)
        else:
            image_rgb = cv2.fastNlMeansDenoisingColored(
                image_rgb, None, h=h, hColor=h, templateWindowSize=7, searchWindowSize=21)

    gamma = float(params['gamma'])
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype('uint8')
    return cv2.LUT(image_rgb, table)


# Intensity jitter is OFF by default, and that is a domain decision rather than an oversight.
# In renal ultrasound the diagnosis is partly *defined* by echogenicity: two of the dataset's
# labels are poor corticomedullary differentiation and hyper-echoic cortex. Jittering
# brightness, contrast or gamma perturbs exactly the feature the classifier must read. These
# images are also dark (mean intensity about 25/255), so a +-15% brightness shift moves pixels
# by more than the image's own mean. Measured effect of the aggressive profile: diagnosis AUC
# 0.9014 -> 0.8363. Geometric transforms carry no such risk.
GEOMETRIC_AUGMENTATION = {
    'hflip_prob': 0.5,
    'rotate_deg': 10.0,
    'scale_jitter': 0.10,
    'translate_frac': 0.05,
    'brightness': 0.0,
    'contrast': 0.0,
    'gamma_jitter': 0.0,
}

# Retained to test whether a little intensity variation helps once it is small enough not to
# swamp the echogenicity signal.
MILD_AUGMENTATION = dict(GEOMETRIC_AUGMENTATION,
                         brightness=0.03, contrast=0.05, gamma_jitter=0.05)

DEFAULT_AUGMENTATION = GEOMETRIC_AUGMENTATION


def augment_sample(image, masks, rng, params=None):
    """Geometric transforms are applied identically to the image and every mask; intensity
    jitter touches the image only.

    No vertical flip: in an ultrasound sweep depth increases downward, so flipping vertically
    produces an anatomically impossible image. Horizontal flip is safe because either kidney
    can be imaged from either side.
    """
    p = params or DEFAULT_AUGMENTATION
    h, w = image.shape[:2]

    if rng.random() < p['hflip_prob']:
        image = np.ascontiguousarray(image[:, ::-1])
        masks = [np.ascontiguousarray(m[:, ::-1]) for m in masks]

    angle = rng.uniform(-p['rotate_deg'], p['rotate_deg'])
    scale = 1.0 + rng.uniform(-p['scale_jitter'], p['scale_jitter'])
    tx = rng.uniform(-p['translate_frac'], p['translate_frac']) * w
    ty = rng.uniform(-p['translate_frac'], p['translate_frac']) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    masks = [cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0) for m in masks]

    if not (p['contrast'] or p['brightness'] or p['gamma_jitter']):
        return image, masks

    img = image.astype(np.float32)
    img *= 1.0 + rng.uniform(-p['contrast'], p['contrast'])
    img += rng.uniform(-p['brightness'], p['brightness']) * 255.0
    img = np.clip(img, 0, 255)
    g = 1.0 + rng.uniform(-p['gamma_jitter'], p['gamma_jitter'])
    img = 255.0 * np.power(img / 255.0, 1.0 / g)
    return np.clip(img, 0, 255).astype(np.uint8), masks


class URICADSMultiTaskDataset(Dataset):
    """Full 5-head sample: image, kidney mask, lesion union mask, and all three label vectors."""

    def __init__(self, manifest_df, img_size=IMG_SIZE, enhancement=None, fast_denoise=False,
                 augment=False, augment_params=None, seed=0):
        self.manifest = manifest_df.reset_index(drop=True)
        self.img_size = img_size
        self.enhancement = enhancement
        self.fast_denoise = fast_denoise
        self.augment = augment
        self.augment_params = augment_params
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch):
        """Changes the augmentation stream between epochs while staying reproducible."""
        self._epoch = int(epoch)

    def __len__(self):
        return len(self.manifest)

    def _load_image(self, idx, row, target_h, target_w):
        image = Image.open(row['image_path']).convert('RGB')
        image = image.resize((target_w, target_h), Image.BILINEAR)
        image = np.array(image).astype(np.uint8)
        if self.enhancement is not None:
            image = enhance_ultrasound(image, self.enhancement, fast_denoise=self.fast_denoise)
        return image

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        target_h, target_w = self.img_size

        with Image.open(row['image_path']) as probe:
            orig_width, orig_height = probe.size
        image = self._load_image(idx, row, target_h, target_w)

        kidney_mask = load_kidney_mask(row['mask_path'])
        kidney_mask = np.array(Image.fromarray(kidney_mask).resize(
            (target_w, target_h), Image.NEAREST))

        scale_x = target_w / orig_width
        scale_y = target_h / orig_height
        lesions = parse_local_lesions(row['extra_lines'])

        lesion_union_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        local_pathology_vec = np.zeros(NUM_LOCAL_PATHOLOGY_IDS, dtype=np.float32)
        for lesion in lesions:
            x_min, y_min, x_max, y_max = lesion['bbox']
            scaled = (x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y)
            lesion_union_mask = np.maximum(
                lesion_union_mask, bbox_to_coarse_mask(scaled, target_h, target_w))
            pid = lesion['pathology_id']
            if 1 <= pid <= NUM_LOCAL_PATHOLOGY_IDS:
                local_pathology_vec[pid - 1] = 1.0

        if self.augment:
            rng = np.random.default_rng((self.seed * 1_000_003 + self._epoch) * 1_000_003 + idx)
            image, (kidney_mask, lesion_union_mask) = augment_sample(
                image, [kidney_mask, lesion_union_mask], rng, self.augment_params)

        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5

        return {
            'image': torch.from_numpy(image).permute(2, 0, 1).float(),
            'kidney_mask': torch.from_numpy(kidney_mask).float(),
            'global_diagnosis': torch.tensor(int(row['global_diagnosis']), dtype=torch.float32),
            'global_pathologies': torch.tensor(row['global_pathologies'], dtype=torch.float32),
            'lesion_mask': torch.from_numpy(lesion_union_mask).float(),
            'local_pathologies': torch.from_numpy(local_pathology_vec),
            'has_lesion': torch.tensor(float(lesion_union_mask.max() > 0)),
            'case_id': row['case_id'],
        }


def split_datasets(manifest, **kwargs):
    return (URICADSMultiTaskDataset(manifest[manifest['split'] == s], **kwargs)
            for s in ('train', 'val', 'test'))
