"""Network definitions, kept bit-identical to the notebooks so saved checkpoints still load."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .config import NUM_GLOBAL_PATHOLOGIES, NUM_LOCAL_PATHOLOGY_IDS


ENCODER_NAMES = ('resnet34', 'efficientnet_b0', 'mobilenet_v3_large', 'shufflenet_v2_x1_0')


def build_encoder(name, pretrained=True):
    """Returns (feature extractor, output channels).

    Every encoder here downsamples by 32, which is what the shared decoder expects. The channel
    count differs per backbone (512, 1280, 960, 1024), so it is measured with a dummy forward
    rather than hardcoded - a wrong constant would surface only as a shape error deep in
    training.
    """
    weights = 'IMAGENET1K_V1' if pretrained else None
    if name == 'resnet34':
        m = torchvision.models.resnet34(weights=weights)
        enc = nn.Sequential(*list(m.children())[:-2])
    elif name == 'efficientnet_b0':
        enc = torchvision.models.efficientnet_b0(weights=weights).features
    elif name == 'mobilenet_v3_large':
        enc = torchvision.models.mobilenet_v3_large(weights=weights).features
    elif name == 'shufflenet_v2_x1_0':
        m = torchvision.models.shufflenet_v2_x1_0(weights=weights)
        enc = nn.Sequential(m.conv1, m.maxpool, m.stage2, m.stage3, m.stage4, m.conv5)
    else:
        raise ValueError(f'unknown encoder {name!r}; choose from {ENCODER_NAMES}')

    was_training = enc.training
    enc.eval()
    with torch.no_grad():
        channels = enc(torch.zeros(1, 3, 64, 64)).shape[1]
    enc.train(was_training)
    return enc, int(channels)


def _up_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    )


def _decoder(in_ch=512):
    return nn.Sequential(
        _up_block(in_ch, 256), _up_block(256, 128), _up_block(128, 64),
        _up_block(64, 32), _up_block(32, 16),
        nn.Conv2d(16, 1, kernel_size=1),
    )


class MultiTaskNet(nn.Module):
    """Phase 2 / Phase 3 architecture: diagnosis + global pathology + kidney segmentation."""

    def __init__(self, num_global_pathologies=NUM_GLOBAL_PATHOLOGIES, pretrained=True):
        super().__init__()
        backbone = torchvision.models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.diagnosis_head = nn.Linear(512, 1)
        self.global_pathology_head = nn.Linear(512, num_global_pathologies)
        self.seg_decoder = _decoder()

    def forward(self, x):
        feat = self.encoder(x)
        pooled = self.global_pool(feat).flatten(1)
        return (self.diagnosis_head(pooled).squeeze(1),
                self.global_pathology_head(pooled),
                self.seg_decoder(feat).squeeze(1))


class MultiTaskNetV2(nn.Module):
    """Phase 4 / 5 / 6 architecture: adds the local-pathology head and lesion decoder."""

    def __init__(self, num_global_pathologies=NUM_GLOBAL_PATHOLOGIES,
                 num_local_pathologies=NUM_LOCAL_PATHOLOGY_IDS, pretrained=True):
        super().__init__()
        backbone = torchvision.models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.diagnosis_head = nn.Linear(512, 1)
        self.global_pathology_head = nn.Linear(512, num_global_pathologies)
        self.local_pathology_head = nn.Linear(512, num_local_pathologies)
        self.kidney_decoder = _decoder()
        self.lesion_decoder = _decoder()

    def forward(self, x):
        feat = self.encoder(x)
        pooled = self.global_pool(feat).flatten(1)
        return {
            'diagnosis': self.diagnosis_head(pooled).squeeze(1),
            'global_pathology': self.global_pathology_head(pooled),
            'local_pathology': self.local_pathology_head(pooled),
            'kidney_seg': self.kidney_decoder(feat).squeeze(1),
            'lesion_seg': self.lesion_decoder(feat).squeeze(1),
        }


class MultiTaskNetV3(MultiTaskNetV2):
    """V2 plus lesion-guided pooling for the local-pathology head.

    Under matched 5-fold cross-validation the V2 design reaches 0.751 multi-pathological AUC
    against URI-CADS's published 0.8190 - the one deficit large enough to exceed its own error
    bar. The likely cause is structural: URI-CADS predicts each pathology from an explicitly
    localized region, whereas V2 must infer nine pathology types from a single globally averaged
    feature vector, so a small lesion is averaged together with the whole image.

    This head therefore also pools the encoder features *weighted by the lesion head's own
    predicted probability map*, and classifies from the global and lesion-focused vectors
    concatenated. It reuses the existing decoder rather than adding a detector.

    The attention map is detached, so the local-pathology loss cannot reshape the lesion decoder
    to make its own job easier - lesion segmentation stays supervised only by its Dice term.
    """

    def __init__(self, num_global_pathologies=NUM_GLOBAL_PATHOLOGIES,
                 num_local_pathologies=NUM_LOCAL_PATHOLOGY_IDS, pretrained=True):
        super().__init__(num_global_pathologies, num_local_pathologies, pretrained)
        self.local_pathology_head = nn.Linear(512 * 2, num_local_pathologies)

    def forward(self, x):
        feat = self.encoder(x)
        pooled = self.global_pool(feat).flatten(1)
        kidney_seg = self.kidney_decoder(feat).squeeze(1)
        lesion_seg = self.lesion_decoder(feat).squeeze(1)

        attn = torch.sigmoid(lesion_seg).detach().unsqueeze(1)
        attn = F.adaptive_avg_pool2d(attn, feat.shape[-2:])
        attn = attn / (attn.sum(dim=(2, 3), keepdim=True) + 1e-6)
        lesion_vec = (feat * attn).sum(dim=(2, 3))

        return {
            'diagnosis': self.diagnosis_head(pooled).squeeze(1),
            'global_pathology': self.global_pathology_head(pooled),
            'local_pathology': self.local_pathology_head(
                torch.cat([pooled, lesion_vec], dim=1)),
            'kidney_seg': kidney_seg,
            'lesion_seg': lesion_seg,
        }


class MultiTaskNetFlex(nn.Module):
    """Configurable multi-task network for the ablation and encoder studies.

    Two axes vary here. The encoder can be any backbone in ENCODER_NAMES, and each task module
    can be switched off independently. A disabled head returns None rather than an untrained
    tensor, so a switched-off module cannot quietly contribute a meaningless number to the
    metrics; the evaluation path already treats None as "absent".

    `lesion_guided_pooling` reproduces the V3 head: the local-pathology classifier sees encoder
    features pooled under the lesion decoder's own probability map, in addition to the global
    average. It requires the lesion decoder, since that map is what supplies the attention.
    """

    def __init__(self, encoder='resnet34', pretrained=True,
                 num_global_pathologies=NUM_GLOBAL_PATHOLOGIES,
                 num_local_pathologies=NUM_LOCAL_PATHOLOGY_IDS,
                 use_diagnosis=True, use_global_pathology=True, use_local_pathology=True,
                 use_kidney_decoder=True, use_lesion_decoder=True,
                 lesion_guided_pooling=True):
        super().__init__()
        self.encoder_name = encoder
        self.encoder, ch = build_encoder(encoder, pretrained)
        self.encoder_channels = ch
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.use_diagnosis = use_diagnosis
        self.use_global_pathology = use_global_pathology
        self.use_local_pathology = use_local_pathology
        self.use_kidney_decoder = use_kidney_decoder
        self.use_lesion_decoder = use_lesion_decoder
        self.lesion_guided_pooling = bool(lesion_guided_pooling and use_lesion_decoder)

        self.diagnosis_head = nn.Linear(ch, 1) if use_diagnosis else None
        self.global_pathology_head = (nn.Linear(ch, num_global_pathologies)
                                      if use_global_pathology else None)
        local_in = ch * 2 if self.lesion_guided_pooling else ch
        self.local_pathology_head = (nn.Linear(local_in, num_local_pathologies)
                                     if use_local_pathology else None)
        self.kidney_decoder = _decoder(ch) if use_kidney_decoder else None
        self.lesion_decoder = _decoder(ch) if use_lesion_decoder else None

    def forward(self, x):
        feat = self.encoder(x)
        pooled = self.global_pool(feat).flatten(1)

        kidney_seg = self.kidney_decoder(feat).squeeze(1) if self.kidney_decoder else None
        lesion_seg = self.lesion_decoder(feat).squeeze(1) if self.lesion_decoder else None

        local_logits = None
        if self.local_pathology_head is not None:
            if self.lesion_guided_pooling:
                attn = torch.sigmoid(lesion_seg).detach().unsqueeze(1)
                attn = F.adaptive_avg_pool2d(attn, feat.shape[-2:])
                attn = attn / (attn.sum(dim=(2, 3), keepdim=True) + 1e-6)
                lesion_vec = (feat * attn).sum(dim=(2, 3))
                local_logits = self.local_pathology_head(torch.cat([pooled, lesion_vec], dim=1))
            else:
                local_logits = self.local_pathology_head(pooled)

        return {
            'diagnosis': (self.diagnosis_head(pooled).squeeze(1)
                          if self.diagnosis_head is not None else None),
            'global_pathology': (self.global_pathology_head(pooled)
                                 if self.global_pathology_head is not None else None),
            'local_pathology': local_logits,
            'kidney_seg': kidney_seg,
            'lesion_seg': lesion_seg,
        }
