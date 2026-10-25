"""MIL model with per-finding attention (requires ``torch`` and ``timm``).

Each study is a bag of 2.5D windows. A shared ImageNet-pretrained backbone
encodes every window; a slice-position embedding tells the head which slot a
window comes from; 12 independent attention heads let every finding select its
own windows. The attention weights are returned on request, which makes each
prediction inspectable.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_backbone(arch="convnext_tiny", pretrained=True):
    import timm
    return timm.create_model(arch, pretrained=pretrained, num_classes=0, global_pool="avg", in_chans=3)


class PerFindingAttentionHead(nn.Module):
    """K feature vectors of size F -> 12 logits, one attention per class.
    Softmax runs over windows: each class picks its own windows."""

    def __init__(self, f_dim, n_classes=12, hidden=256, drop=0.2, n_pos=64):
        super().__init__()
        self.pos = nn.Embedding(n_pos, f_dim)        # slice position: tells the model which slot it is looking at
        nn.init.normal_(self.pos.weight, std=0.02)
        self.norm = nn.LayerNorm(f_dim)
        self.att = nn.Sequential(nn.Linear(f_dim, hidden), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(hidden, n_classes))
        self.clsW = nn.Parameter(torch.zeros(n_classes, f_dim))
        self.clsb = nn.Parameter(torch.zeros(n_classes))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.drop = nn.Dropout(drop)

    def forward(self, feats, pos, mask=None, return_attention=False):
        # feats (B, K, F)  pos (B, K) slice indices  mask (B, K) bool
        h = self.norm(feats + self.pos(pos))
        a = self.att(h)                                         # (B, K, 12)
        if mask is not None:
            a = a.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        a = torch.softmax(a.float(), dim=1).to(h.dtype)         # sums to 1 over windows
        pooled = torch.einsum("bkn,bkf->bnf", a, self.drop(h))  # (B, 12, F)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return (logits, a) if return_attention else logits


class KneeMILModel(nn.Module):
    def __init__(self, arch="convnext_tiny", n_classes=12, pretrained=True):
        super().__init__()
        self.backbone = build_backbone(arch, pretrained)
        self.head = PerFindingAttentionHead(self.backbone.num_features, n_classes)

    def forward(self, x, pos, mask=None, return_attention=False):
        # x (B, K, 3, H, W) float, already normalised
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1)).view(B, K, -1)
        return self.head(f, pos, mask, return_attention)


def to_input(windows_uint8, device):
    """(B, K, 3, H, W) uint8 -> ImageNet-normalised float on the GPU."""
    x = windows_uint8.to(device, non_blocking=True).float().div_(255.0)
    B, K = x.shape[:2]
    x = x.flatten(0, 1)
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return x.view(B, K, *x.shape[1:])


def gpu_augment(x):
    """Mild augmentations, identical for all windows of a study.
    No horizontal flip (it would swap medial and lateral).
    x : (B, K, 3, H, W) float."""
    B, K, C, H, W = x.shape
    dev = x.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * np.deg2rad(8)
    sc = 1 + (torch.rand(B, device=dev) * 2 - 1) * 0.10
    tx = (torch.rand(B, device=dev) * 2 - 1) * 0.05
    ty = (torch.rand(B, device=dev) * 2 - 1) * 0.05
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)  # (B, 2, 3)
    theta = theta.repeat_interleave(K, 0)
    flat = x.flatten(0, 1)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    flat = F.grid_sample(flat, grid, align_corners=False, padding_mode="zeros")
    contrast = (1 + (torch.rand(B, device=dev) * 2 - 1) * 0.15).repeat_interleave(K).view(-1, 1, 1, 1)
    bright = ((torch.rand(B, device=dev) * 2 - 1) * 0.15).repeat_interleave(K).view(-1, 1, 1, 1)
    flat = flat * contrast + bright
    return flat.view(B, K, C, H, W)


@torch.no_grad()
def predict_volume(model, vol, mask, device, centers_fn, max_windows=64):
    """Predict 12 probabilities for a (64, H, W) uint8 volume using all valid windows."""
    centers = centers_fn(mask)
    if len(centers) == 0:
        return None
    centers = centers[:max_windows]
    win = np.stack([vol[c - 1:c + 2] for c in centers])[None]          # (1, K, 3, H, W)
    x = to_input(torch.from_numpy(win), device)
    pos = torch.from_numpy(centers)[None].to(device)
    try:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x, pos)
    except RuntimeError:
        logits = model(x.float(), pos)
    return torch.sigmoid(logits.float())[0].cpu().numpy()
