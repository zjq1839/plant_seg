import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math


class FeatureEnhancementModule(nn.Module):
    """Self-attention based feature enhancement for better lesion representation"""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        B, C, H, W = x.size()
        # Generate query, key, value
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C']
        k = self.key(x).view(B, -1, H * W)  # [B, C', HW]
        v = self.value(x).view(B, -1, H * W)  # [B, C, HW]
        
        # Compute attention
        attention = torch.bmm(q, k)  # [B, HW, HW]
        attention = self.softmax(attention)
        
        # Apply attention to values
        out = torch.bmm(v, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable weight
        return x + self.gamma * out

class SimpleSegHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.enhancement = FeatureEnhancementModule(in_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Conv2d(in_ch, num_classes, 1)

    def forward(self, x):
        x = self.enhancement(x)
        x = self.conv(x)
        return self.cls(x)


class ProjectionHead2D(nn.Module):
    """
    Flexible 2D projection head to map feature maps to CLIP space.
    Supports:
    - head_type 'conv': single 1x1 conv (legacy behavior)
    - head_type 'mlp': 1x1 conv -> norm(optional) -> GELU -> 1x1 conv

    Args:
        in_dim: input channels
        out_dim: output channels (CLIP dim)
        head_type: 'conv' | 'mlp'
        mid_dim: hidden channels for 'mlp' (defaults to out_dim if None)
        norm: 'none' | 'bn' (BatchNorm2d) | 'gn' (GroupNorm)
        dropout: dropout rate between hidden and output (only for 'mlp')
    """
    def __init__(self, in_dim: int, out_dim: int, head_type: str = 'conv', mid_dim: int = None, norm: str = 'none', dropout: float = 0.0):
        super().__init__()
        head_type = (head_type or 'conv').lower()
        mid_dim = mid_dim or out_dim
        self.head_type = head_type
        if head_type == 'conv':
            self.net = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        elif head_type == 'mlp':
            layers = [nn.Conv2d(in_dim, mid_dim, kernel_size=1)]
            n = (norm or 'none').lower()
            if n == 'bn':
                layers.append(nn.BatchNorm2d(mid_dim))
            elif n == 'gn':
                # pick reasonable group count
                g = 32
                while g > 1 and (mid_dim % g != 0):
                    g //= 2
                layers.append(nn.GroupNorm(g, mid_dim))
            layers.append(nn.GELU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))
            layers.append(nn.Conv2d(mid_dim, out_dim, kernel_size=1))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown proj head_type: {head_type}")

    def forward(self, x):
        return self.net(x)


class SegStudent(nn.Module):
    def __init__(self, backbone='tf_efficientnetv2_s_in21k', num_seen_classes=20, feat_dim=256, clip_dim=512, pretrained=True,
                 proj_head: str = 'conv', proj_mid_dim: int = 256, proj_norm: str = 'none', proj_dropout: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(backbone, features_only=True, pretrained=pretrained)
        chs = self.backbone.feature_info.channels()
        # Multi-scale v1 (stride16): fuse f5 (s=32) -> f4 (s=16)
        self.lateral5 = nn.Conv2d(chs[-1], feat_dim, kernel_size=1)
        self.lateral4 = nn.Conv2d(chs[-2], feat_dim, kernel_size=1)
        # Keep a projection conv for compatibility with downstream code expecting `model.proj.out_channels == feat_dim`
        self.proj = nn.Conv2d(feat_dim, feat_dim, kernel_size=1)
        self.seg_head = SimpleSegHead(feat_dim, num_seen_classes)
        # Configurable projection head to project D->C for CLIP feature space alignment
        self.clip_proj_2d = ProjectionHead2D(
            feat_dim, clip_dim, head_type=proj_head, mid_dim=proj_mid_dim, norm=proj_norm, dropout=proj_dropout
        )

    def forward(self, x):
        all_features = self.backbone(x)
        # Expect last two stages: f4 (stride 16), f5 (stride 32)
        f4, f5 = all_features[-2], all_features[-1]
        p5 = self.lateral5(f5)
        p4 = self.lateral4(f4) + F.interpolate(p5, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        feats = self.proj(p4)  # [B, D=feat_dim, H/16, W/16]
        logits = self.seg_head(feats)
        out = {
            'feat': feats,  # [B, D, H, W] with D=feat_dim
            'logits': logits,
        }
        # provide projection [B, C, L] via 2d conv/mlp -> flatten spatial dims
        out['proj'] = self.clip_proj_2d(feats).flatten(2)  # [B, C, L]
        return out


class SegFormerStudent(nn.Module):
    """
    SegFormer-style student using HuggingFace SegformerModel.
    If `pretrained_dir` is provided and exists locally, will load from that directory
    via `SegformerModel.from_pretrained`. Otherwise, if `pretrained` is True,
    attempts to load weights via HuggingFace model id given by `backbone` (best-effort),
    and falls back to a randomly initialized config if that fails.
    """
    def __init__(self, backbone='mit_b0', num_seen_classes=20, feat_dim=256, clip_dim=512, pretrained=True, pretrained_dir: str = None,
                 proj_head: str = 'conv', proj_mid_dim: int = 256, proj_norm: str = 'none', proj_dropout: float = 0.0):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerModel
        except ImportError as e:
            raise ImportError("transformers is required for SegFormerStudent. Please install `transformers`.")
        self.backbone_name = backbone
        self.pretrained_dir = pretrained_dir

        if pretrained_dir and os.path.exists(pretrained_dir):
            # Load full encoder from local directory (no internet needed)
            self.backbone = SegformerModel.from_pretrained(pretrained_dir, ignore_mismatched_sizes=True)
            # Ensure hidden states are returned during forward
            if hasattr(self.backbone, 'config'):
                self.backbone.config.output_hidden_states = True
            config = self.backbone.config
        else:
            # Try to honor `pretrained` flag; otherwise fall back to random init
            if pretrained:
                try:
                    model_id = backbone if isinstance(backbone, str) else 'nvidia/mit-b0'
                    self.backbone = SegformerModel.from_pretrained(model_id, ignore_mismatched_sizes=True)
                    if hasattr(self.backbone, 'config'):
                        self.backbone.config.output_hidden_states = True
                    config = self.backbone.config
                except Exception:
                    config = SegformerConfig()
                    config.output_hidden_states = True
                    self.backbone = SegformerModel(config)
            else:
                config = SegformerConfig()
                config.output_hidden_states = True
                self.backbone = SegformerModel(config)

        # Determine last stage channel dim from config
        last_hidden = config.hidden_sizes[-1] if hasattr(config, 'hidden_sizes') else 256
        self.proj = nn.Conv2d(last_hidden, feat_dim, kernel_size=1)
        self.seg_head = SimpleSegHead(feat_dim, num_seen_classes)
        self.clip_proj_2d = ProjectionHead2D(
            feat_dim, clip_dim, head_type=proj_head, mid_dim=proj_mid_dim, norm=proj_norm, dropout=proj_dropout
        )

    def forward(self, x):
        # SegformerModel expects pixel_values in [B, 3, H, W]
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        hidden_states = getattr(outputs, 'hidden_states', None)

        def _reshape_to_4d(t, H_in, W_in):
            # t: [B, L, C] -> [B, C, h, w]
            B, L, C = t.shape
            # initial guess from input size and stride 32
            h = max(1, H_in // 32)
            w = max(1, W_in // 32)
            if h * w != L:
                # try ceil in case of odd sizes
                h = max(1, int(math.ceil(H_in / 32)))
                w = max(1, int(math.ceil(W_in / 32)))
            if h * w != L:
                # fallback: find factor pair close to square
                s = int(round(L ** 0.5))
                hh, ww = None, None
                for cand in range(max(1, s - 8), s + 9):
                    if L % cand == 0:
                        hh, ww = cand, L // cand
                        break
                if hh is None:
                    # final fallback: assume square
                    hh = ww = int(round(L ** 0.5))
                if hh * ww != L:
                    raise RuntimeError(f"Cannot infer 2D grid from sequence length L={L} (input {H_in}x{W_in}).")
                h, w = hh, ww
            return t.transpose(1, 2).contiguous().view(B, C, h, w)

        feats = None
        if hidden_states is not None and len(hidden_states) > 0:
            last = hidden_states[-1]
            if last.dim() == 4:
                feats = last
            elif last.dim() == 3:
                H_in, W_in = x.shape[-2:]
                feats = _reshape_to_4d(last, H_in, W_in)
            else:
                raise RuntimeError(f"Unexpected SegFormer hidden state dim={last.dim()}.")
        else:
            last = getattr(outputs, 'last_hidden_state', None)
            if last is None:
                raise RuntimeError("SegFormer outputs missing hidden states and last_hidden_state.")
            if last.dim() == 4:
                feats = last
            elif last.dim() == 3:
                H_in, W_in = x.shape[-2:]
                feats = _reshape_to_4d(last, H_in, W_in)
            else:
                raise RuntimeError(f"Unexpected SegFormer last_hidden_state dim={last.dim()}.")

        feats = self.proj(feats)
        logits = self.seg_head(feats)
        out = {
            'feat': feats,
            'logits': logits,
        }
        out['proj'] = self.clip_proj_2d(feats).flatten(2)
        return out


def build_seg_model(cfg, num_seen_classes: int):
    name = cfg.get('name', 'student')
    common_kwargs = dict(
        feat_dim=cfg.get('feat_dim', 256),
        clip_dim=cfg.get('clip_dim', 512),
    )
    # Projection head configs (with safe defaults)
    proj_kwargs = dict(
        proj_head=cfg.get('proj_head', 'conv'),
        proj_mid_dim=cfg.get('proj_mid_dim', cfg.get('clip_dim', 512)),
        proj_norm=cfg.get('proj_norm', 'none'),
        proj_dropout=cfg.get('proj_dropout', 0.0),
    )

    if name == 'student':
        return SegStudent(
            backbone=cfg.get('backbone', 'tf_efficientnetv2_s_in21k'),
            num_seen_classes=num_seen_classes,
            pretrained=cfg.get('pretrained', True),
            **common_kwargs,
            **proj_kwargs,
        )
    elif name == 'segformer':
        # Use transformers-based SegFormer encoder; optionally load local pretrained weights
        return SegFormerStudent(
            backbone=cfg.get('backbone', 'mit_b0'),
            num_seen_classes=num_seen_classes,
            pretrained=cfg.get('pretrained', True),
            pretrained_dir=cfg.get('pretrained_dir', None),
            **common_kwargs,
            **proj_kwargs,
        )
    else:
        raise NotImplementedError(name)