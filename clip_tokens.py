import torch
import torch.nn as nn
import torch.nn.functional as F
import clip  # use OpenAI CLIP API
from typing import List, Dict, Any
import re
import os
import math
from clip.model import VisionTransformer

from loss import info_nce_loss


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def _map_clip_name(name: str) -> str:
    # Normalize various ViT naming styles to OpenAI CLIP keys, case-insensitively.
    # Examples accepted:
    #  - "ViT-B-16" -> "ViT-B/16"
    #  - "ViT-L-14" -> "ViT-L/14"
    #  - "vit-l/14"  -> "ViT-L/14"
    #  - "vit-l-14@336px" -> "ViT-L/14@336px"
    if not isinstance(name, str):
        return name
    s = name.strip()
    m = re.match(r"(?i)vit[-_/]?([blh])[-_/]?(\d+)(@336px)?", s)
    if m:
        size = m.group(1).upper()
        patch = m.group(2)
        suffix = m.group(3) or ""
        return f"ViT-{size}/{patch}{suffix}"
    # Backward-compatible simple replacements
    if 'ViT-' in s and any(t in s for t in ['-B-', '-L-', '-H-', '-G-']):
        return s.replace('-B-', '-B/').replace('-L-', '-L/').replace('-H-', '-H/').replace('-G-', '-G/')
    return s


class CLIPTeacher:
    def __init__(self, model_name='ViT-B-16', pretrained='openai', context_length=77, device='cuda', bank_size=24, temperature=0.07, num_seen=20):
        self.device = device
        model_id = _map_clip_name(model_name)
        self.model, self.preprocess = clip.load(model_id, device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # expose tokenizer interface for compatibility
        self.tokenize = clip.tokenize
        self.bank_size = bank_size
        self.tau = temperature
        self.num_seen = num_seen
        C = self.model.text_projection.shape[1]
        self.C = C
        # target input resolution for CLIP visual encoder
        self.n_px = int(getattr(self.model.visual, 'input_resolution', 224))
        # CLS token bank (queue) for InfoNCE (global)
        self.register_bank(C)
        
        # Initialize text embeddings for plant lesion segmentation
        self.text_embeddings = self._init_text_embeddings()

    def register_bank(self, C):
        self.bank = torch.zeros(self.bank_size, C, device=self.device)
        self.bank_ptr = 0
        self.bank_filled = 0

    @torch.no_grad()
    def encode_image_dense(self, images: torch.Tensor):
        # Resize to CLIP visual resolution
        if images.shape[-2] != self.n_px or images.shape[-1] != self.n_px:
            images = F.interpolate(images, size=(self.n_px, self.n_px), mode='bilinear', align_corners=False)
        # Normalize to CLIP stats; images are float in [0,1]
        mean = CLIP_MEAN.to(images.device, dtype=images.dtype)[None, :, None, None]
        std = CLIP_STD.to(images.device, dtype=images.dtype)[None, :, None, None]
        x_in = (images - mean) / std

        v = self.model.visual
        dense_tokens = None
        try:
            # Vision Transformer path: reconstruct tokens before pooling
            if isinstance(v, VisionTransformer):
                x = x_in.type(v.conv1.weight.dtype)
                x = v.conv1(x)  # [B, width, gh, gw]
                B, width, gh, gw = x.shape
                x = x.reshape(B, width, gh * gw).permute(0, 2, 1)  # [B, N, width]
                cls_tok = v.class_embedding.to(x.dtype) + torch.zeros(B, 1, width, dtype=x.dtype, device=x.device)
                x = torch.cat([cls_tok, x], dim=1)  # [B, 1+N, width]
                x = x + v.positional_embedding.to(x.dtype)
                x = v.ln_pre(x)
                x = x.permute(1, 0, 2)  # [1+N, B, width]
                x = v.transformer(x)
                x = x.permute(1, 0, 2)  # [B, 1+N, width]
                x_cls = x[:, 0, :]
                x_patch = x[:, 1:, :]  # [B, N, width]
                # Apply ln_post to both global and patch tokens
                x_cls = v.ln_post(x_cls)
                x_patch = v.ln_post(x_patch)
                # Project to embed_dim if proj exists
                if getattr(v, 'proj', None) is not None:
                    x_cls = x_cls @ v.proj
                    x_patch = x_patch @ v.proj
                cls_token = x_cls.to(dtype=images.dtype)
                C = x_patch.shape[-1]
                dense_tokens = x_patch.transpose(1, 2).reshape(B, C, gh, gw).contiguous().to(dtype=images.dtype)
            else:
                # Non-ViT (e.g., ResNet) fallback: only global token available
                pooled = self.model.encode_image(x_in)
                cls_token = pooled.to(dtype=images.dtype)
        except Exception:
            pooled = self.model.encode_image(x_in)
            cls_token = pooled.to(dtype=images.dtype)
            dense_tokens = None

        return cls_token, dense_tokens

    @torch.no_grad()
    def _init_text_embeddings(self):
        """
        Initialize text embeddings for plant lesion segmentation task.
        Returns:
            torch.Tensor: [2, C] normalized text embeddings for background and lesion
        """
        classnames = ["background", "plant disease lesion"]
        prompts = [f"a photo of {name}." for name in classnames]
        tokens = self.tokenize(prompts).to(self.device)
        text_feat = self.model.encode_text(tokens)
        text_feat = text_feat.to(dtype=torch.float32)
        text_feat = F.normalize(text_feat, dim=-1)
        return text_feat
    
    @torch.no_grad()
    def get_text_embeddings(self):
        """
        Get pre-computed text embeddings for plant lesion segmentation.
        Returns:
            torch.Tensor: [2, C] normalized text embeddings for background and lesion
        """
        return self.text_embeddings

    @torch.no_grad()
    def forward_tokens_and_pseudo(self, images: torch.Tensor, labels: torch.Tensor, present_seen_ids: List[torch.Tensor]):
        B, _, H, W = images.shape
        Cg, _ = self.encode_image_dense(images)
        # build pseudo masks: start from GT seen, set other pixels as latent id = num_seen (single latent id)
        Yp = labels.clone()
        latent_id = self.num_seen  # unseen collapsed
        Yp[(Yp < 0) | (Yp >= self.num_seen)] = latent_id
        # Local CLS tokens: per sample, for seen present classes, crop masked image and obtain local cls
        Cl = []
        present_ids = []
        for b in range(B):
            ids = present_seen_ids[b]
            if ids.numel() == 0:
                Cl.append(torch.empty(0, Cg.shape[-1], device=images.device))
                present_ids.append(torch.empty(0, dtype=torch.long, device=images.device))
                continue
            tokens_b = []
            for cls_id in ids.tolist():
                mask = (labels[b] == cls_id).float()
                mask_pixels = mask.sum()
                # For background class, use a minimum threshold; for lesion class, use stricter threshold
                min_pixels = 10 if cls_id == 0 else 3
                if mask_pixels < min_pixels:
                    continue
                img_b = images[b:b+1]
                # For background, use original image; for lesion, use masked image
                if cls_id == 0:
                    # For background, apply inverse mask to focus on background regions
                    inv_mask = 1.0 - (labels[b] == 1).float()
                    masked = img_b * inv_mask.unsqueeze(0)
                else:
                    # For lesion, use standard masking
                    masked = img_b * mask.unsqueeze(0)
                c_local, _ = self.encode_image_dense(masked)
                tokens_b.append(c_local[0])
            if len(tokens_b) == 0:
                Cl.append(torch.empty(0, Cg.shape[-1], device=images.device))
                present_ids.append(torch.empty(0, dtype=torch.long, device=images.device))
            else:
                Cl.append(torch.stack(tokens_b, dim=0))
                present_ids.append(ids[:len(tokens_b)])
        # NOTE: do NOT update bank here during training step to avoid in-place bumps before backward
        return {"Cg": Cg, "Cl": Cl, "Yp": Yp, "present_ids": present_ids}

    def enqueue_cls(self, cls_tokens: torch.Tensor):
        for i in range(cls_tokens.shape[0]):
            self.bank[self.bank_ptr] = F.normalize(cls_tokens[i], dim=-1)
            self.bank_ptr = (self.bank_ptr + 1) % self.bank_size
            self.bank_filled = min(self.bank_filled + 1, self.bank_size)

    def info_nce_global(self, Fg: torch.Tensor, Cg_pos: torch.Tensor):
        # Enhanced global distillation with text-visual alignment
        Fg = F.normalize(Fg, dim=-1)
        Cg_pos = F.normalize(Cg_pos, dim=-1)
        
        # Standard visual-visual contrastive loss
        if self.bank_filled == 0:
            logits = (Fg @ Cg_pos.t()) / self.tau
            targets = torch.arange(Fg.size(0), device=Fg.device)
            visual_loss = F.cross_entropy(logits, targets)
        else:
            bank = self.bank[:self.bank_filled].detach().clone()  # [K, C] snapshot to avoid in-place version bump
            pos = (Fg @ Cg_pos.t()).diag().unsqueeze(1)  # [B,1]
            neg = Fg @ bank.t()  # [B,K]
            logits = torch.cat([pos, neg], dim=1) / self.tau
            targets = torch.zeros(Fg.size(0), dtype=torch.long, device=Fg.device)
            visual_loss = F.cross_entropy(logits, targets)
        
        # Additional text-visual alignment loss for enhanced semantic understanding
        text_embeds = self.get_text_embeddings()  # [2, C] for background and lesion
        # Compute similarity between global features and text embeddings
        text_sim = Fg @ text_embeds.t()  # [B, 2]
        # Use softmax to get probability distribution over classes
        text_probs = F.softmax(text_sim / self.tau, dim=-1)
        # Encourage diversity in text-visual alignment (entropy regularization)
        text_entropy = -torch.sum(text_probs * torch.log(text_probs + 1e-8), dim=-1).mean()
        text_alignment_loss = -text_entropy  # Maximize entropy for better generalization
        
        # Combine losses with adaptive weighting
        total_loss = visual_loss + 0.1 * text_alignment_loss
        return total_loss

# =====================
# DINOv2 Teacher (timm)
# =====================
try:
    import timm
    from timm.data import resolve_data_config
except Exception:
    timm = None
    resolve_data_config = None


class DinoTeacher:
    """Image-only teacher built on timm's DINOv2 backbones.
    API matches CLIPTeacher: encode_image_dense, forward_tokens_and_pseudo, info_nce_global, and attributes.
    """
    def __init__(self, model_name: str = 'vit_base_patch14_dinov2.lvd142m', device: str = 'cuda', bank_size: int = 24, temperature: float = 0.07, num_seen: int = 20):
        assert timm is not None, "timm is required for DinoTeacher but not available"
        self.device = device
        self.model = timm.create_model(model_name, pretrained=True).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # Temperature and bank
        self.bank_size = bank_size
        self.tau = temperature
        self.num_seen = num_seen
        # Resolve data config for normalization and input size
        if resolve_data_config is not None:
            # Some timm versions expect args (dict) as first parameter, model given by keyword.
            # Passing the model positionally will bind to `args` and break (no .get on model).
            try:
                cfg = resolve_data_config({}, model=self.model)
            except TypeError:
                # Fallback: older API that may accept only keyword
                cfg = resolve_data_config(model=self.model)
            # cfg['input_size'] is (C,H,W)
            self.input_size = cfg.get('input_size', (3, 224, 224))
            self.mean = torch.tensor(cfg.get('mean', (0.485, 0.456, 0.406)))
            self.std = torch.tensor(cfg.get('std', (0.229, 0.224, 0.225)))
        else:
            self.input_size = (3, 224, 224)
            self.mean = torch.tensor((0.485, 0.456, 0.406))
            self.std = torch.tensor((0.229, 0.224, 0.225))
        self.n_px = int(self.input_size[-1])
        # Feature dim
        self.C = getattr(self.model, 'num_features', None)
        if self.C is None:
            # Fallback by doing one dummy forward of zeros
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.n_px, self.n_px, device=self.device)
                pooled = self._encode_global(dummy)
                self.C = pooled.shape[-1]
        # Bank
        self.register_bank(self.C)

    def register_bank(self, C):
        self.bank = torch.zeros(self.bank_size, C, device=self.device)
        self.bank_ptr = 0
        self.bank_filled = 0

    @torch.no_grad()
    def _encode_global(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(images)
        # Prefer standardized pre-logits via forward_head
        try:
            pooled = self.model.forward_head(feats, pre_logits=True)
        except Exception:
            # Heuristics
            if isinstance(feats, dict):
                if 'x_norm_clstoken' in feats:
                    pooled = feats['x_norm_clstoken']
                elif 'cls_token' in feats:
                    pooled = feats['cls_token']
                elif 'x' in feats and feats['x'] is not None:
                    x = feats['x']
                    pooled = x[:, 0] if x.ndim == 3 else x
                else:
                    # Take the first tensor value
                    pooled = next(v for v in feats.values() if isinstance(v, torch.Tensor) and v.ndim >= 2)
                    if pooled.ndim == 3:
                        pooled = pooled[:, 0]
            else:
                # Tensor output
                if feats.ndim == 3:
                    pooled = feats[:, 0]
                elif feats.ndim == 4:
                    pooled = feats.mean(dim=(2, 3))
                else:
                    pooled = feats
        return pooled

    @torch.no_grad()
    def encode_image_dense(self, images: torch.Tensor):
        # Resize
        if images.shape[-2:] != (self.n_px, self.n_px):
            images = F.interpolate(images, size=(self.n_px, self.n_px), mode='bilinear', align_corners=False)
        # Normalize to timm cfg stats
        mean = self.mean.to(images.device, dtype=images.dtype)[None, :, None, None]
        std = self.std.to(images.device, dtype=images.dtype)[None, :, None, None]
        x = (images - mean) / std
        # Forward once to get both global and dense tokens
        feats = self.model.forward_features(x)
        # Global pooled (CLS/pre-logits) with robust fallback
        try:
            pooled = self.model.forward_head(feats, pre_logits=True)
        except Exception:
            if isinstance(feats, dict):
                if 'x_norm_clstoken' in feats:
                    pooled = feats['x_norm_clstoken']
                elif 'cls_token' in feats:
                    pooled = feats['cls_token']
                elif 'x' in feats and feats['x'] is not None:
                    xx = feats['x']
                    pooled = xx[:, 0] if xx.ndim == 3 else xx
                else:
                    pooled = next(v for v in feats.values() if isinstance(v, torch.Tensor) and v.ndim >= 2)
                    if pooled.ndim == 3:
                        pooled = pooled[:, 0]
            else:
                if feats.ndim == 3:
                    pooled = feats[:, 0]
                elif feats.ndim == 4:
                    pooled = feats.mean(dim=(2, 3))
                else:
                    pooled = feats
        cls_token = pooled.to(dtype=images.dtype)

        # Dense patch tokens -> grid [B, C, Ht, Wt]
        dense_tokens = None
        try:
            patch_tokens = None
            if isinstance(feats, dict):
                if 'x_norm_patchtokens' in feats and feats['x_norm_patchtokens'] is not None:
                    patch_tokens = feats['x_norm_patchtokens']  # [B, N, C]
                elif 'x' in feats and feats['x'] is not None and feats['x'].ndim == 3:
                    x_all = feats['x']  # [B, 1+N, C]
                    if x_all.size(1) > 1:
                        patch_tokens = x_all[:, 1:, :]
                elif 'tokens' in feats and feats['tokens'] is not None and feats['tokens'].ndim == 3:
                    x_all = feats['tokens']
                    patch_tokens = x_all[:, 1:, :] if x_all.size(1) > 1 else x_all
            else:
                if feats.ndim == 3:
                    patch_tokens = feats[:, 1:, :] if feats.size(1) > 1 else feats  # [B, N, C]
                elif feats.ndim == 4:
                    # CNN-like [B, C, H, W] -> flatten to [B, N, C]
                    Bc, Cc, Hc, Wc = feats.shape
                    patch_tokens = feats.permute(0, 2, 3, 1).reshape(Bc, Hc * Wc, Cc)
            if patch_tokens is not None:
                Bp, Np, Cp = patch_tokens.shape
                # Resolve grid size from model if available
                gh = gw = None
                try:
                    if hasattr(self.model, 'patch_embed'):
                        grid_size = getattr(self.model.patch_embed, 'grid_size', None)
                        if grid_size is not None:
                            if isinstance(grid_size, (tuple, list)):
                                gh, gw = int(grid_size[0]), int(grid_size[1])
                            else:
                                gh = gw = int(grid_size)
                except Exception:
                    gh = gw = None
                if gh is None or gw is None:
                    s = int(math.sqrt(Np))
                    if s * s == Np:
                        gh = gw = s
                    else:
                        # Fallback based on patch size if present
                        try:
                            ps = getattr(self.model.patch_embed, 'patch_size', (14, 14))
                            if isinstance(ps, (tuple, list)):
                                gh = int(self.n_px // ps[0])
                                gw = int(self.n_px // ps[1])
                            else:
                                gh = gw = int(self.n_px // ps)
                        except Exception:
                            gh = gw = s if s > 0 else Np
                # Reshape to grid [B, C, H, W]
                dense_tokens = patch_tokens.transpose(1, 2).reshape(Bp, Cp, gh, gw).contiguous().to(dtype=images.dtype)
        except Exception:
            dense_tokens = None

        return cls_token, dense_tokens

    @torch.no_grad()
    def forward_tokens_and_pseudo(self, images: torch.Tensor, labels: torch.Tensor, present_seen_ids: List[torch.Tensor]):
        B, _, H, W = images.shape
        Cg, _ = self.encode_image_dense(images)
        # Pseudo labels: same strategy as CLIP teacher
        Yp = labels.clone()
        latent_id = self.num_seen
        Yp[(Yp < 0) | (Yp >= self.num_seen)] = latent_id
        # Local tokens via masked crops
        Cl = []
        present_ids = []
        for b in range(B):
            ids = present_seen_ids[b]
            if ids.numel() == 0:
                Cl.append(torch.empty(0, Cg.shape[-1], device=images.device))
                present_ids.append(torch.empty(0, dtype=torch.long, device=images.device))
                continue
            tokens_b = []
            for cls_id in ids.tolist():
                mask = (labels[b] == cls_id).float()
                if mask.sum() < (10 if cls_id == 0 else 3):
                    continue
                img_b = images[b:b+1]
                if cls_id == 0:
                    inv_mask = 1.0 - (labels[b] == 1).float()
                    masked = img_b * inv_mask.unsqueeze(0)
                else:
                    masked = img_b * mask.unsqueeze(0)
                c_local, _ = self.encode_image_dense(masked)
                tokens_b.append(c_local[0])
            if len(tokens_b) == 0:
                Cl.append(torch.empty(0, Cg.shape[-1], device=images.device))
                present_ids.append(torch.empty(0, dtype=torch.long, device=images.device))
            else:
                Cl.append(torch.stack(tokens_b, dim=0))
                present_ids.append(ids[:len(tokens_b)])
        # IMPORTANT: do NOT update bank here to avoid inplace modifications before backward
        # self.enqueue_cls(Cg)
        return {"Cg": Cg, "Cl": Cl, "Yp": Yp, "present_ids": present_ids}

    def enqueue_cls(self, cls_tokens: torch.Tensor):
        for i in range(cls_tokens.shape[0]):
            self.bank[self.bank_ptr] = F.normalize(cls_tokens[i], dim=-1)
            self.bank_ptr = (self.bank_ptr + 1) % self.bank_size
            self.bank_filled = min(self.bank_filled + 1, self.bank_size)

    def info_nce_global(self, Fg: torch.Tensor, Cg_pos: torch.Tensor):
        Fg = F.normalize(Fg, dim=-1)
        Cg_pos = F.normalize(Cg_pos, dim=-1)
        if self.bank_filled == 0:
            logits = (Fg @ Cg_pos.t()) / self.tau
            targets = torch.arange(Fg.size(0), device=Fg.device)
            loss = F.cross_entropy(logits, targets)
        else:
            # Take a snapshot of the bank to prevent version bump during backward
            bank = self.bank[:self.bank_filled].detach().clone()
            pos = (Fg @ Cg_pos.t()).diag().unsqueeze(1)
            neg = Fg @ bank.t()
            logits = torch.cat([pos, neg], dim=1) / self.tau
            targets = torch.zeros(Fg.size(0), dtype=torch.long, device=Fg.device)
            loss = F.cross_entropy(logits, targets)
        return loss


class MoCoTeacher:
    """Image-only teacher built on timm backbones pretrained with MoCo v3 (or similar self-supervised).
    API matches CLIPTeacher/DinoTeacher: encode_image_dense, forward_tokens_and_pseudo, info_nce_global, and attributes.
    """
    def __init__(self, model_name: str = 'vit_base_patch16_224', device: str = 'cuda', bank_size: int = 24, temperature: float = 0.07, num_seen: int = 20, pretrained: bool = True, checkpoint: str = None):
        assert timm is not None, "timm is required for MoCoTeacher but not available"
        self.device = device
        # Build backbone with optional pretrained flag and robust fallback
        try:
            self.model = timm.create_model(model_name, pretrained=pretrained).to(device)
        except Exception as e:
            # If model name includes an unsupported pretrained tag (e.g., '.mocov3'),
            # fall back to the base architecture without the tag.
            base_name = model_name.split('.')[0] if '.' in model_name else model_name
            try:
                self.model = timm.create_model(base_name, pretrained=False).to(device)
                if base_name != model_name:
                    print(f"[MoCoTeacher] Falling back to base arch '{base_name}' from '{model_name}' (invalid pretrained tag).")
            except Exception as e2:
                # Re-raise with clearer context
                raise RuntimeError(f"Failed to create timm model for MoCoTeacher with names '{model_name}' and base '{base_name}': {e2}")
        # Optional checkpoint loading (local path)
        if checkpoint and os.path.isfile(checkpoint):
            try:
                state = torch.load(checkpoint, map_location='cpu')
                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                self.model.load_state_dict(state, strict=False)
            except Exception:
                pass
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # Temperature and bank
        self.bank_size = bank_size
        self.tau = temperature
        self.num_seen = num_seen
        # Resolve data config for normalization and input size (reuse timm utilities)
        if resolve_data_config is not None:
            try:
                cfg = resolve_data_config({}, model=self.model)
            except TypeError:
                cfg = resolve_data_config(model=self.model)
            self.input_size = cfg.get('input_size', (3, 224, 224))
            self.mean = torch.tensor(cfg.get('mean', (0.485, 0.456, 0.406)))
            self.std = torch.tensor(cfg.get('std', (0.229, 0.224, 0.225)))
        else:
            self.input_size = (3, 224, 224)
            self.mean = torch.tensor((0.485, 0.456, 0.406))
            self.std = torch.tensor((0.229, 0.224, 0.225))
        self.n_px = int(self.input_size[-1])
        # Feature dim inference
        self.C = getattr(self.model, 'num_features', None)
        if self.C is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.n_px, self.n_px, device=self.device)
                pooled = self._encode_global(dummy)
                self.C = pooled.shape[-1]
        # Bank
        self.register_bank(self.C)

    def register_bank(self, C):
        self.bank = torch.zeros(self.bank_size, C, device=self.device)
        self.bank_ptr = 0
        self.bank_filled = 0

    @torch.no_grad()
    def _encode_global(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(images)
        try:
            pooled = self.model.forward_head(feats, pre_logits=True)
        except Exception:
            if isinstance(feats, dict):
                if 'x_norm_clstoken' in feats:
                    pooled = feats['x_norm_clstoken']
                elif 'cls_token' in feats:
                    pooled = feats['cls_token']
                elif 'x' in feats and feats['x'] is not None:
                    x = feats['x']
                    pooled = x[:, 0] if x.ndim == 3 else x
                else:
                    pooled = next(v for v in feats.values() if isinstance(v, torch.Tensor) and v.ndim >= 2)
                    if pooled.ndim == 3:
                        pooled = pooled[:, 0]
            else:
                if feats.ndim == 3:
                    pooled = feats[:, 0]
                elif feats.ndim == 4:
                    pooled = feats.mean(dim=(2, 3))
                else:
                    pooled = feats
        return pooled

    @torch.no_grad()
    def encode_image_dense(self, images: torch.Tensor):
        # Resize
        if images.shape[-2:] != (self.n_px, self.n_px):
            images = F.interpolate(images, size=(self.n_px, self.n_px), mode='bilinear', align_corners=False)
        # Normalize to timm cfg stats
        mean = self.mean.to(images.device, dtype=images.dtype)[None, :, None, None]
        std = self.std.to(images.device, dtype=images.dtype)[None, :, None, None]
        x = (images - mean) / std
        pooled = self._encode_global(x).to(dtype=images.dtype)
        cls_token = pooled  # [B, C]
        dense_tokens = None
        return cls_token, dense_tokens

    @torch.no_grad()
    def forward_tokens_and_pseudo(self, images: torch.Tensor, labels: torch.Tensor, present_seen_ids: List[torch.Tensor]):
        B, _, H, W = images.shape
        Cg, _ = self.encode_image_dense(images)
        # Pseudo labels: same strategy as CLIP/DINO teacher
        Yp = labels.clone()
        latent_id = self.num_seen
        Yp[(Yp < 0) | (Yp >= self.num_seen)] = latent_id
        # Local tokens via masked crops
        Cl = []
        present_ids = []
        for b in range(B):
            ids = present_seen_ids[b]
            if ids.numel() == 0:
                Cl.append(torch.empty(0, Cg.shape[-1], device=images.device))
                present_ids.append(torch.empty(0, dtype=torch.long, device=images.device))
                continue
            tokens_b = []
            for cls_id in ids.tolist():
                mask = (labels[b] == cls_id).float()
                if mask.sum() < (10 if cls_id == 0 else 3):
                    continue
                img_b = images[b:b+1]
                if cls_id == 0:
                    inv_mask = 1.0 - (labels[b] == 1).float()
                    masked = img_b * inv_mask.unsqueeze(0)
                else:
                    masked = img_b * mask.unsqueeze(0)
                c_local, _ = self.encode_image_dense(masked)
                tokens_b.append(c_local[0])
            if len(tokens_b) == 0:
                Cl.append(torch.empty(0, Cg.shape[-1], device=images.device))
                present_ids.append(torch.empty(0, dtype=torch.long, device=images.device))
            else:
                Cl.append(torch.stack(tokens_b, dim=0))
                present_ids.append(ids[:len(tokens_b)])
        self.enqueue_cls(Cg)
        return {"Cg": Cg, "Cl": Cl, "Yp": Yp, "present_ids": present_ids}

    def enqueue_cls(self, cls_tokens: torch.Tensor):
        for i in range(cls_tokens.shape[0]):
            self.bank[self.bank_ptr] = F.normalize(cls_tokens[i], dim=-1)
            self.bank_ptr = (self.bank_ptr + 1) % self.bank_size
            self.bank_filled = min(self.bank_filled + 1, self.bank_size)

    def info_nce_global(self, Fg: torch.Tensor, Cg_pos: torch.Tensor):
        Fg = F.normalize(Fg, dim=-1)
        Cg_pos = F.normalize(Cg_pos, dim=-1)
        if self.bank_filled == 0:
            logits = (Fg @ Cg_pos.t()) / self.tau
            targets = torch.arange(Fg.size(0), device=Fg.device)
            loss = F.cross_entropy(logits, targets)
        else:
            bank = self.bank[:self.bank_filled]
            pos = (Fg @ Cg_pos.t()).diag().unsqueeze(1)
            neg = Fg @ bank.t()
            logits = torch.cat([pos, neg], dim=1) / self.tau
            targets = torch.zeros(Fg.size(0), dtype=torch.long, device=Fg.device)
            loss = F.cross_entropy(logits, targets)
        return loss