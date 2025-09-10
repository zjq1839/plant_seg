import torch
import torch.nn as nn
import torch.nn.functional as F
import clip  # use OpenAI CLIP API
from typing import List, Dict, Any
import re

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
        # Ensure spatial size matches CLIP's expected resolution
        if images.shape[-1] != self.n_px or images.shape[-2] != self.n_px:
            images = F.interpolate(images, size=(self.n_px, self.n_px), mode='bilinear', align_corners=False)
        # Normalize to CLIP stats; images are float in [0,1]
        mean = CLIP_MEAN.to(images.device, dtype=images.dtype)[None, :, None, None]
        std = CLIP_STD.to(images.device, dtype=images.dtype)[None, :, None, None]
        x = (images - mean) / std
        pooled = self.model.encode_image(x)
        # Cast to match images dtype (usually float32)
        if pooled.dtype != images.dtype:
            pooled = pooled.to(images.dtype)
        # In OpenAI CLIP, encode_image returns [B, embed_dim]
        cls_token = pooled
        dense_tokens = None  # not available from clip.encode_image
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
        # update bank with current global cls
        self.enqueue_cls(Cg)

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
            bank = self.bank[:self.bank_filled]  # [K, C]
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