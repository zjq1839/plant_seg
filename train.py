import os
import yaml
import math
import random
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from loguru import logger

import timm

from utils import seed_everything, AverageMeter
from datasets import build_dataset
from datasets import seg_collate
from models import build_seg_model
from loss import info_nce_loss
from clip_tokens import CLIPTeacher


def build_optimizer(model, lr, weight_decay):
    params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        wd = weight_decay if p.dim() > 1 else 0.0
        params.append({"params": p, "weight_decay": wd})
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    return opt


def poly_lr(base_lr, cur_iter, max_iter, power=0.9):
    return base_lr * ((1 - cur_iter / max_iter) ** power)


def adjust_lr(optimizer, base_lr, cur_iter, max_iter):
    lr = poly_lr(base_lr, cur_iter, max_iter)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr

# ===== VOC class metadata (ids align with mask values; 0=background, 1..20=foreground) =====
VOC_CLASS_NAMES = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "potted-plant", "sheep", "sofa", "train", "tv/monitor",
]
VOC_FG_IDS = list(range(1, 21))


def train_one_epoch(model, teacher: CLIPTeacher, loader, optimizer, device, epoch, max_iters, cfg, rank=0):
    model.train()
    meters = {k: AverageMeter() for k in ["loss", "loss_global", "loss_local", "loss_ce"]}

    iters_done = epoch * len(loader)
    for it, batch in enumerate(loader):
        imgs, labels, class_labels = batch  # labels: HxW, class_labels: list of present seen classes
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        cur_iter = iters_done + it
        lr = adjust_lr(optimizer, cfg["train"]["lr"], cur_iter, max_iters)

        optimizer.zero_grad(set_to_none=True)

        # 1) CLIP teacher tokens and pseudo masks
        with torch.no_grad():
            teacher_out = teacher.forward_tokens_and_pseudo(imgs, labels, class_labels)
            Cg = teacher_out["Cg"]  # [B, C]
            Cl = teacher_out["Cl"]   # list of tensors [n_present_seen, C] per sample
            present_ids = teacher_out["present_ids"]  # list of tensors with seen class ids order-aligned with Cl
            Yp = teacher_out["Yp"]   # [B, H, W] pseudo labels for latent and seen

        # 2) Student dense features and logits for seen classes
        out = model(imgs)  # dict: {"feat": [B, D, H, W], "logits": [B, S, H, W], "proj": [B, C, L]}
        F_dense = out["feat"]
        S_logits = out["logits"]

        B, D, H, W = F_dense.shape
        L = H * W
        F_flat = F_dense.flatten(2)  # [B, D, L]

        # Global distillation: features already projected to CLIP space as [B, C, L]
        Cg_norm = F.normalize(Cg, dim=-1)  # [B, C]
        F_proj = out["proj"]  # [B, C, L]
        F_proj = F.normalize(F_proj, dim=1)

        # similarity weights
        W_attn = torch.softmax(torch.einsum("bc,bcl->bl", Cg_norm, F_proj) / math.sqrt(F_proj.shape[1]), dim=-1)
        Fg = torch.einsum("bcl,bl->bc", F_proj, W_attn)  # global prototype in CLIP space

        # InfoNCE with bank (use current Cg as positives)
        loss_global = teacher.info_nce_global(Fg, Cg)

        # Local distillation: class-wise prototypes using pseudo labels
        # Resize Yp to match feature map resolution
        Yp_resized = F.interpolate(Yp.float().unsqueeze(1), size=(H, W), mode='nearest').squeeze(1).long()
        
        loss_local = 0.0
        for b in range(B):
            yb = Yp_resized[b]
            fb = F_proj[b]  # [C, L]
            cl_tokens = Cl[b]
            pids = present_ids[b]
            if cl_tokens is None or (torch.is_tensor(cl_tokens) and cl_tokens.shape[0] == 0):
                continue
            cls_feats = []
            cls_tokens = []
            for i, cls_id in enumerate(pids.tolist()):
                mask = (yb == cls_id).flatten()  # [L]
                cnt = mask.sum()
                if cnt < 4:
                    continue
                fl = fb[:, mask].mean(dim=1)  # [C]
                cls_feats.append(fl)
                cls_tokens.append(cl_tokens[i])
            if len(cls_feats) and len(cls_tokens):
                f_stack = torch.stack(cls_feats, dim=0)
                c_stack = torch.stack(cls_tokens, dim=0)
                loss_local += info_nce_loss(f_stack, c_stack, temperature=teacher.tau)
        loss_local = loss_local / max(B, 1)

        # Cross-entropy on seen classes using Yp where label < S
        S = S_logits.shape[1]
        ce_mask = (Yp_resized >= 0) & (Yp_resized < S)
        if ce_mask.any():
            # Reshape for cross entropy: [B*H*W, S] and [B*H*W]
            S_logits_flat = S_logits.permute(0, 2, 3, 1).reshape(-1, S)  # [B*H*W, S]
            Yp_flat = Yp_resized.reshape(-1)  # [B*H*W]
            ce_mask_flat = ce_mask.reshape(-1)  # [B*H*W]
            loss_ce = F.cross_entropy(S_logits_flat[ce_mask_flat], Yp_flat[ce_mask_flat], reduction='mean')
        else:
            loss_ce = torch.tensor(0.0, device=device)

        loss = cfg["loss"]["w_global"] * loss_global + cfg["loss"]["w_local"] * loss_local + cfg["loss"]["w_ce"] * loss_ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 1.0))
        optimizer.step()

        meters["loss"].update(loss.item())
        meters["loss_global"].update(loss_global.item() if torch.is_tensor(loss_global) else loss_global)
        meters["loss_local"].update(float(loss_local))
        meters["loss_ce"].update(loss_ce.item())

        if ((it % cfg["train"].get("print_freq", 50) == 0) or (it == len(loader)-1)) and rank == 0:
            logger.info(f"Epoch {epoch} Iter {it}/{len(loader)} lr={lr:.5e} "
                        f"loss={meters['loss'].avg:.4f} g={meters['loss_global'].avg:.4f} "
                        f"l={meters['loss_local'].avg:.4f} ce={meters['loss_ce'].avg:.4f}")

    return {k: m.avg for k, m in meters.items()}


# Original seen-only mIoU (kept for reference)
def validate(model, loader, device, num_classes):
    model.eval()
    inter = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            logits = out["logits"]
            # Upsample logits to match label size
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            pred = logits.argmax(1)
            for cls in range(num_classes):
                pred_i = (pred == cls)
                label_i = (labels == cls)
                inter[cls] += (pred_i & label_i).sum()
                union[cls] += (pred_i | label_i).sum()
    # Synchronize metrics across processes for correct distributed validation
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(inter, op=dist.ReduceOp.SUM)
        dist.all_reduce(union, op=dist.ReduceOp.SUM)
    iou = inter / (union + 1e-6)
    miou = iou.mean().item()
    return miou


# Zero-shot validation using CLIP text embeddings over full 20 VOC classes (excluding background)
@torch.no_grad()
def validate_zs(model, teacher: CLIPTeacher, loader, device, seen_ids: List[int], unseen_ids: List[int]):
    model.eval()
    # Prepare CLIP text embeddings for all 20 foreground classes once
    fg_names = [VOC_CLASS_NAMES[i] for i in VOC_FG_IDS]
    E_text = teacher.encode_text_labels(fg_names)  # [20, C], normalized
    E_text = E_text.to(device)

    # Accumulate inter/union for original label ids up to max 21 (0..20)
    max_cls_id = 20
    inter = torch.zeros(max_cls_id + 1, device=device)
    union = torch.zeros(max_cls_id + 1, device=device)

    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        out = model(imgs)
        proj = out["proj"]  # [B, C, L]
        feat_map = out["feat"]  # [B, D, Hf, Wf]
        B, D, Hf, Wf = feat_map.shape
        # Normalize features in CLIP space
        proj = F.normalize(proj, dim=1)  # [B, C, L]
        # Similarity logits over 20 classes at feature resolution (Hf, Wf)
        sim = torch.einsum('kc,bcl->bkl', E_text, proj)  # [B, 20, L]
        sim = sim.view(sim.size(0), sim.size(1), Hf, Wf)   # [B, 20, Hf, Wf]
        # Upsample similarity to label resolution before argmax
        if sim.shape[-2:] != labels.shape[-2:]:
            sim = F.interpolate(sim, size=labels.shape[-2:], mode='bilinear', align_corners=False)
        # Predicted class indices in 0..19 -> map to original ids 1..20
        pred_idx = sim.argmax(1)  # [B, H, W]
        pred_ids = pred_idx + 1   # shift to VOC ids (1..20)
        # Compute IoU per class id, ignoring void (255)
        valid_mask = (labels != 255)
        for cid in VOC_FG_IDS:  # 1..20
            pred_i = (pred_ids == cid) & valid_mask
            label_i = (labels == cid) & valid_mask
            inter[cid] += (pred_i & label_i).sum()
            union[cid] += (pred_i | label_i).sum()

    # Reduce across processes
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(inter, op=dist.ReduceOp.SUM)
        dist.all_reduce(union, op=dist.ReduceOp.SUM)

    eps = 1e-6
    iou = inter / (union + eps)
    # Compute mIoU over seen/unseen groups using original ids
    seen_tensor = torch.tensor(seen_ids, device=device, dtype=torch.long)
    unseen_tensor = torch.tensor(unseen_ids, device=device, dtype=torch.long)
    miou_seen = iou[seen_tensor].mean().item() if len(seen_ids) else 0.0
    miou_unseen = iou[unseen_tensor].mean().item() if len(unseen_ids) else 0.0
    # Harmonic mean
    if (miou_seen + miou_unseen) > 0:
        hiou = 2 * miou_seen * miou_unseen / (miou_seen + miou_unseen)
    else:
        hiou = 0.0
    return miou_seen, miou_unseen, hiou


def init_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/voc.yaml')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    args = parser.parse_args()

    # Initialize distributed training
    rank, world_size, local_rank = init_distributed()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get('seed', 42))

    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(local_rank)

    # Determine seen/unseen class ids (VOC original ids)
    seen_ids = cfg['data'].get('seen_classes', []) or []
    unseen_ids = cfg['data'].get('unseen_classes', []) or []
    if len(seen_ids) == 0 or len(unseen_ids) == 0:
        # Built-in default 15/5 split (VOC ids): seen=15, unseen=5
        # You can override these in configs/voc.yaml
        seen_ids = [1, 2, 4, 5, 6, 7, 9, 11, 13, 15, 16, 17, 19, 20, 3]  # 15 seen
        unseen_ids = [8, 10, 12, 14, 18]  # 5 unseen
        logger.warning(f"Using built-in VOC 15/5 split. Seen: {seen_ids}; Unseen: {unseen_ids}")

    # Dataset (inject splits so training can remap labels for seen classes)
    data_cfg = dict(cfg['data'])
    data_cfg['seen_classes'] = seen_ids
    data_cfg['unseen_classes'] = unseen_ids

    train_set = build_dataset(data_cfg, split='train')
    val_set = build_dataset(data_cfg, split='val')

    # Distributed samplers
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_set,
        batch_size=cfg['train']['batch_size'] // world_size,  # Divide batch size by world size
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg['train'].get('workers', 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=seg_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg['train'].get('val_batch_size', 4) // world_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg['train'].get('workers', 4),
        pin_memory=True,
        collate_fn=seg_collate,
    )

    # Model (align seen count)
    num_seen = len(seen_ids)
    model = build_seg_model(cfg['model'], num_seen_classes=num_seen)
    model.to(device)
    
    # Wrap model with DDP for distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Teacher (align seen count)
    teacher = CLIPTeacher(model_name=cfg['clip']['backbone'], pretrained=cfg['clip'].get('pretrained', 'openai'),
                          context_length=77, device=device, bank_size=cfg['clip'].get('bank_size', 24),
                          temperature=cfg['loss'].get('temperature', 0.07), num_seen=num_seen)

    # Optimizer
    optimizer = build_optimizer(model, cfg['train']['lr'], cfg['train'].get('weight_decay', 1e-4))

    # Training loop
    max_iters = cfg['train']['max_iters']
    epochs = math.ceil(max_iters / len(train_loader))

    best_hiou = 0.0
    for epoch in range(epochs):
        # Set epoch for distributed sampler
        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        stats = train_one_epoch(model, teacher, train_loader, optimizer, device, epoch, max_iters, cfg, rank)
        
        if (epoch + 1) % cfg['train'].get('eval_every', 1) == 0:
            miou_seen, miou_unseen, hiou = validate_zs(model, teacher, val_loader, device, seen_ids, unseen_ids)
            
            # Only log and save on main process (rank 0)
            if rank == 0:
                logger.info(f"Val mIoU(seen)={miou_seen:.4f} | mIoU(unseen)={miou_unseen:.4f} | hIoU={hiou:.4f} @ epoch {epoch}")
                if hiou > best_hiou:
                    best_hiou = hiou
                    os.makedirs('checkpoints', exist_ok=True)
                    # Save the underlying model state (unwrap DDP if needed)
                    model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
                    torch.save({'model': model_state, 'cfg': cfg, 'best_hiou': best_hiou}, f"checkpoints/best.pth")

    if rank == 0:
        logger.info(f"Training done. Best hIoU={best_hiou:.4f}")
    
    # Clean up distributed training
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()