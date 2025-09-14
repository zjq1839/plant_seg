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

from utils import seed_everything, AverageMeter
from datasets import build_dataset
from datasets import seg_collate
from models import build_seg_model, ProjectionHead2D
from loss import info_nce_loss, CombinedSegLoss

from clip_tokens import CLIPTeacher, DinoTeacher, MoCoTeacher


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


# ========== Few-shot plant lesion training/validation ==========

def train_one_epoch_lesion(model, loader, optimizer, device, epoch, max_iters, cfg, rank=0, criterion: CombinedSegLoss = None, teacher: CLIPTeacher = None):
    # Distillation can be disabled via cfg['distill']['enabled']; teacher may be None
    model.train()
    meters = {k: AverageMeter() for k in ["loss", "ce", "dice", "g", "l", "wg", "wl", "posfrac"]}
    if criterion is None:
        criterion = CombinedSegLoss(
            ce_weight=cfg['loss'].get('ce_weight', 1.0),
            dice_weight=cfg['loss'].get('dice_weight', 1.0),
            ignore_index=cfg['loss'].get('ignore_index', -1),
        )
    # Debug flags
    debug = bool(cfg.get('train', {}).get('debug', False))
    debug_freq = int(cfg.get('train', {}).get('debug_freq', 200))

    # Distill switch
    distill_enabled = bool(cfg.get('distill', {}).get('enabled', True))

    iters_done = epoch * len(loader)
    for it, (imgs, labels, _) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # Track foreground (lesion) prevalence per batch
        try:
            pos_frac_val = float((labels == 1).float().mean().item())
        except Exception:
            pos_frac_val = 0.0

        cur_iter = iters_done + it
        lr = adjust_lr(optimizer, cfg['train']['lr'], cur_iter, max_iters)
        # Early stop for dry-run according to max_iters
        if cur_iter >= max_iters:
            break
        optimizer.zero_grad(set_to_none=True)

        out = model(imgs)
        logits = out['logits']
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)

        # supervised seg loss (CE + Dice)
        total, parts = criterion(logits, labels)

        # Defaults when distillation disabled
        g_val = 0.0
        l_val = 0.0
        w_global = 0.0
        w_local = 0.0
        local_debug_info = []

        if distill_enabled and (teacher is not None):
            # Global CLIP distillation (teacher-guided)
            # Student path MUST keep grad; only teacher encoding is no_grad
            feat = out['feat']  # [B, D, Hf, Wf]
            # Project student's dense features to teacher dim if needed
            proj = out.get('proj', None)
            if proj is None:
                # fallback: project via model.clip_proj_2d if exists
                if hasattr(model, 'clip_proj_2d'):
                    proj = model.clip_proj_2d(feat)
                else:
                    proj = feat
            if proj.dim() == 4:
                # [B, C, Hf, Wf] -> [B, C]
                proj_g = proj.mean(dim=(2, 3))
            elif proj.dim() == 3:
                # [B, C, L] -> [B, C]
                proj_g = proj.mean(dim=2)
            else:
                proj_g = proj
            with torch.no_grad():
                Cg_pos, T_dense = teacher.encode_image_dense(imgs)  # [B, C], [B, C, Ht, Wt] or None
            # detach + clone to avoid any potential in-place modification on saved tensors for backward
            Cg_pos = Cg_pos.detach().clone()
            T_dense = T_dense.detach().clone() if (T_dense is not None and isinstance(T_dense, torch.Tensor)) else None
            loss_global = teacher.info_nce_global(proj_g, Cg_pos)

            # Local dense distillation (pixel-wise cosine on lesion mask)
            loss_local = 0.0
            try:
                # Build student dense projection [B, C, Hs, Ws]
                if hasattr(model, 'module') and hasattr(model.module, 'clip_proj_2d'):
                    S_dense = model.module.clip_proj_2d(out['feat'])
                elif hasattr(model, 'clip_proj_2d'):
                    S_dense = model.clip_proj_2d(out['feat'])
                else:
                    S_dense = out['proj'] if ('proj' in out) else out['feat']
                # Ensure 4D spatial map
                if S_dense.dim() == 3 and out['feat'].dim() == 4:
                    Bc, Cc, Hf, Wf = out['feat'].shape
                    if S_dense.size(-1) == Hf * Wf:
                        S_dense = S_dense.view(Bc, S_dense.size(1), Hf, Wf)
                if S_dense.dim() != 4:
                    raise RuntimeError(f"Student dense proj must be 4D, got shape={tuple(S_dense.shape)}")

                # Teacher dense grid [B, C, Ht, Wt]
                if T_dense is None or (not isinstance(T_dense, torch.Tensor)):
                    raise RuntimeError("Teacher did not provide dense tokens; cannot run dense distillation.")
                T_dense_ = T_dense
                # Align spatial size to student
                if T_dense_.shape[-2:] != S_dense.shape[-2:]:
                    T_dense_ = F.interpolate(T_dense_, size=S_dense.shape[-2:], mode='bilinear', align_corners=False)

                # Channel check
                if T_dense_.size(1) != S_dense.size(1):
                    raise RuntimeError(f"Channel mismatch: teacher C={T_dense_.size(1)} vs student C={S_dense.size(1)}. Ensure ProjectionHead2D maps to teacher dim.")

                # Normalize per-location across channels
                Sn = F.normalize(S_dense, dim=1)
                Tn = F.normalize(T_dense_, dim=1)
                # Cosine similarity per pixel -> loss map
                cos_map = (Sn * Tn).sum(dim=1)  # [B, H, W]
                dense_loss_map = 1.0 - cos_map

                # Lesion mask at student resolution
                lesion = (labels == 1).float().unsqueeze(1)  # [B,1,H,W]
                if lesion.shape[-2:] != S_dense.shape[-2:]:
                    lesion = F.interpolate(lesion, size=S_dense.shape[-2:], mode='nearest')
                lesion = lesion.squeeze(1)

                # Compute masked average; fallback to all-pixel avg if no lesion present
                eps = 1e-6
                mask_sum = lesion.sum()
                if mask_sum > 0:
                    loss_local = (dense_loss_map * lesion).sum() / (mask_sum + eps)
                    local_debug_info.append(
                        f"dense_local: T{tuple(T_dense.shape if isinstance(T_dense, torch.Tensor) else ('NA',))} -> {tuple(T_dense_.shape)} S{tuple(S_dense.shape)} mask_sum={float(mask_sum.item()):.1f} loss={float(loss_local):.4f}"
                    )
                else:
                    loss_local = dense_loss_map.mean()
                    local_debug_info.append(
                        f"dense_local: no_lesion_in_batch; fallback_mean loss={float(loss_local):.4f}"
                    )
            except Exception as e:
                loss_local = 0.0
                local_debug_info.append(f"dense_local_exception: {e}")

            # Adaptive weighting mechanism based on loss magnitudes
            base_w_global = cfg['loss'].get('w_global', 0.5)
            base_w_local = cfg['loss'].get('w_local', 0.5)
            
            # Dynamic weight adjustment based on relative loss magnitudes
            if isinstance(loss_local, torch.Tensor) and loss_local.item() > 0:
                loss_ratio = loss_global.item() / (loss_local.item() + 1e-8)
                # If global loss is much larger, reduce its weight; if local loss is larger, reduce local weight
                adaptive_factor = torch.sigmoid(torch.tensor(1.0 - loss_ratio)).item()
                w_global = base_w_global * (0.5 + 0.5 * adaptive_factor)
                w_local = base_w_local * (0.5 + 0.5 * (1.0 - adaptive_factor))
            else:
                w_global = base_w_global
                w_local = base_w_local
            total = total + w_global * loss_global + w_local * loss_local
            g_val = float(loss_global.item() if torch.is_tensor(loss_global) else loss_global)
            l_val = float(loss_local)

        # Track actual weights used
        meters['wg'].update(float(w_global))
        meters['wl'].update(float(w_local))

        if debug and rank == 0 and (debug_freq > 0 and (it % debug_freq == 0)):
            logger.info(f"[DEBUG][LESION] iter={it} lr={lr:.3e} loss_global={g_val:.4f} loss_local={l_val:.4f} wg={meters['wg'].avg:.3f} wl={meters['wl'].avg:.3f} ce={float(parts['ce']):.4f} dice={float(parts['dice']):.4f}")
            logger.info(f"[DEBUG][LESION] local_debug: {' | '.join(local_debug_info)}")
            # Additional debug: check feature map resolution vs original
            proj_shape = tuple(out['proj'].shape) if isinstance(out, dict) and ('proj' in out) and hasattr(out['proj'], 'shape') else ('NA',)
            logger.info(f"[DEBUG][LESION] resolution: orig={tuple(labels.shape[-2:])} feat={tuple(out['feat'].shape[-2:])} proj={proj_shape} downscale={labels.shape[-1]/out['feat'].shape[-1]:.1f}x")

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train'].get('grad_clip', 1.0))
        optimizer.step()
        # Update teacher bank after optimizer step to avoid in-place version bump during backward
        if distill_enabled and (teacher is not None):
            try:
                teacher.enqueue_cls(Cg_pos.detach())
            except Exception:
                pass

        meters['loss'].update(total.item())
        meters['ce'].update(float(parts['ce']))
        meters['dice'].update(float(parts['dice']))
        meters['g'].update(g_val)
        meters['l'].update(l_val)
        meters['posfrac'].update(pos_frac_val)

        if ((it % cfg['train'].get('print_freq', 50) == 0) or (it == len(loader)-1)) and rank == 0:
            logger.info(f"[LESION] Epoch {epoch} Iter {it}/{len(loader)} lr={lr:.5e} "
                        f"loss={meters['loss'].avg:.4f} ce={meters['ce'].avg:.4f} "
                        f"dice={meters['dice'].avg:.4f} g={meters['g'].avg:.4f} l={meters['l'].avg:.4f} "
                        f"wg={meters['wg'].avg:.3f} wl={meters['wl'].avg:.3f} posfrac={meters['posfrac'].avg:.4f}")

    return {k: m.avg for k, m in meters.items()}


@torch.no_grad()
def validate_lesion(model, loader, device, num_classes: int = 2):
    model.eval()
    inter = torch.zeros(num_classes, device=device, dtype=torch.float64)
    union = torch.zeros(num_classes, device=device, dtype=torch.float64)
    dice_num = torch.tensor(0.0, device=device)
    dice_den = torch.tensor(0.0, device=device)
    # Extra diagnostics for class-1 (lesion)
    tp1 = torch.tensor(0.0, device=device)
    fp1 = torch.tensor(0.0, device=device)
    fn1 = torch.tensor(0.0, device=device)
    inter_pos1 = torch.tensor(0.0, device=device)
    union_pos1 = torch.tensor(0.0, device=device)
    images_with_lesion = torch.tensor(0.0, device=device)
    images_total = torch.tensor(0.0, device=device)
    pos_prob_sum = torch.tensor(0.0, device=device)
    neg_prob_sum = torch.tensor(0.0, device=device)
    pos_count = torch.tensor(0.0, device=device)
    neg_count = torch.tensor(0.0, device=device)

    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        out = model(imgs)
        logits = out['logits']
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
        pred = logits.argmax(1)
        # Softmax probabilities for lesion class if available
        if logits.size(1) > 1:
            probs = F.softmax(logits, dim=1)
            prob1 = probs[:, 1, ...]
        else:
            # Fallback: treat single-logit as probability via sigmoid
            prob1 = torch.sigmoid(logits[:, 0, ...])
        images_total += float(imgs.size(0))
        # IoU per class
        for cls in range(num_classes):
            pred_i = (pred == cls)
            label_i = (labels == cls)
            inter[cls] += (pred_i & label_i).sum()
            union[cls] += (pred_i | label_i).sum()
        # Dice for lesion if class 1 exists
        if num_classes > 1:
            p1 = (pred == 1)
            g1 = (labels == 1)
            dice_num += 2.0 * (p1 & g1).sum()
            dice_den += p1.sum() + g1.sum() + 1e-6
            # Diagnostics for class-1
            tp1 += (p1 & g1).sum()
            fp1 += (p1 & (~g1)).sum()
            fn1 += ((~p1) & g1).sum()
            # IoU on images that actually contain lesion
            if (g1.any()).item():
                inter_pos1 += (p1 & g1).sum()
                union_pos1 += (p1 | g1).sum()
                images_with_lesion += 1.0
            # Probabilities on GT-pos and GT-neg pixels
            pos_prob_sum += prob1[g1].sum()
            pos_count += g1.sum()
            neg_prob_sum += prob1[~g1].sum()
            neg_count += (~g1).sum()
        else:
            # Fallback: use foreground as class 0 in binary-like setup
            p0 = (pred == 0)
            g0 = (labels == 0)
            dice_num += 2.0 * (p0 & g0).sum()
            dice_den += p0.sum() + g0.sum() + 1e-6

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(inter, op=dist.ReduceOp.SUM)
        dist.all_reduce(union, op=dist.ReduceOp.SUM)
        dist.all_reduce(dice_num, op=dist.ReduceOp.SUM)
        dist.all_reduce(dice_den, op=dist.ReduceOp.SUM)
        dist.all_reduce(tp1, op=dist.ReduceOp.SUM)
        dist.all_reduce(fp1, op=dist.ReduceOp.SUM)
        dist.all_reduce(fn1, op=dist.ReduceOp.SUM)
        dist.all_reduce(inter_pos1, op=dist.ReduceOp.SUM)
        dist.all_reduce(union_pos1, op=dist.ReduceOp.SUM)
        dist.all_reduce(images_with_lesion, op=dist.ReduceOp.SUM)
        dist.all_reduce(images_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(pos_prob_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(neg_prob_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(pos_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(neg_count, op=dist.ReduceOp.SUM)

    iou = inter / (union + 1e-6)
    miou = iou.mean().item()
    dice_pos = (dice_num / dice_den).item()
    # Extra summaries
    eps = 1e-6
    prec1 = (tp1 / (tp1 + fp1 + eps)).item() if (tp1 + fp1) > 0 else 0.0
    rec1 = (tp1 / (tp1 + fn1 + eps)).item() if (tp1 + fn1) > 0 else 0.0
    iou1_pos = (inter_pos1 / (union_pos1 + eps)).item() if images_with_lesion > 0 else float('nan')
    pos_prob = (pos_prob_sum / (pos_count + eps)).item() if pos_count > 0 else float('nan')
    bg_prob = (neg_prob_sum / (neg_count + eps)).item() if neg_count > 0 else float('nan')
    extra = {
        'iou1': float(iou[1].item()) if iou.numel() > 1 else float(iou[0].item()),
        'precision1': float(prec1),
        'recall1': float(rec1),
        'iou1_pos': float(iou1_pos),
        'pos_prob': float(pos_prob),
        'bg_prob': float(bg_prob),
        'images_with_lesion': int(images_with_lesion.item()),
        'images_total': int(images_total.item()),
    }
    return dice_pos, miou, iou.detach().float().tolist(), extra


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
    parser.add_argument('--config', type=str, default='configs/plant_fewshot.yaml')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--val_split', type=str, default='val', help='Validation split name (e.g., val or test)')
    args = parser.parse_args()

    # Initialize distributed training
    rank, world_size, local_rank = init_distributed()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get('seed', 42))

    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Log config and distillation flag at startup
    distill_enabled_start = bool(cfg.get('distill', {}).get('enabled', True))
    if rank == 0:
        logger.info(f"[LESION] Config loaded: path={args.config} distill.enabled={distill_enabled_start}")

    # ========== Lesion few-shot branch ==========
    if cfg.get('data', {}).get('name') == 'plant_lesion':
        data_cfg = dict(cfg['data'])
        train_set = build_dataset(data_cfg, split='train')
        val_set = build_dataset(data_cfg, split=args.val_split)

        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
        val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

        train_loader = DataLoader(
            train_set,
            batch_size=cfg['train']['batch_size'] // world_size,
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

        num_classes = cfg['data'].get('num_classes', 2)
        model = build_seg_model(cfg['model'], num_seen_classes=num_classes)
        model.to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        # ===== Verify projection head configuration & architecture =====
        try:
            mh = cfg.get('model', {})
            logger.info(
                f"[LESION] model.proj_head={mh.get('proj_head','conv')} "
                f"proj_mid_dim={mh.get('proj_mid_dim', mh.get('clip_dim', 512))} "
                f"proj_norm={mh.get('proj_norm','none')} proj_dropout={mh.get('proj_dropout', 0.0)}"
            )
            head = model.module.clip_proj_2d if isinstance(model, DDP) else model.clip_proj_2d
            def _count_params(m):
                return sum(p.numel() for p in m.parameters() if p.requires_grad)
            logger.info(f"[LESION] ProjectionHead2D arch: {head} | params={_count_params(head)}")
            # BN stability warning for small per-GPU batch sizes
            try:
                per_rank_bs = max(1, int(cfg['train']['batch_size']) // max(1, world_size))
                if mh.get('proj_norm', 'none') == 'bn' and per_rank_bs < 4:
                    logger.warning("[LESION] Warning: proj_norm=bn with per-GPU batch <4 may be unstable. Consider proj_norm='none' or using SyncBN/GroupNorm.")
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"[LESION] Could not log projection head details: {e}")
        # ===== end verification =====

        # Teacher (CLIP/DINO/MoCo) for lesion guidance (honors distill.enabled)
        distill_enabled = bool(cfg.get('distill', {}).get('enabled', True))
        teacher = None
        if distill_enabled:
            teacher_cfg = cfg.get('teacher', {}) or {}
            teacher_type = str(teacher_cfg.get('type', 'clip')).lower()
            if rank == 0:
                logger.info(f"[LESION] Distillation enabled: teacher_type={teacher_type}")
            if teacher_type == 'dino':
                dino_cfg = cfg.get('dino', {})
                backbone = teacher_cfg.get('backbone', dino_cfg.get('backbone', 'vit_base_patch14_dinov2.lvd142m'))
                bank_size = int(teacher_cfg.get('bank_size', dino_cfg.get('bank_size', cfg['clip'].get('bank_size', 24))))
                temperature = float(teacher_cfg.get('temperature', cfg['loss'].get('temperature', 0.07)))
                teacher = DinoTeacher(
                    model_name=backbone,
                    device=device,
                    bank_size=bank_size,
                    temperature=temperature,
                    num_seen=num_classes,
                )
                try:
                    per_rank_bs = cfg['train']['batch_size'] // world_size
                    logger.info(
                        f"[LESION] Debug summary -> device={device} world_size={world_size} "
                        f"train_size={len(train_set)} val_size={len(val_set)} "
                        f"batch={per_rank_bs}x{world_size} num_classes={num_classes}"
                    )
                    logger.info(f"[LESION] Using DinoTeacher: backbone={backbone} bank_size={bank_size} tau={temperature}")
                except Exception as e:
                    logger.warning(f"[LESION] debug summary failed: {e}")
            elif teacher_type == 'moco':
                moco_cfg = cfg.get('moco', {})
                backbone = teacher_cfg.get('backbone', moco_cfg.get('backbone', 'vit_base_patch16_224'))
                bank_size = int(teacher_cfg.get('bank_size', moco_cfg.get('bank_size', cfg['clip'].get('bank_size', 24))))
                temperature = float(teacher_cfg.get('temperature', cfg['loss'].get('temperature', 0.07)))
                # Optional flags for MoCoTeacher
                pretrained = bool(teacher_cfg.get('pretrained', moco_cfg.get('pretrained', False)))
                checkpoint = teacher_cfg.get('checkpoint', moco_cfg.get('checkpoint', None))
                teacher = MoCoTeacher(
                    model_name=backbone,
                    device=device,
                    bank_size=bank_size,
                    temperature=temperature,
                    num_seen=num_classes,
                    pretrained=pretrained,
                    checkpoint=checkpoint,
                )
                try:
                    per_rank_bs = cfg['train']['batch_size'] // world_size
                    logger.info(
                        f"[LESION] Debug summary -> device={device} world_size={world_size} "
                        f"train_size={len(train_set)} val_size={len(val_set)} "
                        f"batch={per_rank_bs}x{world_size} num_classes={num_classes}"
                    )
                    logger.info(f"[LESION] Using MoCoTeacher: backbone={backbone} bank_size={bank_size} tau={temperature} pretrained={pretrained} checkpoint={checkpoint}")
                except Exception as e:
                    logger.warning(f"[LESION] debug summary failed: {e}")
            else:
                teacher = CLIPTeacher(
                    model_name=cfg['clip']['backbone'],
                    pretrained=cfg['clip'].get('pretrained', 'openai'),
                    context_length=77,
                    device=device,
                    bank_size=cfg['clip'].get('bank_size', 24),
                    temperature=cfg['loss'].get('temperature', 0.07),
                    num_seen=num_classes,
                )
                try:
                    per_rank_bs = cfg['train']['batch_size'] // world_size
                    logger.info(
                        f"[LESION] Debug summary -> device={device} world_size={world_size} "
                        f"train_size={len(train_set)} val_size={len(val_set)} "
                        f"batch={per_rank_bs}x{world_size} num_classes={num_classes}"
                    )
                    logger.info(
                        f"[LESION] Using CLIPTeacher: backbone={cfg['clip']['backbone']} pretrained={cfg['clip'].get('pretrained','openai')} "
                        f"bank_size={cfg['clip'].get('bank_size',24)} tau={cfg['loss'].get('temperature',0.07)}"
                    )
                except Exception as e:
                    logger.warning(f"[LESION] debug summary failed: {e}")
        else:
            if rank == 0:
                logger.info("[LESION] Distillation disabled via cfg['distill']['enabled']=False. Skipping teacher construction and KD losses.")

        # === Align student projection dim to teacher embedding dim if needed (only when distillation enabled) ===
        if distill_enabled and (teacher is not None):
            try:
                head = model.module.clip_proj_2d if isinstance(model, DDP) else model.clip_proj_2d
                # infer current head out dim
                head_out_dim = None
                if hasattr(head, 'net'):
                    if isinstance(head.net, nn.Conv2d):
                        head_out_dim = head.net.out_channels
                    elif isinstance(head.net, nn.Sequential):
                        for m in reversed(list(head.net.children())):
                            if isinstance(m, nn.Conv2d):
                                head_out_dim = m.out_channels
                                break
                # fallback if above failed
                if head_out_dim is None and hasattr(head, 'out_channels'):
                    head_out_dim = head.out_channels
                teacher_dim = int(getattr(teacher, 'C', 0))
                if teacher_dim and head_out_dim and head_out_dim != teacher_dim:
                    mh = cfg.get('model', {})
                    proj_conv = model.module.proj if isinstance(model, DDP) else model.proj
                    in_dim = proj_conv.out_channels if hasattr(proj_conv, 'out_channels') else mh.get('feat_dim', 256)
                    head_type = str(mh.get('proj_head', 'conv'))
                    # If proj_mid_dim is tied to clip_dim in cfg, retie it to new teacher_dim
                    mid_cfg = mh.get('proj_mid_dim', None)
                    default_mid = mh.get('clip_dim', 512)
                    proj_mid_dim = teacher_dim if (mid_cfg is None or mid_cfg == default_mid) else int(mid_cfg)
                    norm = str(mh.get('proj_norm', 'none'))
                    dropout = float(mh.get('proj_dropout', 0.0))
                    new_head = ProjectionHead2D(in_dim, teacher_dim, head_type=head_type, mid_dim=proj_mid_dim, norm=norm, dropout=dropout).to(device)
                    if isinstance(model, DDP):
                        model.module.clip_proj_2d = new_head
                    else:
                        model.clip_proj_2d = new_head
                    # best-effort: reflect in cfg for logs/checkpoints
                    try:
                        cfg['model']['clip_dim'] = teacher_dim
                    except Exception:
                        pass
                    logger.info(f"[LESION] Adjusted projection head out_dim from {head_out_dim} -> teacher_dim {teacher_dim} to avoid dim mismatch")
            except Exception as e:
                logger.warning(f"[LESION] projection/teacher dim alignment skipped due to: {e}")
        else:
            if rank == 0:
                logger.info("[LESION] Distillation disabled or no teacher; skip projection head alignment to teacher dim.")

        optimizer = build_optimizer(model, cfg['train']['lr'], cfg['train'].get('weight_decay', 1e-4))
        criterion = CombinedSegLoss(
            ce_weight=cfg['loss'].get('ce_weight', 1.0),
            dice_weight=cfg['loss'].get('dice_weight', 1.0),
            ignore_index=cfg['loss'].get('ignore_index', -1),
        )

        max_iters = cfg['train']['max_iters']
        epochs = math.ceil(max_iters / len(train_loader))
        best_dice = 0.0

        # Optional: print first epoch plan for debugging
        if cfg.get('train', {}).get('debug', False) and rank == 0:
            logger.info(f"[LESION] Plan -> epochs={epochs} max_iters={max_iters} print_freq={cfg['train'].get('print_freq',50)} debug_freq={cfg['train'].get('debug_freq',200)}")

        for epoch in range(epochs):
            if world_size > 1 and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            _ = train_one_epoch_lesion(model, train_loader, optimizer, device, epoch, max_iters, cfg, rank, criterion, teacher)

            if (epoch + 1) % cfg['train'].get('eval_every', 1) == 0:
                num_classes = cfg['data'].get('num_classes', 2)
                dice_pos, miou, iou_per_class, extra = validate_lesion(model, val_loader, device, num_classes)
                if rank == 0:
                    try:
                        iou_str = ", ".join([f"{v:.4f}" for v in iou_per_class])
                    except Exception:
                        iou_str = str(iou_per_class)
                    logger.info(f"[LESION] Val Dice(lesion)={dice_pos:.4f} | mIoU({num_classes}cls)={miou:.4f} | IoU_per_class=[{iou_str}] @ epoch {epoch}")
                    # Additional diagnostics for lesion class
                    try:
                        logger.info(
                            f"[LESION][Diag] IoU1={extra.get('iou1', float('nan')):.4f} | Prec1={extra.get('precision1', float('nan')):.4f} | "
                            f"Rec1={extra.get('recall1', float('nan')):.4f} | IoU1|pos={extra.get('iou1_pos', float('nan')):.4f} | "
                            f"p(pos|lesion)={extra.get('pos_prob', float('nan')):.3f} | p(pos|bg)={extra.get('bg_prob', float('nan')):.3f} | "
                            f"lesion_imgs={extra.get('images_with_lesion', 0)}/{extra.get('images_total', 0)}"
                        )
                    except Exception:
                        pass
                    if dice_pos > best_dice:
                        best_dice = dice_pos
                        os.makedirs('checkpoints', exist_ok=True)
                        model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
                        torch.save({'model': model_state, 'cfg': cfg, 'best_dice': best_dice}, "checkpoints/best_lesion.pth")

        if rank == 0:
            logger.info(f"[LESION] Training done. Best Dice={best_dice:.4f}")
        if world_size > 1:
            dist.destroy_process_group()
        return
    else:
        raise ValueError("Misuse: this training script only supports data.name='plant_lesion'. Please use configs/plant_fewshot.yaml and set data.name='plant_lesion' in your config.")


if __name__ == '__main__':
    main()