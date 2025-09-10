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
from loss import info_nce_loss, CombinedSegLoss
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


# ========== Few-shot plant lesion training/validation ==========

def train_one_epoch_lesion(model, loader, optimizer, device, epoch, max_iters, cfg, rank=0, criterion: CombinedSegLoss = None, teacher: CLIPTeacher = None):
    assert teacher is not None, "CLIP teacher must be provided for lesion few-shot training"
    model.train()
    meters = {k: AverageMeter() for k in ["loss", "ce", "dice", "g", "l"]}
    if criterion is None:
        criterion = CombinedSegLoss(
            ce_weight=cfg['loss'].get('ce_weight', 1.0),
            dice_weight=cfg['loss'].get('dice_weight', 1.0),
            ignore_index=cfg['loss'].get('ignore_index', -1),
        )
    # Debug flags
    debug = bool(cfg.get('train', {}).get('debug', False))
    debug_freq = int(cfg.get('train', {}).get('debug_freq', 200))

    iters_done = epoch * len(loader)
    for it, (imgs, labels, _) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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

        # CLIP-guided auxiliary losses (always enabled)
        g_val = 0.0
        l_val = 0.0
        # teacher is always provided; enforce guidance
        if True:
            # prepare present ids per sample (only lesion class=1 to avoid huge background tokens)
            present_batch = []
            for b in range(labels.shape[0]):
                ids = torch.unique(labels[b])
                keep = ids[(ids == 1)]  # only lesion class
                if keep.numel() > 0:
                    present_batch.append(keep.to(labels.device))
                else:
                    present_batch.append(torch.empty(0, dtype=torch.long, device=labels.device))
            with torch.no_grad():
                teacher_out = teacher.forward_tokens_and_pseudo(imgs, labels, present_batch)
                Cg = teacher_out["Cg"]           # [B, C]
                Cl = teacher_out["Cl"]           # list of [n_present, C]
                Yp = teacher_out["Yp"]           # [B, H, W]
                present_ids = teacher_out["present_ids"]

            # Debug: print shapes and teacher meta on first/bounded iterations
            if debug and ((epoch == 0 and it == 0) or (debug_freq > 0 and (it % debug_freq == 0))):
                try:
                    logger.info(f"[DEBUG][LESION] epoch={epoch} iter={it} imgs={tuple(imgs.shape)} labels={tuple(labels.shape)} logits={tuple(logits.shape)} proj={tuple(out['proj'].shape)} feat={tuple(out['feat'].shape)}")
                    B_dbg = labels.shape[0]
                    cl_lens = []
                    for b in range(B_dbg):
                        cb = Cl[b]
                        if cb is None:
                            cl_lens.append(0)
                        elif torch.is_tensor(cb):
                            cl_lens.append(int(cb.shape[0]))
                        else:
                            try:
                                cl_lens.append(len(cb))
                            except Exception:
                                cl_lens.append(0)
                    present_ids_list = [p.tolist() if torch.is_tensor(p) else [] for p in present_ids]
                    logger.info(f"[DEBUG][LESION] teacher: Cg={tuple(Cg.shape)} Cl_lens={cl_lens} present_ids={present_ids_list}")
                    for b in range(min(B_dbg, 2)):
                        uniq, counts = torch.unique(Yp[b], return_counts=True)
                        uniq = uniq.tolist()
                        counts = counts.tolist()
                        logger.info(f"[DEBUG][LESION] pseudo labels sample{b}: uniq={uniq} counts={counts[:min(len(counts), 5)]} total={int(Yp[b].numel())}")
                except Exception as e:
                    logger.warning(f"[DEBUG][LESION] debug print failed: {e}")

            # student projection in CLIP space
            F_proj = out['proj']                 # [B, C, L]
            F_proj = F.normalize(F_proj, dim=1)

            # global guidance
            Cg_norm = F.normalize(Cg, dim=-1)
            W_attn = torch.softmax(torch.einsum("bc,bcl->bl", Cg_norm, F_proj) / math.sqrt(F_proj.shape[1]), dim=-1)
            Fg = torch.einsum("bcl,bl->bc", F_proj, W_attn)
            loss_global = teacher.info_nce_global(Fg, Cg)

            # local guidance (class-wise; resize pseudo to feature size)
            B = labels.shape[0]
            Hf, Wf = out['feat'].shape[-2:]
            Yp_resized = F.interpolate(Yp.float().unsqueeze(1), size=(Hf, Wf), mode='nearest').squeeze(1).long()
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
                    if cnt < 1:
                        continue
                    fl = fb[:, mask].mean(dim=1)  # [C]
                    cls_feats.append(fl)
                    cls_tokens.append(cl_tokens[i])
                if len(cls_feats) and len(cls_tokens):
                    f_stack = torch.stack(cls_feats, dim=0)
                    c_stack = torch.stack(cls_tokens, dim=0)
                    loss_local += info_nce_loss(f_stack, c_stack, temperature=teacher.tau)
            loss_local = loss_local / max(B, 1)

            w_global = cfg['loss'].get('w_global', 0.5)
            w_local = cfg['loss'].get('w_local', 0.5)
            total = total + w_global * loss_global + w_local * loss_local
            g_val = float(loss_global.item() if torch.is_tensor(loss_global) else loss_global)
            l_val = float(loss_local)

            if debug and (debug_freq > 0 and (it % debug_freq == 0)):
                logger.info(f"[DEBUG][LESION] iter={it} lr={lr:.3e} loss_global={g_val:.4f} loss_local={l_val:.4f} ce={float(parts['ce']):.4f} dice={float(parts['dice']):.4f}")

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train'].get('grad_clip', 1.0))
        optimizer.step()

        meters['loss'].update(total.item())
        meters['ce'].update(float(parts['ce']))
        meters['dice'].update(float(parts['dice']))
        meters['g'].update(g_val)
        meters['l'].update(l_val)

        if ((it % cfg['train'].get('print_freq', 50) == 0) or (it == len(loader)-1)) and rank == 0:
            logger.info(f"[LESION] Epoch {epoch} Iter {it}/{len(loader)} lr={lr:.5e} "
                        f"loss={meters['loss'].avg:.4f} ce={meters['ce'].avg:.4f} "
                        f"dice={meters['dice'].avg:.4f} g={meters['g'].avg:.4f} l={meters['l'].avg:.4f}")

    return {k: m.avg for k, m in meters.items()}


@torch.no_grad()
def validate_lesion(model, loader, device):
    model.eval()
    inter = torch.zeros(2, device=device)
    union = torch.zeros(2, device=device)
    dice_num = torch.tensor(0.0, device=device)
    dice_den = torch.tensor(0.0, device=device)

    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        out = model(imgs)
        logits = out['logits']
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
        pred = logits.argmax(1)
        # IoU per class
        for cls in [0, 1]:
            pred_i = (pred == cls)
            label_i = (labels == cls)
            inter[cls] += (pred_i & label_i).sum()
            union[cls] += (pred_i | label_i).sum()
        # Dice for lesion (class 1)
        p1 = (pred == 1)
        g1 = (labels == 1)
        dice_num += 2.0 * (p1 & g1).sum()
        dice_den += p1.sum() + g1.sum() + 1e-6

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(inter, op=dist.ReduceOp.SUM)
        dist.all_reduce(union, op=dist.ReduceOp.SUM)
        dist.all_reduce(dice_num, op=dist.ReduceOp.SUM)
        dist.all_reduce(dice_den, op=dist.ReduceOp.SUM)

    iou = inter / (union + 1e-6)
    miou = iou.mean().item()
    dice_pos = (dice_num / dice_den).item()
    return dice_pos, miou


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

        # CLIP teacher for lesion guidance (always enabled)
        teacher = CLIPTeacher(
            model_name=cfg['clip']['backbone'],
            pretrained=cfg['clip'].get('pretrained', 'openai'),
            context_length=77,
            device=device,
            bank_size=cfg['clip'].get('bank_size', 24),
            temperature=cfg['loss'].get('temperature', 0.07),
            num_seen=num_classes,
        )
        # Debug: teacher/config summary
        try:
            per_rank_bs = cfg['train']['batch_size'] // world_size
            logger.info(
                f"[LESION] Debug summary -> device={device} world_size={world_size} "
                f"train_size={len(train_set)} val_size={len(val_set)} "
                f"batch={per_rank_bs}x{world_size} num_classes={num_classes}"
            )
            logger.info(
                f"[LESION] CLIPTeacher: backbone={cfg['clip']['backbone']} pretrained={cfg['clip'].get('pretrained','openai')} "
                f"bank_size={cfg['clip'].get('bank_size',24)} tau={cfg['loss'].get('temperature',0.07)}"
            )
        except Exception as e:
            logger.warning(f"[LESION] debug summary failed: {e}")

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
                dice_pos, miou = validate_lesion(model, val_loader, device)
                if rank == 0:
                    logger.info(f"[LESION] Val Dice(lesion)={dice_pos:.4f} | mIoU(2cls)={miou:.4f} @ epoch {epoch}")
                    if dice_pos > best_dice:
                        best_dice = dice_pos
                        os.makedirs('checkpoints', exist_ok=True)
                        model_state = model.module.state_dict() if world_size > 1 else model.state_dict()
                        torch.save({'model': model_state, 'cfg': cfg, 'best_dice': best_dice}, f"checkpoints/best_lesion.pth")

        if rank == 0:
            logger.info(f"[LESION] Training done. Best Dice={best_dice:.4f}")
        if world_size > 1:
            dist.destroy_process_group()
        return
    else:
        raise ValueError("Misuse: this training script only supports data.name='plant_lesion'. Please use configs/plant_fewshot.yaml and set data.name='plant_lesion' in your config.")


if __name__ == '__main__':
    main()