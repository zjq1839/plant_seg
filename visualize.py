import os
import math
import random
import argparse
from typing import List

import yaml
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from datasets import build_dataset


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def to_pil_uint8(t: torch.Tensor) -> Image.Image:
    # t: [3,H,W] in [0,1]
    t = t.clamp(0, 1).cpu().numpy()
    t = (t * 255.0 + 0.5).astype(np.uint8)
    t = np.transpose(t, (1, 2, 0))  # HWC
    return Image.fromarray(t)


def colorize_mask(mask: np.ndarray, num_classes: int, ignore_index: int = -1, seed: int = 42) -> Image.Image:
    rng = np.random.RandomState(seed)
    palette = rng.randint(0, 255, size=(num_classes + 1, 3), dtype=np.uint8)
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    # ignore stays black/transparent in overlay step
    m = mask >= 0
    out[m] = palette[np.clip(mask[m], 0, num_classes)]
    return Image.fromarray(out)


def overlay(img: Image.Image, color_mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    cm = color_mask.convert('RGBA')
    # build alpha from non-black pixels
    arr = np.array(cm)
    alpha_map = (arr[..., :3].sum(axis=-1) > 0).astype(np.float32) * (alpha * 255.0)
    arr[..., 3] = alpha_map.astype(np.uint8)
    cm = Image.fromarray(arr)
    base = img.convert('RGBA')
    out = Image.alpha_composite(base, cm).convert('RGB')
    return out


def make_grid(images: List[Image.Image], cols: int = 4, pad: int = 4, bg=(30, 30, 30)) -> Image.Image:
    if len(images) == 0:
        return None
    w, h = images[0].size
    rows = math.ceil(len(images) / cols)
    grid = Image.new('RGB', (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), bg)
    for i, im in enumerate(images):
        r = i // cols
        c = i % cols
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        grid.paste(im, (x, y))
    return grid


@torch.no_grad()
def vis_dataset(cfg_path: str, split: str = 'train', num: int = 8, outdir: str = 'out_vis/dataset', seed: int = 0):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ds = build_dataset(cfg['data'], split=split)
    ensure_dir(outdir)
    random.seed(seed)
    idxs = random.sample(range(len(ds)), k=min(num, len(ds)))
    tiles = []
    for i, idx in enumerate(idxs):
        x, y, _ = ds[idx]
        img = to_pil_uint8(x)
        cm = colorize_mask(y.numpy(), num_classes=cfg['data']['num_seen'])
        over = overlay(img, cm, alpha=0.5)
        # stack [img | mask | overlay]
        row = Image.new('RGB', (img.width * 3, img.height), (0, 0, 0))
        row.paste(img, (0, 0))
        row.paste(cm, (img.width, 0))
        row.paste(over, (img.width * 2, 0))
        row.save(os.path.join(outdir, f'sample_{i:02d}.jpg'))
        tiles.append(row)
    grid = make_grid(tiles, cols=2)
    if grid is not None:
        grid.save(os.path.join(outdir, 'dataset_grid.jpg'))


@torch.no_grad()
def vis_pseudo(cfg_path: str, split: str = 'train', num: int = 8, outdir: str = 'out_vis/pseudo', seed: int = 0):
    # lazy import to avoid heavy deps when not needed
    from clip_tokens import CLIPTeacher

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = build_dataset(cfg['data'], split=split)
    teacher = CLIPTeacher(model_name=cfg['clip']['backbone'], pretrained=cfg['clip'].get('pretrained', 'openai'),
                          device=device, bank_size=cfg['clip'].get('bank_size', 24),
                          temperature=cfg['loss'].get('temperature', 0.07), num_seen=cfg['data']['num_seen'])
    ensure_dir(outdir)
    random.seed(seed)
    idxs = random.sample(range(len(ds)), k=min(num, len(ds)))
    tiles = []
    for i, idx in enumerate(idxs):
        x, y, present = ds[idx]
        img = to_pil_uint8(x)
        xb = x.unsqueeze(0).to(device)
        yb = y.unsqueeze(0).to(device)
        present_ids = [present.to(device)]
        out = teacher.forward_tokens_and_pseudo(xb, yb, present_ids)
        Yp = out['Yp'][0].cpu().numpy()
        cm_gt = colorize_mask(y.numpy(), num_classes=cfg['data']['num_seen'])
        cm_pseudo = colorize_mask(Yp, num_classes=cfg['data']['num_seen'])
        over_gt = overlay(img, cm_gt, alpha=0.45)
        over_ps = overlay(img, cm_pseudo, alpha=0.45)
        row = Image.new('RGB', (img.width * 4, img.height), (0, 0, 0))
        row.paste(img, (0, 0))
        row.paste(cm_gt, (img.width, 0))
        row.paste(cm_pseudo, (img.width * 2, 0))
        row.paste(over_ps, (img.width * 3, 0))
        row.save(os.path.join(outdir, f'ps_{i:02d}.jpg'))
        tiles.append(row)
    grid = make_grid(tiles, cols=2)
    if grid is not None:
        grid.save(os.path.join(outdir, 'pseudo_grid.jpg'))


@torch.no_grad()
def vis_pred(cfg_path: str, ckpt: str, split: str = 'val', num: int = 8, outdir: str = 'out_vis/pred', seed: int = 0):
    # lazy import to avoid heavy deps when not needed
    from models import build_seg_model

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = build_dataset(cfg['data'], split=split)
    model = build_seg_model(cfg['model'], num_seen_classes=cfg['data']['num_seen']).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state['model'], strict=False)
    model.eval()

    ensure_dir(outdir)
    random.seed(seed)
    idxs = random.sample(range(len(ds)), k=min(num, len(ds)))
    tiles = []
    for i, idx in enumerate(idxs):
        x, y, _ = ds[idx]
        img = to_pil_uint8(x)
        xb = x.unsqueeze(0).to(device)
        out = model(xb)
        logits = out['logits']
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.int64)
        cm_gt = colorize_mask(y.numpy(), num_classes=cfg['data']['num_seen'])
        cm_pd = colorize_mask(pred, num_classes=cfg['data']['num_seen'])
        over_pd = overlay(img, cm_pd, alpha=0.45)
        row = Image.new('RGB', (img.width * 4, img.height), (0, 0, 0))
        row.paste(img, (0, 0))
        row.paste(cm_gt, (img.width, 0))
        row.paste(cm_pd, (img.width * 2, 0))
        row.paste(over_pd, (img.width * 3, 0))
        row.save(os.path.join(outdir, f'pred_{i:02d}.jpg'))
        tiles.append(row)
    grid = make_grid(tiles, cols=2)
    if grid is not None:
        grid.save(os.path.join(outdir, 'pred_grid.jpg'))


@torch.no_grad()
def vis_attn(cfg_path: str, ckpt: str, image_path: str = None, out: str = 'out_vis/attn/attn.jpg', split: str = 'val', seed: int = 0):
    # lazy imports
    from models import build_seg_model
    from clip_tokens import CLIPTeacher

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_seg_model(cfg['model'], num_seen_classes=cfg['data']['num_seen']).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state['model'], strict=False)
    model.eval()

    teacher = CLIPTeacher(model_name=cfg['clip']['backbone'], pretrained=cfg['clip'].get('pretrained', 'openai'),
                          device=device, bank_size=cfg['clip'].get('bank_size', 24),
                          temperature=cfg['loss'].get('temperature', 0.07), num_seen=cfg['data']['num_seen'])

    if image_path is None:
        # sample from dataset
        ds = build_dataset(cfg['data'], split=split)
        random.seed(seed)
        idx = random.randrange(len(ds))
        x, y, _ = ds[idx]
        img = to_pil_uint8(x)
    else:
        # load single image and resize to cfg size
        img = Image.open(image_path).convert('RGB')
        sz = cfg['data'].get('img_size', 512)
        img = img.resize((sz, sz))
        x = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)

    xb = x.unsqueeze(0).to(device)
    out_s = model(xb)
    proj = out_s['proj']  # [1,C,L]
    Cg, _ = teacher.encode_image_dense(xb)
    Cg = F.normalize(Cg, dim=-1)  # [1,C]
    proj = F.normalize(proj, dim=1)  # [1,C,L]
    C = proj.shape[1]
    # weights: [1,L]
    W_attn = torch.softmax(torch.einsum('bc,bcl->bl', Cg, proj) / math.sqrt(C), dim=-1)
    L = W_attn.shape[1]
    Hf = out_s['feat'].shape[-2]
    Wf = out_s['feat'].shape[-1]
    heat = W_attn.view(1, 1, Hf, Wf)
    heat = F.interpolate(heat, size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
    heat_np = heat.cpu().numpy()
    heat_np = (heat_np - heat_np.min()) / (heat_np.max() - heat_np.min() + 1e-6)
    # colorize heatmap as red overlay
    cmap = np.zeros((heat_np.shape[0], heat_np.shape[1], 3), dtype=np.uint8)
    cmap[..., 0] = (heat_np * 255).astype(np.uint8)
    heat_img = Image.fromarray(cmap)
    over = overlay(img, heat_img, alpha=0.5)
    ensure_dir(os.path.dirname(out))
    # side-by-side: img | heat | overlay
    row = Image.new('RGB', (img.width * 3, img.height), (0, 0, 0))
    row.paste(img, (0, 0))
    row.paste(heat_img, (img.width, 0))
    row.paste(over, (img.width * 2, 0))
    row.save(out)


def main():
    parser = argparse.ArgumentParser(description='Visualization utilities')
    sub = parser.add_subparsers(dest='mode', required=True)

    p1 = sub.add_parser('dataset', help='可视化数据集样本: 原图/GT/叠加')
    p1.add_argument('--config', required=True)
    p1.add_argument('--split', default='train', choices=['train', 'val'])
    p1.add_argument('--num', type=int, default=8)
    p1.add_argument('--outdir', default='out_vis/dataset')

    p2 = sub.add_parser('pseudo', help='可视化教师伪标签 Yp')
    p2.add_argument('--config', required=True)
    p2.add_argument('--split', default='train', choices=['train', 'val'])
    p2.add_argument('--num', type=int, default=8)
    p2.add_argument('--outdir', default='out_vis/pseudo')

    p3 = sub.add_parser('pred', help='可视化学生模型预测')
    p3.add_argument('--config', required=True)
    p3.add_argument('--ckpt', required=True)
    p3.add_argument('--split', default='val', choices=['train', 'val'])
    p3.add_argument('--num', type=int, default=8)
    p3.add_argument('--outdir', default='out_vis/pred')

    p4 = sub.add_parser('attn', help='可视化全局对齐权重热力图 (W_attn)')
    p4.add_argument('--config', required=True)
    p4.add_argument('--ckpt', required=True)
    p4.add_argument('--image', default=None, help='可选，若不指定则随机抽样一张验证图')
    p4.add_argument('--out', default='out_vis/attn/attn.jpg')

    args = parser.parse_args()

    if args.mode == 'dataset':
        vis_dataset(args.config, split=args.split, num=args.num, outdir=args.outdir)
    elif args.mode == 'pseudo':
        vis_pseudo(args.config, split=args.split, num=args.num, outdir=args.outdir)
    elif args.mode == 'pred':
        vis_pred(args.config, ckpt=args.ckpt, split=args.split, num=args.num, outdir=args.outdir)
    elif args.mode == 'attn':
        vis_attn(args.config, ckpt=args.ckpt, image_path=args.image, out=args.out)
    else:
        raise ValueError('unknown mode')


if __name__ == '__main__':
    main()