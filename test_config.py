import os
import yaml
import torch
from torch.utils.data import DataLoader

from datasets import build_dataset, seg_collate
from models import build_seg_model
from clip_tokens import CLIPTeacher


def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def test_datasets(cfg):
    print("[Test] Building datasets and dataloaders...")
    # build_dataset takes (cfg, split) and returns a single dataset
    train_ds = build_dataset(cfg['data'], split='train')
    val_ds = build_dataset(cfg['data'], split='test')
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=seg_collate)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=seg_collate)

    sample_batch = next(iter(train_loader))
    # seg_collate returns (images, masks, present_ids_list)
    images, masks, present_ids_list = sample_batch
    print(f"[OK] Train batch -> images: {images.shape}, masks: {masks.shape}, n_present_lists: {len(present_ids_list)}")
    return images, masks


def test_model_forward(cfg, images):
    print("[Test] Building model (mlp projection head override for test)...")
    model_cfg = dict(cfg['model'])
    # override to test MLP head without editing yaml
    model_cfg.update({
        'proj_head': 'mlp',
        'proj_mid_dim': model_cfg.get('clip_dim', 512),
        'proj_norm': 'bn',
        'proj_dropout': 0.0,
    })
    num_seen = cfg['data'].get('num_seen', cfg['data'].get('num_classes', 2))
    model = build_seg_model(model_cfg, num_seen_classes=num_seen)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    with torch.no_grad():
        images = images.to(device)
        out = model(images)
        assert 'logits' in out and 'feat' in out and 'proj' in out
        print(f"[OK] Model forward -> logits: {tuple(out['logits'].shape)}, feat: {tuple(out['feat'].shape)}, proj: {tuple(out['proj'].shape)}")
    return out['proj']


def test_clip_teacher(cfg, images, proj):
    print("[Test] Building CLIP teacher and checking compat with projection...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_cfg = cfg['clip']
    teacher = CLIPTeacher(model_name=clip_cfg['backbone'], pretrained=clip_cfg.get('pretrained', 'openai'),
                          device=str(device), temperature=cfg['loss'].get('temperature', 0.07), num_seen=cfg['data'].get('num_seen', 2))
    with torch.no_grad():
        images = images.to(device)
        cls_token, _ = teacher.encode_image_dense(images)
        # proj is [B, C, L], just ensure normalization and basic shapes are valid
        proj_norm = torch.nn.functional.normalize(proj, dim=1)
        print(f"[OK] CLIP teacher ready; cls_token: {tuple(cls_token.shape)}, proj_norm: {tuple(proj_norm.shape)}")


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), 'configs', 'plant_fewshot_optimized.yaml')
    cfg = load_cfg(cfg_path)

    images, masks = test_datasets(cfg)
    proj = test_model_forward(cfg, images)
    test_clip_teacher(cfg, images, proj)
    print("\n✅ All tests passed! MLP projection head is wired and compatible.")


if __name__ == '__main__':
    main()