import os
from typing import Tuple, Dict, Any
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# --- New helpers for explicit class/image split control ---
def _read_int_list(path: str):
    with open(path, 'r') as f:
        content = f.read().strip().replace(',', ' ')
    return [int(x) for x in content.split() if x.strip()]


def _read_str_list(path: str):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


class SimpleSegDataset(Dataset):
    def __init__(self, root: str, split: str = 'train', img_size: int = 512, num_seen: int = 20,
                 seen_classes=None, unseen_classes=None, id_list=None,
                 img_subdir: str = None, mask_subdir: str = None,
                 img_exts=None, mask_ext: str = ".png", remap_seen: bool = True):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.remap_seen = remap_seen
        # If explicit class splits are provided, build a mapping from original class id -> [0..S-1]
        if seen_classes is not None:
            self.seen_classes = list(seen_classes)
            self.unseen_classes = list(unseen_classes) if unseen_classes is not None else []
            self.num_seen = len(self.seen_classes)
            if remap_seen:
                self.class_to_train_id = {c: i for i, c in enumerate(self.seen_classes)}
            else:
                self.class_to_train_id = None  # keep original mask ids for eval
        else:
            self.seen_classes = None
            self.unseen_classes = None
            self.class_to_train_id = None
            self.num_seen = num_seen
        # Directories: allow overriding subdirs (e.g., VOC: JPEGImages, SegmentationClass)
        self.img_dir = os.path.join(root, img_subdir) if img_subdir else os.path.join(root, 'images', split)
        self.mask_dir = os.path.join(root, mask_subdir) if mask_subdir else os.path.join(root, 'masks', split)
        # Extensions
        self.img_exts = tuple(img_exts) if img_exts is not None else ('.jpg', '.jpeg', '.png')
        self.mask_ext = mask_ext
        # Build ids
        if id_list is not None:
            self.ids = list(id_list)
        else:
            ids_from_dir = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir)
                            if f.lower().endswith(self.img_exts)]
            ids_from_dir.sort()
            self.ids = ids_from_dir
        # Transforms
        self.to_tensor = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])
        self.resize_mask = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        id_ = self.ids[idx]
        # image
        img_path = None
        for ext in self.img_exts:
            p = os.path.join(self.img_dir, id_ + ext)
            if os.path.isfile(p):
                img_path = p
                break
        if img_path is None:
            raise FileNotFoundError(f"Image not found for id {id_} with extensions {self.img_exts} in {self.img_dir}")
        img = Image.open(img_path).convert('RGB')
        # mask
        mask_path = os.path.join(self.mask_dir, id_ + self.mask_ext)
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = Image.open(mask_path)
        mask = self.resize_mask(mask)
        mask_np = np.array(mask, dtype=np.int64)
        # Remap labels: seen -> [0..S-1], others -> -1 (ignored) for training only
        if self.class_to_train_id is not None:
            train_mask = np.full_like(mask_np, fill_value=-1)
            for orig, tid in self.class_to_train_id.items():
                train_mask[mask_np == orig] = tid
            mask_t = torch.from_numpy(train_mask)
        else:
            mask_t = torch.from_numpy(mask_np)
        x = self.to_tensor(img)
        # Derive present seen class ids (already contiguous if remapped). For eval, keep empty list.
        if self.remap_seen and self.seen_classes is not None:
            uniques = torch.unique(mask_t)
            present_seen = uniques[(uniques >= 0) & (uniques < self.num_seen)]
        else:
            present_seen = torch.empty(0, dtype=torch.long)
        return x, mask_t, present_seen


def build_dataset(cfg: Dict[str, Any], split: str):
    name = cfg.get('name', 'simple')
    if name == 'simple':
        # Optional explicit class split controls
        seen_classes = cfg.get('seen_classes', None)
        if isinstance(seen_classes, str) and os.path.isfile(seen_classes):
            seen_classes = _read_int_list(seen_classes)
        unseen_classes = cfg.get('unseen_classes', None)
        if isinstance(unseen_classes, str) and os.path.isfile(unseen_classes):
            unseen_classes = _read_int_list(unseen_classes)
        # Optional explicit image id list per split (e.g., train_list/val_list)
        list_key = f'{split}_list'
        id_list = None
        if list_key in cfg and isinstance(cfg[list_key], str) and os.path.isfile(cfg[list_key]):
            id_list = _read_str_list(cfg[list_key])
        # Auto-detect VOC-style ImageSets if not provided
        if id_list is None:
            voc_list = os.path.join(cfg['root'], 'ImageSets', 'Segmentation', f'{split}.txt')
            if os.path.isfile(voc_list):
                id_list = _read_str_list(voc_list)
        return SimpleSegDataset(
            cfg['root'],
            split=split,
            img_size=cfg.get('img_size', 512),
            num_seen=cfg.get('num_seen', 20),
            seen_classes=seen_classes,
            unseen_classes=unseen_classes,
            id_list=id_list,
            img_subdir=cfg.get('img_subdir', None),
            mask_subdir=cfg.get('mask_subdir', None),
            img_exts=cfg.get('img_exts', None),
            mask_ext=cfg.get('mask_ext', '.png'),
            remap_seen=(split == 'train'),
        )
    else:
        raise NotImplementedError(name)


def seg_collate(batch):
    xs, ys, ids = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0)
    # keep variable-length present ids as list of tensors
    return xs, ys, list(ids)