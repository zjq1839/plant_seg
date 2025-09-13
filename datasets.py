import os
from typing import Tuple, Dict, Any
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from collections import defaultdict

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


# ========== New: Plant lesion dataset with few-shot sampling and joint augmentations ==========
class PlantLesionDataset(Dataset):
    """
    Binary lesion segmentation dataset.
    Assumes directory structure:
      root/
        images/{split}/*.jpg|png
        masks/{split}/*.png  (0=background, 1=lesion)
    Or use img_subdir/mask_subdir to match custom layouts.
    """
    def __init__(self, root: str, split: str = 'train', img_size: int = 512,
                 img_subdir: str = None, mask_subdir: str = None,
                 img_exts=None, mask_ext: str = '.png', id_list=None,
                 shots: int = 0, shots_per_class: bool = False, shots_group_by: str = None,
                 aug: bool = True):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.aug = bool(aug)
        # Resolve image/mask directories; allow subdir to be a base folder that contains split
        if img_subdir:
            candidate = os.path.join(root, img_subdir, split)
            if os.path.isdir(candidate):
                self.img_dir = candidate
            else:
                self.img_dir = os.path.join(root, img_subdir)
        else:
            self.img_dir = os.path.join(root, 'images', split)
        if mask_subdir:
            candidate = os.path.join(root, mask_subdir, split)
            if os.path.isdir(candidate):
                self.mask_dir = candidate
            else:
                self.mask_dir = os.path.join(root, mask_subdir)
        else:
            self.mask_dir = os.path.join(root, 'masks', split)
        self.img_exts = tuple(img_exts) if img_exts is not None else ('.jpg', '.jpeg', '.png')
        self.mask_ext = mask_ext
        # Ensure mask resize transform is available before few-shot selection
        self.mask_resize = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)
        # Build ids
        if id_list is not None:
            ids = list(id_list)
        else:
            ids = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir)
                   if f.lower().endswith(self.img_exts)]
            ids.sort()
        # Few-shot selection (all splits if shots>0)
        if shots and shots > 0:
            if shots_group_by == 'plant':
                ids = self._few_shot_by_plant(ids, shots)
            else:
                ids = self._few_shot_select(ids, shots, shots_per_class)
        self.ids = ids
        # Transforms: joint spatial + color aug for train
        if aug and split == 'train':
            self.img_trans = T.Compose([
                T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToTensor(),
            ])
        else:
            self.img_trans = T.Compose([
                T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
            ])
        # Geo aug hyper-params
        self.hflip_p = 0.5
        self.vflip_p = 0.2
        self.rotate_deg = 10.0
        self.rrc_scale = (0.8, 1.0)
        self.rrc_ratio = (0.9, 1.1)
        self.blur_p = 0.1
        self.gray_p = 0.05

    def _few_shot_select(self, ids, shots: int, per_class: bool):
        # Option 1: total-K sampling
        if not per_class:
            return ids[:shots] if shots <= len(ids) else ids
        # Option 2: K per class (bg/lesion presence)
        pos, neg = [], []
        for id_ in ids:
            mpath = os.path.join(self.mask_dir, id_ + self.mask_ext)
            if not os.path.isfile(mpath):
                continue
            m = Image.open(mpath)
            m = self.mask_resize(m)
            arr = np.array(m)
            if (arr == 1).any():
                pos.append(id_)
            else:
                neg.append(id_)
        random.shuffle(pos)
        random.shuffle(neg)
        keep = pos[:shots] + neg[:shots]
        if not keep:
            return ids[:shots]
        return keep

    def _few_shot_by_plant(self, ids, shots: int):
        """Select K images per plant species, where species is inferred from filename
        pattern 'plant_disease_index.*' -> species = token before first underscore.
        If a species has fewer than K images, take all available.
        """
        groups = defaultdict(list)
        for id_ in ids:
            # id_ is basename without extension
            if '_' in id_:
                plant = id_.split('_', 1)[0]
            else:
                plant = id_
            # ensure corresponding mask exists
            mpath = os.path.join(self.mask_dir, id_ + self.mask_ext)
            if not os.path.isfile(mpath):
                continue
            groups[plant].append(id_)
        selected = []
        for plant in sorted(groups.keys()):
            lst = groups[plant]
            random.shuffle(lst)
            selected.extend(lst[:shots])
        if not selected:
            return ids[:shots] if shots <= len(ids) else ids
        return selected

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
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

        # ===== Joint geometric augmentations (train only) =====
        if self.split == 'train' and self.aug:
            # Random Resized Crop (apply same params to image & mask)
            try:
                i, j, h, w = T.RandomResizedCrop.get_params(img, scale=self.rrc_scale, ratio=self.rrc_ratio)
                # Keep original size after crop to preserve resolution before final resize
                oh, ow = img.size[1], img.size[0]
                img = TF.resized_crop(img, i, j, h, w, size=(oh, ow), interpolation=T.InterpolationMode.BILINEAR)
                mask = TF.resized_crop(mask, i, j, h, w, size=(oh, ow), interpolation=T.InterpolationMode.NEAREST)
            except Exception:
                # Fallback: skip RRC if params fail
                pass
            # Horizontal flip
            if random.random() < self.hflip_p:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            # Vertical flip
            if random.random() < self.vflip_p:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            # Small rotation
            angle = random.uniform(-self.rotate_deg, self.rotate_deg)
            if abs(angle) > 1e-3:
                img = TF.rotate(img, angle, interpolation=T.InterpolationMode.BILINEAR, fill=0)
                mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST, fill=0)

        # Resize to training size
        mask = self.mask_resize(mask)
        img_t = self.img_trans(img)

        # Optional lightweight image-only augs (on tensor)
        if self.split == 'train' and self.aug:
            if random.random() < self.gray_p:
                img_t = T.RandomGrayscale(p=1.0)(img_t)
            if random.random() < self.blur_p:
                img_t = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))(img_t)

        mask_np = np.array(mask, dtype=np.int64)
        # Ensure binary {0,1}
        mask_np = (mask_np > 0).astype(np.int64)
        mask_t = torch.from_numpy(mask_np)
        # For compatibility, return empty present ids list
        present = torch.empty(0, dtype=torch.long)
        return img_t, mask_t, present


# New: Synthetic dataset for quick dry-run when data path is unavailable
class SyntheticLesionDataset(Dataset):
    def __init__(self, length: int = 8, img_size: int = 512, aug: bool = True):
        self.length = max(1, int(length))
        self.img_size = int(img_size)
        self.aug = bool(aug)
        # simple color jitter (works with tensor inputs in recent torchvision)
        self.jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05) if aug else None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        H = W = self.img_size
        # random image in [0,1]
        img_t = torch.rand(3, H, W)
        if self.jitter is not None:
            img_t = self.jitter(img_t)
        # synthetic circular lesion mask
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        yy = yy.float(); xx = xx.float()
        r = float(max(8, min(H, W) // 6))
        cy = float((idx * 37) % (H - 2*int(r)) + r)
        cx = float((idx * 53) % (W - 2*int(r)) + r)
        mask_t = (((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)).to(torch.long)
        present = torch.empty(0, dtype=torch.long)
        return img_t, mask_t, present


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
    elif name == 'plant_lesion':
        list_key = f'{split}_list'
        id_list = None
        if list_key in cfg and isinstance(cfg[list_key], str) and os.path.isfile(cfg[list_key]):
            id_list = _read_str_list(cfg[list_key])
        # Fallback to synthetic data if directories are missing (for quick dry-run)
        root = cfg['root']
        img_dir = os.path.join(root, cfg.get('img_subdir')) if cfg.get('img_subdir', None) else os.path.join(root, 'images', split)
        mask_dir = os.path.join(root, cfg.get('mask_subdir')) if cfg.get('mask_subdir', None) else os.path.join(root, 'masks', split)
        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            length = cfg.get('dummy_len', 8)
            return SyntheticLesionDataset(length=length, img_size=cfg.get('img_size', 512), aug=cfg.get('aug', True))
        # Determine shots by split (val/test fall back to train shots if not specified)
        if split == 'train':
            shots_for_split = cfg.get('shots', 0)
        elif split == 'val':
            shots_for_split = cfg.get('val_shots', cfg.get('shots', 0))
        else:  # test
            shots_for_split = cfg.get('test_shots', cfg.get('shots', 0))
        return PlantLesionDataset(
            root=cfg['root'],
            split=split,
            img_size=cfg.get('img_size', 512),
            img_subdir=cfg.get('img_subdir', None),
            mask_subdir=cfg.get('mask_subdir', None),
            img_exts=cfg.get('img_exts', None),
            mask_ext=cfg.get('mask_ext', '.png'),
            id_list=id_list,
            shots=shots_for_split,
            shots_per_class=cfg.get('shots_per_class', False),
            shots_group_by=cfg.get('shots_group_by', None),
            aug=cfg.get('aug', True),
        )
    else:
        raise NotImplementedError(name)


def seg_collate(batch):
    xs, ys, ids = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0)
    # keep variable-length present ids as list of tensors
    return xs, ys, list(ids)