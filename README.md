# CLIP-to-Segmentation Distillation

基于CLIP的语义分割知识蒸馏项目，实现了从预训练CLIP模型到轻量级分割模型的知识转移。

## 项目概述

本项目实现了论文中提出的CLIP到分割模型的知识蒸馏方法，通过全局和局部特征对齐，将CLIP的视觉-语言理解能力转移到专门的分割网络中。

## 代码结构与论文对应关系

### 核心模型实现

#### `models.py`
- **SegStudent类**: 实现论文中的学生分割网络架构
  - `backbone`: EfficientNet特征提取器
  - `proj`: 特征投影层，将backbone特征映射到CLIP空间
  - `seg_head`: 分割预测头，输出像素级分类结果
  - `clip_proj`: CLIP特征投影层，用于特征对齐

#### `clip_tokens.py`
- **CLIPTeacher类**: 实现论文中的教师CLIP模型
  - `encode_image_dense()`: 提取密集的图像特征，对应论文中的全局特征提取
  - `encode_text()`: 文本编码，用于类别语义理解
  - 实现了图像预处理和特征归一化，确保与预训练CLIP模型的兼容性

### 损失函数实现

#### `loss.py`
- **info_nce_loss()**: 实现论文中的InfoNCE对比学习损失
  - 用于全局特征对齐，使学生网络学习CLIP的全局语义表示
  - 支持温度参数调节和负样本采样

### 训练流程

#### `train.py`
- **train_one_epoch()**: 实现论文中的完整训练流程
  - **全局蒸馏**: 通过InfoNCE损失对齐全局特征 (`loss_g`)
  - **局部蒸馏**: 通过注意力机制对齐局部特征 (`loss_l`)
    - 计算注意力权重: `W_attn = torch.softmax(torch.einsum('bc,bcl->bl', Cg, F_proj) / cfg['loss']['temperature'], dim=1)`
    - 局部特征聚合: `Fg = torch.einsum('bl,bcl->bc', W_attn, F_proj)`
  - **交叉熵损失**: 监督学习损失 (`loss_ce`)
  - **总损失**: `loss = w_global * loss_g + w_local * loss_l + w_ce * loss_ce`

- **validate()**: 模型验证，计算mIoU指标

### 数据处理

#### `datasets.py`
- **SimpleSegDataset**: 分割数据集加载器
  - 支持图像和掩码的同步加载
  - 实现数据增强和预处理
- **seg_collate()**: 批次数据整理函数

### 推理实现

#### `infer.py`
- 实现模型推理和结果可视化
- 支持单张图像的分割预测
- 包含结果着色和保存功能

### 工具函数

#### `utils.py`
- **seed_everything()**: 随机种子设置，确保实验可重现
- **AverageMeter**: 训练指标统计

## 论文方法对应

### 1. 双分支架构
- **教师分支**: `CLIPTeacher` (clip_tokens.py)
- **学生分支**: `SegStudent` (models.py)

### 2. 多层次知识蒸馏
- **全局特征对齐**: InfoNCE损失 (loss.py)
- **局部特征对齐**: 注意力机制蒸馏 (train.py)
- **任务特定监督**: 交叉熵损失 (train.py)

### 3. 特征空间对齐
- **投影层**: `proj` 和 `clip_proj` (models.py)
- **特征归一化**: L2归一化确保特征对齐

## 配置文件

- `configs/voc.yaml`: VOC数据集训练配置
- `configs/voc_toy.yaml`: 小规模测试配置
- `configs/coco_stuff.yaml`: COCO-Stuff数据集配置
- `configs/pcontext.yaml`: Pascal Context数据集配置

## 使用方法

### 训练
```bash
python train.py --config configs/voc.yaml
```

### 推理
```bash
python infer.py --model checkpoints/best.pth --image path/to/image.jpg
```

## 主要特性

1. **多尺度特征融合**: 结合全局和局部特征进行知识转移
2. **轻量级设计**: 学生网络使用EfficientNet，平衡精度和效率
3. **灵活配置**: 支持多种数据集和超参数配置
4. **完整流程**: 从训练到推理的完整实现

## 依赖环境

- PyTorch
- torchvision
- timm
- PIL
- numpy
- loguru
- yaml

本实现忠实地复现了论文中的核心思想，通过多层次的知识蒸馏实现了CLIP视觉理解能力向专用分割网络的有效转移。