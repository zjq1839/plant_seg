# 植物病斑少样本分割系统

基于CLIP知识蒸馏的植物病斑少样本语义分割项目，实现了从预训练CLIP模型到轻量级分割模型的知识转移，专门用于植物病斑检测与分割。

## 项目概述

本项目专注于植物病斑的少样本语义分割任务，通过CLIP教师模型的全局和局部特征蒸馏，训练轻量级学生分割网络，在少量标注数据下实现高精度的病斑分割。

## 核心架构

### 教师-学生蒸馏框架

- **教师模型**: 预训练CLIP模型，提供丰富的视觉-语言语义知识
- **学生模型**: 轻量级分割网络，专门优化用于病斑分割任务
- **知识蒸馏**: 通过全局和局部特征对齐实现知识转移

## 代码结构

### 核心模块

#### `models.py` - 学生分割网络
- **SegStudent类**: 主要的学生分割模型
  - `backbone`: EfficientNet特征提取器
  - `proj`: 特征投影层，映射到CLIP特征空间
  - `seg_head`: 分割预测头，输出病斑/背景二分类结果
  - `clip_proj`: CLIP特征对齐投影层

#### `clip_tokens.py` - CLIP教师模型
- **CLIPTeacher类**: CLIP教师网络实现
  - `forward_tokens_and_pseudo()`: 生成全局特征、局部token和伪标签
  - `encode_text_labels()`: 编码类别文本（"lesion", "background"）
  - `info_nce_global()`: 计算全局蒸馏损失
  - 特征银行机制，维护类别特征的动态更新

#### `loss.py` - 损失函数
- **info_nce_loss()**: InfoNCE对比学习损失，用于特征对齐
- **dice_loss()**: Soft Dice损失，优化分割边界
- **CombinedSegLoss**: 组合交叉熵和Dice损失的分割损失

#### `train.py` - 训练流程
- **train_one_epoch_lesion()**: 病斑少样本训练的核心函数
  - **全局蒸馏**: 通过InfoNCE损失对齐图像级全局特征
  - **局部蒸馏**: 对齐病斑区域的局部特征，支持批量对比学习
  - **监督损失**: 结合交叉熵和Dice损失的分割监督
  - **损失权重**: `loss = w_global * loss_g + w_local * loss_l + w_seg * loss_seg`

- **validate_lesion()**: 病斑分割验证，计算Dice系数

### 数据处理

#### `datasets.py` - 数据集加载
- **SimpleSegDataset**: 植物病斑分割数据集
  - 支持图像和病斑掩码的加载
  - 数据增强：随机裁剪、翻转、颜色变换
  - 归一化处理，兼容CLIP预处理

#### `load_dataset.py` - 数据集构建
- **get_plant_lesion_loaders()**: 构建植物病斑数据加载器
- 支持训练/验证/测试集划分
- Few-shot采样策略

### 推理与可视化

#### `infer.py` - 模型推理
- 单张图像病斑分割预测
- 结果可视化和保存
- 支持批量推理

#### `visualize.py` - 结果可视化
- 病斑分割结果着色显示
- 对比原图、标注和预测结果

## 配置文件

### `configs/plant_fewshot.yaml` - 植物病斑少样本配置
```yaml
data:
  name: plant_lesion
  root: /path/to/plant_lesion_dataset
  num_classes: 2  # 病斑 + 背景
  
model:
  backbone: efficientnet_b0
  clip_model: ViT-B/32
  
loss:
  w_global: 1.0    # 全局蒸馏权重
  w_local: 1.0     # 局部蒸馏权重  
  w_seg: 1.0       # 分割监督权重
  temperature: 0.07
  
train:
  batch_size: 8
  lr: 1e-4
  epochs: 100
```

## 使用方法

### 环境安装
```bash
pip install torch torchvision timm
pip install pillow numpy loguru pyyaml
pip install ftfy regex
```

### 数据准备
植物病斑数据集应包含：
- `images/`: 植物图像文件
- `masks/`: 对应的病斑分割掩码（0=背景，1=病斑）
- `train.txt`, `val.txt`, `test.txt`: 数据集划分文件

### 训练模型
```bash
# 少样本训练
python train.py --config configs/plant_fewshot.yaml

# 指定验证集
python train.py --config configs/plant_fewshot.yaml --val_split test
```

### 模型推理
```bash
# 单张图像推理
python infer.py --model checkpoints/best.pth --image path/to/plant_image.jpg

# 批量推理
python infer.py --model checkpoints/best.pth --input_dir images/ --output_dir results/
```

### 结果可视化
```bash
python visualize.py --model checkpoints/best.pth --data_root /path/to/dataset
```

## 核心特性

### 1. 少样本学习能力
- 利用CLIP的预训练知识，在少量标注数据下实现高精度分割
- 特征银行机制，动态维护和更新类别特征表示

### 2. 多层次知识蒸馏
- **全局蒸馏**: 学习图像级的语义理解能力
- **局部蒸馏**: 专注于病斑区域的细粒度特征学习
- **批量对比学习**: 跨样本的特征对比，避免单样本退化

### 3. 轻量级设计
- EfficientNet backbone，平衡精度和推理速度
- 专门优化的分割头，适配病斑检测任务

### 4. 鲁棒的训练策略
- 组合损失函数：InfoNCE + 交叉熵 + Dice
- 自适应损失权重，平衡不同训练目标
- 详细的训练日志和调试信息

## 实验结果

在植物病斑数据集上的性能表现：
- **Dice系数**: ~0.63（少样本设置下）
- **推理速度**: ~50ms/张（GPU）
- **模型大小**: ~20MB（EfficientNet-B0）

## 技术亮点

1. **CLIP知识迁移**: 将大规模视觉-语言预训练知识应用于专业领域
2. **少样本优化**: 专门设计的few-shot学习策略和数据增强
3. **端到端训练**: 统一的蒸馏和分割训练框架
4. **实用性强**: 完整的训练、推理和可视化工具链

本项目为植物病害检测提供了一个高效、实用的少样本分割解决方案，特别适合农业AI应用场景。