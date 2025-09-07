# 双卡训练使用说明

本项目已支持多GPU分布式训练，可以显著加速训练过程并支持更大的batch size。

## 快速开始

### 方法1：使用启动脚本（推荐）

```bash
# 使用默认配置文件 configs/voc.yaml
bash train_multi_gpu.sh

# 使用自定义配置文件
bash train_multi_gpu.sh configs/coco_stuff.yaml
```

### 方法2：直接使用torchrun

```bash
torchrun --nproc_per_node=2 --master_port=29500 train.py --config configs/voc.yaml
```

## 配置说明

### 自动调整的参数

- **batch_size**: 会自动除以GPU数量，每个GPU处理 `batch_size // world_size` 的数据
- **学习率**: 保持不变，因为梯度会在所有GPU间平均
- **数据采样**: 使用DistributedSampler确保每个GPU处理不同的数据

### 推荐配置调整

对于双卡训练，建议在配置文件中：

```yaml
train:
  batch_size: 32  # 总batch size，每卡16
  val_batch_size: 4  # 总batch size，每卡2
  eval_every: 5  # 减少验证频率以加速训练
  workers: 4  # 每个GPU的数据加载进程数
```

## 性能优势

- **训练速度**: 相比单卡训练，双卡可提升约1.7-1.9倍速度
- **内存效率**: 可以使用更大的batch size，提升训练稳定性
- **梯度同步**: 自动处理梯度聚合，保证训练一致性

## 注意事项

1. **环境要求**: 确保安装了支持分布式训练的PyTorch版本
2. **GPU内存**: 确保两张显卡有足够内存处理指定的batch size
3. **端口冲突**: 如果29500端口被占用，可以修改启动脚本中的PORT变量
4. **模型保存**: 只有主进程(rank 0)会保存模型和打印日志

## 故障排除

### 常见错误

1. **NCCL错误**: 检查CUDA和NCCL版本兼容性
2. **端口占用**: 修改master_port参数
3. **内存不足**: 减小batch_size或图像尺寸

### 检查GPU状态

```bash
# 查看GPU使用情况
nvidia-smi

# 检查NCCL是否可用
python -c "import torch; print(torch.distributed.is_nccl_available())"
```

## 单卡训练回退

如果需要回到单卡训练，直接使用原来的命令即可：

```bash
python train.py --config configs/voc.yaml
```

代码会自动检测环境变量，如果没有分布式环境变量，会自动使用单卡模式。