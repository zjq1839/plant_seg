#!/usr/bin/env python3
"""
Test script to validate configuration and data loading
"""

import yaml
import torch
from datasets import build_dataset
from models import build_model
from clip_tokens import CLIPTeacher

def test_config():
    # Load configuration
    with open('configs/plant_fewshot_optimized.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    print("Configuration loaded successfully!")
    print(f"Data root: {cfg['data']['root']}")
    print(f"Model backbone: {cfg['model']['backbone']}")
    print(f"Shots: {cfg['data']['shots']}")
    
    # Test data loading
    print("\nTesting data loading...")
    train_set = build_dataset(cfg['data'], 'train')
    val_set = build_dataset(cfg['data'], 'test')  # Use test as validation
    
    print(f"Train dataset length: {len(train_set)}")
    print(f"Val dataset length: {len(val_set)}")
    
    # Test data loader
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=cfg['train']['batch_size'], 
        shuffle=True,
        num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set, 
        batch_size=cfg['train']['val_batch_size'], 
        shuffle=False,
        num_workers=2
    )
    
    print(f"Train loader length: {len(train_loader)}")
    print(f"Val loader length: {len(val_loader)}")
    
    # Test model creation
    print("\nTesting model creation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg['model'], cfg['data']['num_classes'])
    model = model.to(device)
    print(f"Model created successfully on {device}")
    
    # Test CLIP teacher
    print("\nTesting CLIP teacher...")
    clip_teacher = CLIPTeacher(
        backbone=cfg['clip']['backbone'],
        pretrained=cfg['clip']['pretrained'],
        bank_size=cfg['clip']['bank_size']
    )
    clip_teacher = clip_teacher.to(device)
    print("CLIP teacher created successfully")
    
    # Test forward pass with a sample
    print("\nTesting forward pass...")
    try:
        sample_batch = next(iter(train_loader))
        images = sample_batch['image'].to(device)
        masks = sample_batch['mask'].to(device)
        
        print(f"Sample batch - Images: {images.shape}, Masks: {masks.shape}")
        
        with torch.no_grad():
            outputs = model(images)
            print(f"Model output shape: {outputs.shape}")
            
        print("Forward pass successful!")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return False
    
    print("\n✅ All tests passed! Configuration is ready for training.")
    return True

if __name__ == '__main__':
    test_config()