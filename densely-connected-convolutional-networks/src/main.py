import os
import random
import numpy as np
import torch
import argparse
from pathlib import Path

from config.config import Config
from models.densenet import DenseNet
from data.dataset import build_dataloaders
from train import train
from evaluate import evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="DenseNet Training and Evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Mode: train or eval"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (optional)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for evaluation or resume training"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="densenet201",
        choices=["densenet121", "densenet169", "densenet201", "densenet264"],
        help="DenseNet variant to use"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "imagenet"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    
    args = parser.parse_args()
    
    cfg = Config()
    
    if args.model == "densenet121":
        cfg.model.name = "densenet121"
        cfg.model.block_layers = [6, 12, 24, 16]
    elif args.model == "densenet169":
        cfg.model.name = "densenet169"
        cfg.model.block_layers = [6, 12, 32, 32]
    elif args.model == "densenet201":
        cfg.model.name = "densenet201"
        cfg.model.block_layers = [6, 12, 48, 32]
    elif args.model == "densenet264":
        cfg.model.name = "densenet264"
        cfg.model.block_layers = [6, 12, 64, 48]

    cfg.data.dataset = args.dataset
    if args.dataset == "cifar10":
        cfg.model.num_classes = 10
        cfg.model.input_size = 32
    elif args.dataset == "cifar100":
        cfg.model.num_classes = 100
        cfg.model.input_size = 32
    elif args.dataset == "imagenet":
        cfg.model.num_classes = 1000
        cfg.model.input_size = 224

    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.lr is not None:
        cfg.optimizer.lr = args.lr
    
    set_seed(cfg.runtime.seed)
    
    os.makedirs(cfg.runtime.output_dir, exist_ok=True)
    
    print(f"Building {cfg.model.name} model...")
    model = DenseNet(cfg.model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
 
    if args.checkpoint is not None:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    if args.mode == "train":
        print("Building data loaders...")
        train_loader, val_loader = build_dataloaders(cfg)
        
        print("Starting training...")
        train(model, train_loader, val_loader, cfg)
        print("Training completed!")
        
    elif args.mode == "eval":
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for evaluation mode")
        
        print("Building data loaders...")
        _, test_loader = build_dataloaders(cfg)
        
        print("Starting evaluation...")
        metrics = evaluate(model, test_loader, cfg)
        
        print("\n=== Evaluation Results ===")
        print(f"Test Loss: {metrics['test_loss']:.4f}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        if 'test_top5_accuracy' in metrics:
            print(f"Test Top-5 Accuracy: {metrics['test_top5_accuracy']:.4f}")


if __name__ == "__main__":
    main()
