import argparse
import torch
from pathlib import Path

from .config import SwimConfig, DataConfig
from .models import SwinTransformer
from .data import TinyImageNetDataset, get_dataloader, train_transformation, test_transformation
from .train import Trainer
from .evaluate import Evaluator
from .utils import print_model_summary


def train_model(args):
    print("=" * 50)
    print("Training Swin Transformer")
    print("=" * 50)
    
    # Configurations
    data_config = DataConfig(
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    model_config = SwimConfig(
        image_size=args.image_size,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depths=args.depths,
        num_heads=args.num_heads,
        window_size=args.window_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size
    )
    
    # Data loading
    print("\nLoading dataset...")
    train_transform = train_transformation(
        mean=data_config.mean,
        std=data_config.std,
        image_size=data_config.image_size
    )
    test_transform = test_transformation(
        mean=data_config.mean,
        std=data_config.std,
        image_size=data_config.image_size
    )
    
    dataset = TinyImageNetDataset(transform=train_transform)
    train_dataset, _ = dataset.get_dataset_splits()
    
    # Use test transform for validation
    val_dataset_loader = TinyImageNetDataset(transform=test_transform)
    _, val_dataset = val_dataset_loader.get_dataset_splits()
    
    train_loader, val_loader = get_dataloader(
        train_dataset=train_dataset,
        test_dataset=val_dataset,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Model initialization
    print("\nInitializing model...")
    model = SwinTransformer(model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print_model_summary(model)
    
    # Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=model_config,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        max_epochs=args.max_epochs,
        max_training_hours=args.max_training_hours,
        gradient_clip=args.gradient_clip,
        mixed_precision=args.mixed_precision,
        device=args.device
    )
    
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    print("\nStarting training...")
    trainer.train()
    print("\nTraining complete!")


def evaluate_model(args):
    print("=" * 50)
    print("Evaluating Swin Transformer")
    print("=" * 50)
    
    if not Path(args.model_path).exists():
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return
    
    # Data configuration
    data_config = DataConfig(
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    # Load test data
    print("\nLoading dataset...")
    test_transform = test_transformation(
        mean=data_config.mean,
        std=data_config.std,
        image_size=data_config.image_size
    )
    
    dataset = TinyImageNetDataset(transform=test_transform)
    _, test_dataset = dataset.get_dataset_splits()
    
    _, test_loader = get_dataloader(
        train_dataset=test_dataset,  # Dummy, not used
        test_dataset=test_dataset,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory
    )
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    
    if "config" in checkpoint:
        model_config = checkpoint["config"]
    else:
        model_config = SwimConfig(
            image_size=args.image_size,
            num_classes=args.num_classes
        )
    
    model = SwinTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Evaluate
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        num_classes=args.num_classes,
        device=args.device
    )
    evaluator.evaluate()


def main():
    parser = argparse.ArgumentParser(description="Swin Transformer Training and Evaluation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    train_parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    train_parser.add_argument("--max_training_hours", type=float, default=24.0, help="Max training time (hours)")
    train_parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    train_parser.add_argument("--num_classes", type=int, default=200, help="Number of classes")
    train_parser.add_argument("--embed_dim", type=int, default=96, help="Embedding dimension")
    train_parser.add_argument("--depths", type=int, nargs="+", default=[2, 2, 6, 2], help="Depths per stage")
    train_parser.add_argument("--num_heads", type=int, nargs="+", default=[3, 6, 12, 24], help="Heads per stage")
    train_parser.add_argument("--window_size", type=int, default=7, help="Window size")
    train_parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping")
    train_parser.add_argument("--mixed_precision", action="store_true", default=True, help="Use mixed precision")
    train_parser.add_argument("--checkpoint_dir", type=str, default="shifted-window-transformers/checkpoints", help="Checkpoint directory")
    train_parser.add_argument("--log_dir", type=str, default="runs/swin_transformer", help="TensorBoard log directory")
    train_parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    train_parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    eval_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    eval_parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    eval_parser.add_argument("--num_classes", type=int, default=200, help="Number of classes")
    eval_parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model(args)
    elif args.command == "evaluate":
        evaluate_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
