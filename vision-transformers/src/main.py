import argparse
import torch
from pathlib import Path
from .config import ViTConfig, TrainingConfig, DataConfig
from .models import VisionTransformer
from .data import get_dataloaders
from .train import Trainer
from .evaluate import Evaluator
from .utils import print_model_summary


def train_model(args):
    print("=" * 50)
    print("Training Vision Transformer")
    print("=" * 50)
    
    data_config = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        max_training_hours=args.max_training_hours,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )
    
    model_config = ViTConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_size=args.mlp_size
    )
    
    print("\\nLoading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        root=data_config.data_dir,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print("\\nInitializing model...")
    model = VisionTransformer(model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    print_model_summary(model)
    
    trainer = Trainer(model, train_loader, val_loader, training_config, args.checkpoint_dir)
    
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    print("\\nStarting training...")
    trainer.train()
    
    print("\\nTraining complete!")


def evaluate_model(args):
    print("=" * 50)
    print("Evaluating Vision Transformer")
    print("=" * 50)
    
    if not Path(args.model_path).exists():
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return
    
    data_config = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    print("\\nLoading dataset...")
    _, _, test_loader = get_dataloaders(
        root=data_config.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    print("\\nLoading model...")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    
    if 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        model_config = ViTConfig()
    
    model = VisionTransformer(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    evaluator = Evaluator(model, test_loader, args.device)
    evaluator.evaluate()


def main():
    parser = argparse.ArgumentParser(description='Vision Transformer Training and Evaluation')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    train_parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    train_parser.add_argument('--max_epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--max_training_hours', type=float, default=24.0, help='Maximum training time in hours')
    train_parser.add_argument('--image_size', type=int, default=32, help='Image size')
    train_parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
    train_parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    train_parser.add_argument('--embedding_dim', type=int, default=192, help='Embedding dimension')
    train_parser.add_argument('--num_heads', type=int, default=3, help='Number of attention heads')
    train_parser.add_argument('--num_layers', type=int, default=12, help='Number of layers')
    train_parser.add_argument('--mlp_size', type=int, default=768, help='MLP size')
    train_parser.add_argument('--checkpoint_dir', type=str, default='vision-transformers/checkpoints', help='Checkpoint directory')
    train_parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    train_parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    eval_parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    eval_parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
