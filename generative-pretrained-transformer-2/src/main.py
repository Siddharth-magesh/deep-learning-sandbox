import argparse
import torch
from pathlib import Path
from .config import GPT2Config, TrainingConfig, DataConfig, OptunaConfig
from .models import GPT2Model
from .data import load_text_data, get_dataloaders
from .train import Trainer
from .evaluate import Evaluator
from .optim import OptunaOptimizer
from .utils import print_model_summary


def train_model(args):
    print("=" * 50)
    print("Training GPT-2 Model")
    print("=" * 50)
    
    data_config = DataConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        max_length=args.max_length
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        max_training_hours=args.max_training_hours,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )
    
    model_config = GPT2Config(
        vocab_size=50257,
        context_length=args.max_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff
    )
    
    print("\nLoading dataset...")
    train_dataset, val_dataset, test_dataset, tokenizer = load_text_data(data_config)
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, training_config)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    print("\nInitializing model...")
    model = GPT2Model(model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    print_model_summary(model)
    
    trainer = Trainer(model, train_loader, val_loader, training_config, args.checkpoint_dir)
    
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    print("\nStarting training...")
    trainer.train()
    
    print("\nTraining complete!")


def evaluate_model(args):
    print("=" * 50)
    print("Evaluating GPT-2 Model")
    print("=" * 50)
    
    if not Path(args.model_path).exists():
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return
    
    data_config = DataConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        max_length=args.max_length
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        device=args.device
    )
    
    print("\nLoading dataset...")
    _, _, test_dataset, tokenizer = load_text_data(data_config)
    _, _, test_loader = get_dataloaders(test_dataset, test_dataset, test_dataset, training_config)
    
    print(f"Test samples: {len(test_dataset)}")
    
    print("\nLoading model...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model = GPT2Model(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    evaluator = Evaluator(model, test_loader, args.device)
    
    print("\nEvaluating...")
    metrics = evaluator.evaluate()
    evaluator.print_metrics(metrics)


def optimize_hyperparameters(args):
    print("=" * 50)
    print("Optimizing Hyperparameters with Optuna")
    print("=" * 50)
    
    data_config = DataConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        max_length=args.max_length
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        device=args.device
    )
    
    optuna_config = OptunaConfig(
        n_trials=args.n_trials,
        study_name=args.study_name
    )
    
    base_config = GPT2Config()
    
    print("\nLoading dataset...")
    train_dataset, val_dataset, _, tokenizer = load_text_data(data_config)
    train_loader, val_loader, _ = get_dataloaders(train_dataset, val_dataset, val_dataset, training_config)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    optimizer = OptunaOptimizer(train_loader, val_loader, base_config, training_config, optuna_config)
    
    print("\nStarting optimization...")
    best_params = optimizer.optimize()
    
    print("\nOptimization complete!")


def main():
    parser = argparse.ArgumentParser(description='GPT-2 Training, Evaluation, and Optimization')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--dataset_name', type=str, default='wikitext', help='Dataset name')
    train_parser.add_argument('--dataset_config', type=str, default='wikitext-2-raw-v1', help='Dataset config')
    train_parser.add_argument('--max_length', type=int, default=768, help='Maximum sequence length')
    train_parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    train_parser.add_argument('--max_epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--max_training_hours', type=float, default=12.0, help='Maximum training time in hours')
    train_parser.add_argument('--d_model', type=int, default=640, help='Model dimension')
    train_parser.add_argument('--num_heads', type=int, default=10, help='Number of attention heads')
    train_parser.add_argument('--num_layers', type=int, default=8, help='Number of layers')
    train_parser.add_argument('--d_ff', type=int, default=2560, help='Feed-forward dimension')
    train_parser.add_argument('--checkpoint_dir', type=str, default='generative-pretrained-transformer-2/checkpoints', help='Checkpoint directory')
    train_parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    train_parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--dataset_name', type=str, default='wikitext', help='Dataset name')
    eval_parser.add_argument('--dataset_config', type=str, default='wikitext-2-raw-v1', help='Dataset config')
    eval_parser.add_argument('--max_length', type=int, default=768, help='Maximum sequence length')
    eval_parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    eval_parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    optim_parser = subparsers.add_parser('optimize', help='Optimize hyperparameters')
    optim_parser.add_argument('--dataset_name', type=str, default='wikitext', help='Dataset name')
    optim_parser.add_argument('--dataset_config', type=str, default='wikitext-2-raw-v1', help='Dataset config')
    optim_parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    optim_parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    optim_parser.add_argument('--n_trials', type=int, default=50, help='Number of optimization trials')
    optim_parser.add_argument('--study_name', type=str, default='gpt2_optimization', help='Study name')
    optim_parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'optimize':
        optimize_hyperparameters(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
