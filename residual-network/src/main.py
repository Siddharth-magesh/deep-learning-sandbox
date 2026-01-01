import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from pathlib import Path

from config import Config
from data_loader import create_data_loaders
from resnet import ResNet50, ResNet100
from train import Trainer
from evaluate import Evaluator


def main():
    config = Config()
    config.display()
    
    device = torch.device(config.device)
    print(f"\n✓ Using device: {device}")
    
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    print("\n" + "=" * 60)
    print("DATA LOADING")
    print("=" * 60)
    train_loader, val_loader, test_loader, class_names = create_data_loaders(config)
    
    print("\n" + "=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)
    
    if config.model_name.lower() == "resnet50":
        model = ResNet50(num_classes=len(class_names), img_channels=3).to(device)
    elif config.model_name.lower() == "resnet100":
        model = ResNet100(num_classes=len(class_names), img_channels=3).to(device)
    else:
        raise ValueError(f"Unknown model: {config.model_name}. Choose 'resnet50' or 'resnet100'")
    
    print(f"✓ Using model: {config.model_name}")
    
    print("\nModel Summary:")
    summary(model, input_size=(config.batch_size, 3, config.img_size, config.img_size), 
            device=str(device), col_names=["input_size", "output_size", "num_params", "trainable"])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    scheduler = None
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    
    print("\n" + "=" * 60)
    print("PROFILING")
    print("=" * 60)
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('residual-network/profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        sample_input, _ = next(iter(train_loader))
        sample_input = sample_input.to(device)
        
        for _ in range(5):
            model(sample_input)
            prof.step()
    
    print("✓ Profiling complete. Results saved to residual-network/profiler_logs/")
    print("  View with: tensorboard --logdir=residual-network/profiler_logs")
    
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device)
    history = trainer.train()
    
    print("\n" + "=" * 60)
    print("LOADING BEST MODEL FOR EVALUATION")
    print("=" * 60)
    checkpoint_path = Path("residual-network/checkpoints/best_model.pth")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model from epoch {checkpoint['epoch']+1}")
    
    evaluator = Evaluator(model, test_loader, config, device, class_names)
    results = evaluator.evaluate()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Final Test Accuracy: {results['accuracy']:.2f}%")
    print("\nResults saved in residual-network/:")
    print("  - checkpoints/")
    print(f"  - runs/{config.model_name}/ (TensorBoard logs)")
    print("  - profiler_logs/ (Profiler data)")
    print("\nTo view TensorBoard:")
    print("  tensorboard --logdir=residual-network/runs")
    print("=" * 60)


if __name__ == "__main__":
    main()
