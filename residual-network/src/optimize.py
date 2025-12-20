import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd

from config import Config
from data_loader import create_data_loaders
from resnet100 import ResNet100


def objective(trial):
    config = Config()
    
    config.batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    config.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    config.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    config.img_size = trial.suggest_categorical('img_size', [64, 128, 224])
    
    scheduler_choice = trial.suggest_categorical('scheduler', ['StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'])
    
    config.num_epochs = 5
    
    device = torch.device(config.device)
    
    train_loader, val_loader, _, class_names = create_data_loaders(config)
    
    model = ResNet100(num_classes=len(class_names), img_channels=3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    if scheduler_choice == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    elif scheduler_choice == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    best_val_acc = 0.0
    
    for epoch in range(config.num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * correct / total
        
        if scheduler_choice == 'ReduceLROnPlateau':
            scheduler.step(val_acc)
        else:
            scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        trial.report(val_acc, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_acc


def run_optuna_study(n_trials=30, study_name='resnet100_optimization'):
    print("=" * 60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Running {n_trials} trials...")
    print("This may take a while...\n")
    
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    print(f"\nBest Trial: {study.best_trial.number}")
    print(f"Best Validation Accuracy: {study.best_value:.4f}%")
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    results_dir = Path("residual-network/optuna_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    df = study.trials_dataframe()
    csv_path = results_dir / "optuna_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    best_params_path = results_dir / "best_hyperparameters.txt"
    with open(best_params_path, 'w') as f:
        f.write("Best Hyperparameters from Optuna Study\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Validation Accuracy: {study.best_value:.4f}%\n\n")
        for key, value in study.best_params.items():
            f.write(f"{key} = {value}\n")
    
    print(f"✓ Best parameters saved to: {best_params_path}")
    
    print("\n" + "=" * 60)
    print("TOP 5 TRIALS")
    print("=" * 60)
    df_sorted = df.sort_values('value', ascending=False).head(5)
    print(df_sorted[['number', 'value', 'params_batch_size', 'params_learning_rate', 
                     'params_img_size', 'params_scheduler']].to_string(index=False))
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_optuna_study(n_trials=30)
