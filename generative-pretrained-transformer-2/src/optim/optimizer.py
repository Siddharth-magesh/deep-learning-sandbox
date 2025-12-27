import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pathlib import Path
import math
from ..models import GPT2Model
from ..config import GPT2Config, TrainingConfig, OptunaConfig
from torch.utils.data import DataLoader


class OptunaOptimizer:
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, 
                 base_config: GPT2Config, training_config: TrainingConfig,
                 optuna_config: OptunaConfig):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.base_config = base_config
        self.training_config = training_config
        self.optuna_config = optuna_config
        
        self.results_dir = Path('optuna_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def objective(self, trial: optuna.Trial) -> float:
        model_config = GPT2Config(
            vocab_size=self.base_config.vocab_size,
            context_length=self.base_config.context_length,
            d_model=trial.suggest_categorical('d_model', [512, 768, 1024]),
            num_heads=trial.suggest_categorical('num_heads', [8, 12, 16]),
            num_layers=trial.suggest_int('num_layers', 6, 12),
            d_ff=trial.suggest_categorical('d_ff', [2048, 3072, 4096]),
            dropout=trial.suggest_float('dropout', 0.05, 0.2),
            attention_dropout=trial.suggest_float('attention_dropout', 0.05, 0.2),
            residual_dropout=trial.suggest_float('residual_dropout', 0.05, 0.2),
            layer_norm_epsilon=self.base_config.layer_norm_epsilon,
            initializer_range=self.base_config.initializer_range,
            use_bias=self.base_config.use_bias
        )
        
        train_config = TrainingConfig(
            batch_size=self.training_config.batch_size,
            learning_rate=trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True),
            weight_decay=trial.suggest_float('weight_decay', 0.0, 0.1),
            beta1=trial.suggest_float('beta1', 0.85, 0.95),
            beta2=trial.suggest_float('beta2', 0.95, 0.999),
            epsilon=self.training_config.epsilon,
            max_epochs=3,
            warmup_steps=trial.suggest_int('warmup_steps', 500, 2000),
            gradient_clip=trial.suggest_float('gradient_clip', 0.5, 2.0),
            accumulation_steps=self.training_config.accumulation_steps,
            save_every=10000,
            eval_every=200,
            log_every=self.training_config.log_every,
            checkpoint_dir=f'optuna_checkpoints/trial_{trial.number}',
            device=self.training_config.device,
            num_workers=self.training_config.num_workers,
            pin_memory=self.training_config.pin_memory,
            mixed_precision=self.training_config.mixed_precision,
            gradient_checkpointing=self.training_config.gradient_checkpointing
        )
        
        # Import Trainer here to avoid circular import
        from ..train import Trainer
        
        model = GPT2Model(model_config)
        trainer = Trainer(model, self.train_loader, self.val_loader, train_config)
        
        for epoch in range(train_config.max_epochs):
            trainer.epoch = epoch
            trainer.train_epoch()
            val_loss = trainer.validate()
            
            trial.report(val_loss, epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        final_val_loss = trainer.validate()
        
        return final_val_loss
    
    def optimize(self):
        if self.optuna_config.sampler == 'TPE':
            sampler = TPESampler(n_startup_trials=self.optuna_config.n_startup_trials)
        else:
            sampler = TPESampler()
        
        if self.optuna_config.pruner == 'MedianPruner':
            pruner = MedianPruner(n_startup_trials=self.optuna_config.n_startup_trials, 
                                 n_warmup_steps=self.optuna_config.n_warmup_steps)
        else:
            pruner = MedianPruner()
        
        study = optuna.create_study(
            study_name=self.optuna_config.study_name,
            direction=self.optuna_config.direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.optuna_config.storage,
            load_if_exists=True
        )
        
        study.optimize(
            self.objective,
            n_trials=self.optuna_config.n_trials,
            timeout=self.optuna_config.timeout
        )
        
        print("\n" + "=" * 50)
        print("Optimization Complete")
        print("=" * 50)
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best validation loss: {study.best_trial.value:.4f}")
        print(f"Best perplexity: {math.exp(min(study.best_trial.value, 20)):.4f}")
        print("\nBest hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        print("=" * 50 + "\n")
        
        results_file = self.results_dir / 'best_hyperparameters.txt'
        with open(results_file, 'w') as f:
            f.write(f"Best trial: {study.best_trial.number}\n")
            f.write(f"Best validation loss: {study.best_trial.value:.4f}\n")
            f.write(f"Best perplexity: {math.exp(min(study.best_trial.value, 20)):.4f}\n\n")
            f.write("Best hyperparameters:\n")
            for key, value in study.best_trial.params.items():
                f.write(f"  {key}: {value}\n")
        
        df = study.trials_dataframe()
        df.to_csv(self.results_dir / 'optuna_results.csv', index=False)
        
        return study.best_trial.params
