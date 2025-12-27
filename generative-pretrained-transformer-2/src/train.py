import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import math
import time
from typing import Optional
from torch.utils.data import DataLoader
from .models import GPT2Model
from .config import TrainingConfig
from .optim import get_cosine_schedule_with_warmup
from .utils import calculate_perplexity


class Trainer:
    def __init__(self, model: GPT2Model, train_loader: DataLoader, val_loader: DataLoader,
                 config: TrainingConfig, checkpoint_dir: Optional[str] = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay
        )
        
        self.total_steps = len(train_loader) * config.max_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=self.total_steps
        )
        
        self.scaler = GradScaler() if config.mixed_precision else None
        
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir='runs/generative-pretrained-transformer-2')
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.start_time = None  # Will be set when training starts
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.config.mixed_precision:
                with autocast():
                    _, loss = self.model(input_ids, labels)
                    loss = loss / self.config.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.config.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                _, loss = self.model(input_ids, labels)
                loss = loss / self.config.accumulation_steps
                loss.backward()
                
                if (step + 1) % self.config.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            total_loss += loss.item() * self.config.accumulation_steps
            
            if self.global_step % self.config.log_every == 0:
                self.writer.add_scalar('train/loss', loss.item() * self.config.accumulation_steps, self.global_step)
                self.writer.add_scalar('train/learning_rate', self.scheduler.get_last_lr()[0], self.global_step)
                self.writer.add_scalar('train/perplexity', math.exp(min(loss.item() * self.config.accumulation_steps, 20)), self.global_step)
            
            if self.global_step % self.config.eval_every == 0 and self.global_step > 0:
                val_loss = self.validate()
                self.writer.add_scalar('validation/loss', val_loss, self.global_step)
                self.writer.add_scalar('validation/perplexity', calculate_perplexity(val_loss), self.global_step)
                
                current_loss = loss.item() * self.config.accumulation_steps
                self.writer.add_scalar('train/loss', current_loss, self.global_step)
                self.writer.add_scalar('train/learning_rate', self.scheduler.get_last_lr()[0], self.global_step)
                self.writer.add_scalar('train/perplexity', calculate_perplexity(current_loss), self.global_step)
                
                self.model.train()
            
            if self.global_step % self.config.save_every == 0 and self.global_step > 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pth')
            
            progress_bar.set_postfix({'loss': loss.item() * self.config.accumulation_steps})
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.config.mixed_precision:
                with autocast():
                    _, loss = self.model(input_ids, labels)
            else:
                _, loss = self.model(input_ids, labels)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self):
        self.start_time = time.time()
        max_seconds = self.config.max_training_hours * 3600
        
        print(f"\nStarting training...")
        print(f"Maximum training time: {self.config.max_training_hours:.1f} hours")
        print(f"Maximum epochs: {self.config.max_epochs}\n")
        
        for epoch in range(self.config.max_epochs):
            # Check if time limit exceeded
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= max_seconds:
                elapsed_hours = elapsed_time / 3600
                print(f"\n⏰ Time limit reached ({elapsed_hours:.2f} hours)")
                print(f"Training stopped at epoch {epoch + 1}/{self.config.max_epochs}")
                self.save_checkpoint('checkpoint_time_limit.pth')
                break
            
            self.epoch = epoch
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            elapsed_hours = elapsed_time / 3600
            remaining_hours = (max_seconds - elapsed_time) / 3600
            
            print(f'Epoch {epoch + 1}/{self.config.max_epochs} | Elapsed: {elapsed_hours:.2f}h | Remaining: {remaining_hours:.2f}h')
            print(f'Train Loss: {train_loss:.4f} | Train Perplexity: {calculate_perplexity(train_loss):.4f}')
            print(f'Val Loss: {val_loss:.4f} | Val Perplexity: {calculate_perplexity(val_loss):.4f}')
            
            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        self.writer.close()
    
    def save_checkpoint(self, filename: str):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.model.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f'Checkpoint saved: {filename}')
    
    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f'Checkpoint loaded from {filepath}')
