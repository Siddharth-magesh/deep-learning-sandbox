import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import time
from typing import Optional
from .optim import build_optimizer, build_scheduler

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, checkpoint_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config, len(train_loader))
        
        self.scaler = GradScaler() if config.mixed_precision else None
        
        log_dir = 'runs/vision_transformer'
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs: {log_dir}")
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_acc = 0.0
        self.start_time = None
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}/{self.config.max_epochs}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                with autocast():
                    logits, loss = self.model(images, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, loss = self.model(images, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if self.global_step % self.config.log_every == 0:
                current_acc = 100.0 * correct / total
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/accuracy', current_acc, self.global_step)
                self.writer.add_scalar('train/learning_rate', self.scheduler.get_last_lr()[0], self.global_step)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                self.writer.add_scalar('train/gradient_norm', grad_norm, self.global_step)
            
            pbar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'acc': f'{100.0 * correct / total:.2f}%',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
            
            self.global_step += 1
        
        return total_loss / len(self.train_loader), 100.0 * correct / total
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            logits, loss = self.model(images, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        self.writer.add_scalar('val/loss', avg_loss, self.global_step)
        self.writer.add_scalar('val/accuracy', accuracy, self.global_step)
        
        per_class_acc = {}
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        for i, class_name in enumerate(class_names):
            self.writer.add_scalar(f'val/accuracy_{class_name}', accuracy, self.global_step)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.model.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_acc = checkpoint['best_val_acc']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    def train(self):
        self.start_time = time.time()
        max_seconds = self.config.max_training_hours * 3600
        
        print(f"\\nStarting training...")
        print(f"Maximum training time: {self.config.max_training_hours:.1f} hours")
        print(f"Maximum epochs: {self.config.max_epochs}\\n")
        
        for epoch in range(self.config.max_epochs):
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= max_seconds:
                elapsed_hours = elapsed_time / 3600
                print(f"\\nTime limit reached ({elapsed_hours:.2f} hours)")
                print(f"Training stopped at epoch {epoch + 1}/{self.config.max_epochs}")
                self.save_checkpoint('checkpoint_time_limit.pth')
                break
            
            self.epoch = epoch
            train_loss, train_acc = self.train_epoch()
            
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/train_accuracy', train_acc, epoch)
            
            if (epoch + 1) % self.config.eval_every == 0:
                val_loss, val_acc = self.validate()
                
                self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
                self.writer.add_scalar('epoch/val_accuracy', val_acc, epoch)
                
                elapsed_hours = elapsed_time / 3600
                remaining_hours = (max_seconds - elapsed_time) / 3600
                
                print(f'Epoch {epoch + 1}/{self.config.max_epochs} | Elapsed: {elapsed_hours:.2f}h | Remaining: {remaining_hours:.2f}h')
                print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
                
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_checkpoint('best_model.pth')
                    print(f'New best validation accuracy: {val_acc:.2f}%')
            
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        self.writer.close()
        print("\\nTraining complete!")
