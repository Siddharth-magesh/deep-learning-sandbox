import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import time
from typing import Optional

from .optim import build_optimizer, build_scheduler
from .config import SwimConfig


class Trainer:
    """
    Trainer for Swin Transformer.
    
    Handles the training loop with mixed precision, gradient clipping,
    checkpointing, and TensorBoard logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: SwimConfig,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "runs/swin_transformer",
        max_epochs: int = 100,
        max_training_hours: float = 24.0,
        gradient_clip: float = 1.0,
        mixed_precision: bool = True,
        log_every: int = 100,
        device: str = "cuda"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.max_epochs = max_epochs
        self.max_training_hours = max_training_hours
        self.gradient_clip = gradient_clip
        self.mixed_precision = mixed_precision
        self.log_every = log_every
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = build_optimizer(
            self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            total_epochs=max_epochs,
            warmup_epochs=5,
            steps_per_epoch=len(train_loader)
        )
        
        self.scaler = GradScaler() if mixed_precision and torch.cuda.is_available() else None
        
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs: {log_dir}")
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_acc = 0.0
        self.start_time = None
    
    def train_epoch(self) -> tuple:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{self.max_epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with autocast():
                    logits, loss = self.model(images, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, loss = self.model(images, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if self.global_step % self.log_every == 0:
                current_acc = 100.0 * correct / total
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/accuracy", current_acc, self.global_step)
                self.writer.add_scalar("train/learning_rate", self.scheduler.get_last_lr()[0], self.global_step)
            
            pbar.set_postfix({
                "loss": f"{total_loss / (batch_idx + 1):.4f}",
                "acc": f"{100.0 * correct / total:.2f}%",
                "lr": f"{self.scheduler.get_last_lr()[0]:.6f}"
            })
            
            self.global_step += 1
        
        return total_loss / len(self.train_loader), 100.0 * correct / total
    
    @torch.no_grad()
    def validate(self) -> tuple:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits, loss = self.model(images, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        self.writer.add_scalar("val/loss", avg_loss, self.global_step)
        self.writer.add_scalar("val/accuracy", accuracy, self.global_step)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": self.config
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"Saved checkpoint: {self.checkpoint_dir / filename}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_acc = checkpoint["best_val_acc"]
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self) -> None:
        """Run full training loop."""
        self.start_time = time.time()
        
        print(f"\nStarting training for {self.max_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.mixed_precision}")
        
        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            
            # Check time limit
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours >= self.max_training_hours:
                print(f"\nTime limit reached ({self.max_training_hours}h). Saving checkpoint...")
                self.save_checkpoint("checkpoint_time_limit.pth")
                break
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint("best_model.pth")
                print(f"New best validation accuracy: {val_acc:.2f}%")
            
            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
        
        self.writer.close()
        print(f"\nTraining complete! Best validation accuracy: {self.best_val_acc:.2f}%")
