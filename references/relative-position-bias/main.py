import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import argparse
from pathlib import Path

from data import SyntheticImageDataset
from models import VisionTransformer
from experiments import (
    load_config,
    train_epoch,
    evaluate,
    save_checkpoint,
    visualize_attention_weights,
    visualize_relative_position_bias
)
from modules import RelativePositionBias


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main(config_path: str):
    setup_logging()
    
    config = load_config(config_path)
    device = torch.device(config['project']['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    torch.manual_seed(config['project']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['project']['seed'])
    
    window_size = tuple(config['relative_position_bias']['window_size'])
    img_size = int(window_size[0] * window_size[1] ** 0.5)
    patch_size = window_size[0]
    
    rpb_kwargs = {
        'num_heads': config['attention']['num_heads'],
        'window_size': window_size,
        'bias_type': config['relative_position_bias']['type'],
        'init_std': config['relative_position_bias']['init_std']
    }
    
    model = VisionTransformer(
        img_size=img_size * patch_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=10,
        embed_dim=config['input']['embed_dim'],
        depth=4,
        num_heads=config['attention']['num_heads'],
        mlp_ratio=4.0,
        dropout=config['attention']['dropout'],
        use_relative_position=config['relative_position_bias']['enabled'],
        rpb_kwargs=rpb_kwargs if config['relative_position_bias']['enabled'] else None
    ).to(device)
    
    train_dataset = SyntheticImageDataset(
        num_samples=1000,
        img_size=img_size * patch_size,
        patch_size=patch_size,
        in_channels=3
    )
    
    val_dataset = SyntheticImageDataset(
        num_samples=200,
        img_size=img_size * patch_size,
        patch_size=patch_size,
        in_channels=3
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['input']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['input']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    
    num_epochs = 5
    best_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        logging.info(f"Epoch {epoch}/{num_epochs}")
        logging.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        logging.info(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
                Path('checkpoints') / 'best_model.pth'
            )
    
    if config['experiment']['visualize_bias'] and config['relative_position_bias']['enabled']:
        logging.info("Visualizing relative position bias...")
        rpb = RelativePositionBias(**rpb_kwargs)
        bias = rpb()
        
        viz_dir = Path(config['visualization']['save_dir'])
        visualize_relative_position_bias(
            bias,
            save_path=viz_dir / 'relative_position_bias.png',
            cmap=config['visualization']['bias_cmap']
        )
        logging.info(f"Bias visualization saved to {viz_dir / 'relative_position_bias.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relative Position Bias Demo")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/rpb_config.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()
    
    main(args.config)
