import torch
from tqdm import tqdm
from typing import Dict
from torch.utils.data import DataLoader
from .models import GPT2Model
from .utils import calculate_perplexity


class Evaluator:
    def __init__(self, model: GPT2Model, test_loader: DataLoader, device: str = 'cuda'):
        self.model = model
        self.test_loader = test_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        for batch in tqdm(self.test_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            _, loss = self.model(input_ids, labels)
            
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = calculate_perplexity(avg_loss)
        
        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        print("\n" + "=" * 50)
        print("Evaluation Metrics")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"{key.capitalize()}: {value:.4f}")
        print("=" * 50 + "\n")
