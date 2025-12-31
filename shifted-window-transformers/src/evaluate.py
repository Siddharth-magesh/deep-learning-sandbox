import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Optional


class Evaluator:
    """
    Evaluator for Swin Transformer.
    
    Handles model evaluation with per-class accuracy reporting.
    """
    
    def __init__(
        self,
        model,
        test_loader: DataLoader,
        num_classes: int = 200,
        class_names: Optional[List[str]] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self) -> tuple:
        """
        Run evaluation on test set.
        
        Returns:
            avg_loss: Average loss on test set
            accuracy: Overall accuracy percentage
        """
        total_loss = 0
        correct = 0
        total = 0
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        
        for images, labels in tqdm(self.test_loader, desc="Evaluating"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits, loss = self.model(images, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        
        print(f"\nTest Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.2f}%")
        
        # Per-class accuracy
        if self.class_names is not None:
            print(f"\nPer-class Accuracy (showing top 10):")
            class_accs = []
            for i in range(self.num_classes):
                if class_total[i] > 0:
                    class_acc = 100.0 * class_correct[i] / class_total[i]
                    class_accs.append((self.class_names[i] if i < len(self.class_names) else f"class_{i}", class_acc))
            
            # Sort by accuracy and show top/bottom
            class_accs.sort(key=lambda x: x[1], reverse=True)
            print("\nTop 5 classes:")
            for name, acc in class_accs[:5]:
                print(f"  {name:20s}: {acc:.2f}%")
            print("\nBottom 5 classes:")
            for name, acc in class_accs[-5:]:
                print(f"  {name:20s}: {acc:.2f}%")
        else:
            # Show summary statistics
            accuracies = [100.0 * class_correct[i] / class_total[i] 
                         for i in range(self.num_classes) if class_total[i] > 0]
            if accuracies:
                print(f"\nPer-class accuracy statistics:")
                print(f"  Mean: {sum(accuracies) / len(accuracies):.2f}%")
                print(f"  Min:  {min(accuracies):.2f}%")
                print(f"  Max:  {max(accuracies):.2f}%")
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def get_predictions(self, loader: DataLoader = None) -> tuple:
        """
        Get all predictions for a dataset.
        
        Args:
            loader: DataLoader to get predictions for (uses test_loader if None)
            
        Returns:
            all_predictions: Tensor of predicted labels
            all_labels: Tensor of true labels
            all_probs: Tensor of prediction probabilities
        """
        if loader is None:
            loader = self.test_loader
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        for images, labels in tqdm(loader, desc="Getting predictions"):
            images = images.to(self.device)
            
            logits, _ = self.model(images)
            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            
            all_predictions.append(predicted.cpu())
            all_labels.append(labels)
            all_probs.append(probs.cpu())
        
        return (
            torch.cat(all_predictions),
            torch.cat(all_labels),
            torch.cat(all_probs)
        )
