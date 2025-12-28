import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, test_loader, device='cuda'):
        self.model = model
        self.test_loader = test_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self):
        total_loss = 0
        correct = 0
        total = 0
        class_correct = [0] * 10
        class_total = [0] * 10
        
        for batch in tqdm(self.test_loader, desc='Evaluating'):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
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
        
        print(f"\\nTest Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"\\nPer-class Accuracy:")
        
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        for i in range(10):
            if class_total[i] > 0:
                class_acc = 100.0 * class_correct[i] / class_total[i]
                print(f"{classes[i]:12s}: {class_acc:.2f}%")
        
        return avg_loss, accuracy
