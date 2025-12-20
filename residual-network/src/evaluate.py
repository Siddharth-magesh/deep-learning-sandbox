import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np


class Evaluator:
    def __init__(self, model, test_loader, config, device, class_names):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.class_names = class_names
        self.model.eval()
    
    def evaluate(self):
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        
        all_preds = []
        all_targets = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluating")
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({'acc': 100.*correct/total})
        
        accuracy = 100. * correct / total
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(all_targets, all_preds, target_names=self.class_names, zero_division=0))
        
        print("\n" + "=" * 60)
        print("CONFUSION MATRIX")
        print("=" * 60)
        cm = confusion_matrix(all_targets, all_preds)
        print(cm)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_preds,
            'targets': all_targets,
            'confusion_matrix': cm
        }
        
        return results
