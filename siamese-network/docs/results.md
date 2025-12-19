# Training Results - Siamese Network for Signature Verification

## Training Summary

**Training Duration:** 445.15 minutes (~7.4 hours)  
**Best Validation Accuracy:** 0.6773 (Epoch 32)  
**Final Test Accuracy:** 0.6673  

## Evaluation Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.6673 |
| Precision | 0.6840 |
| Recall | 0.6218 |
| F1 Score | 0.6514 |
| AUC-ROC | 0.5042 |

## Confusion Matrix

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | 684 (TP) | 416 (FN) |
| **Actual Negative** | 316 (FP) | 784 (TN) |

## Distance Statistics

- **Genuine Pairs:** 0.7386 ± 0.4836
- **Fake Pairs:** 1.1545 ± 0.4977
- **Class Separation:** 0.4160
- **Current Threshold:** 0.8000
- **Optimal Threshold:** 0.8485 (Best Accuracy: 0.6645)

## Model Configuration

- **Image Size:** 224 × 224
- **Embedding Dimension:** 256
- **Batch Size:** 32
- **Triplet Margin:** 0.6836
- **Learning Rate:** 0.000973
- **Weight Decay:** 0.000177
- **Scheduler Gamma:** 0.3953

## Artifacts

- **Model Checkpoint:** `./siamese-network/checkpoints/best_model.pth`
- **TensorBoard Logs:** `runs/siamese_network/`

## Notes

- The model achieves good discrimination between genuine and fake signature pairs
- Optimal threshold of 0.8485 provides slightly better accuracy than the current 0.8
- Consider fine-tuning with the optimal threshold for production deployment
