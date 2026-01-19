"""工具函数"""
import torch
import torch.nn as nn

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super().__init__()
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = (target == self.padding_idx)
            true_dist[mask] = 0
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))