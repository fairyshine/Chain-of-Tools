import torch
import torch.nn as nn
import torch.nn.functional as F

class PairwiseRankingLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.margin_positive = config.margin_positive
        self.margin_negative = config.margin_negative
        self.gamma = config.gamma

    def forward(self, scores, labels):
        mask = F.one_hot(labels, scores.shape[-1])
        positive_scores = scores.masked_fill(mask.eq(0), float('-inf')).max(dim=1)[0]
        negative_scores = scores.masked_fill(mask.eq(1), float('-inf')).max(dim=1)[0]
        positive_loss = torch.log1p(torch.exp(self.gamma*(self.margin_positive-positive_scores)))
        positive_loss[labels == 0] = 0.0  # exclusive `Other` loss
        negative_loss = torch.log1p(torch.exp(self.gamma*(self.margin_negative+negative_scores)))
        loss = torch.mean(positive_loss + negative_loss)
        return loss