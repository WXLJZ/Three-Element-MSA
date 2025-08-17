from torch import nn
import torch

class LinearModel(nn.Module):
    """
    Linear models used for the sentiment_polarity/ metaphor_type representations
    """

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        """
        Returns an encoding of input X and a simple dropout-perturbed version of X
        For use in the SupConLoss calculation
        """

        last_state = torch.mul(x, attention_mask.unsqueeze(-1))
        features_summed = torch.sum(last_state, dim=1)
        dropped = self.dropout(features_summed)
        return torch.stack((self.layer_1(features_summed), self.layer_1(dropped)), 1)