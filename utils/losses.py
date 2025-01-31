from torch import nn
import torch
from einops import repeat, rearrange
import torch.nn.functional as F


class OrdinalEncodingLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def to_classes(self, Phat: torch.Tensor):
        mode_class = Phat.argmax(1)
        Pc = torch.cumsum(Phat, 1)
        median_class = torch.sum(Pc < 0.5, 1)

        return mode_class, median_class

    def forward(self, logits: torch.Tensor, target: torch.Tensor, per_class=False):
        device = logits.device

        # Create ordinal targets
        num_classes_ = repeat(
            torch.arange(self.num_classes, device=device), "n -> b n", b=target.shape[0]
        )
        target_ = (target[:, None] > num_classes_).float()

        # Compute binary cross-entropy loss
        if per_class:
            loss = F.binary_cross_entropy_with_logits(logits, target_, reduction="none")
        else:
            loss = F.binary_cross_entropy_with_logits(logits, target_)

        # Convert mass distribution into probabilities
        Phat = torch.sigmoid(logits)
        Phat = torch.cat((1 - Phat[:, :1], Phat[:, :-1] - Phat[:, 1:], Phat[:, -1:]), 1)
        Phat = torch.clamp(Phat, 0, 1)
        mode_class, median_class = self.to_classes(Phat=Phat)

        return loss, mode_class
