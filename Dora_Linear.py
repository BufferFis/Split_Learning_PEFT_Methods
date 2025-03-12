import math
import torch
import torch.nn as nn

class Dora_Linear(nn.Module):
    """
    This module implements the DoRA update for a given frozen linear layer.
    It replaces the standard low-rank (A @ B) update with a sum over rank-1 components:
    
        W = W0 + sum_{i=1}^{r'} ( c_i * (x @ A_i) * B_i ) * scaling

    Here, A is of shape (in_features, r'), B is (r', out_features) and c is a vector of scalars.
    """
    def __init__(self, config, linear: nn.Linear):
        super().__init__()
        self.config = config
        self.linear = linear
        # Freeze the original weight and bias
        self.linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False

        # r' is the initial number of components (each rank-1)
        self.rank = config.adaptive_lora_start_rank  
        self.dropout = nn.Dropout(config.lora_dropout)
        # Instead of using nn.Linear layers, we directly create parameters
        self.A = nn.Parameter(torch.empty(linear.in_features, self.rank))
        self.B = nn.Parameter(torch.empty(self.rank, linear.out_features))
        # c holds a scalar for each rank-1 component; it will be updated (and pruned) during training
        self.c = nn.Parameter(torch.ones(self.rank, dtype=torch.float32))
        self.lora_scaling = config.lora_alpha / self.rank
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        # c is initialized to ones (i.e. all components active)

    def forward(self, x):
        # Compute the original frozen output
        hidden = self.linear(x)
        # Apply dropout and compute the rank-1 contributions:
        dropped = self.dropout(x)
        # Compute x @ A to get a (batch, rank) activation for each component
        x_A = torch.matmul(dropped, self.A)
        # Multiply each component by its scalar (element-wise along the rank dimension)
        x_A = x_A * self.c  
        # Project back to the output dimension via B and scale
        lora_update = torch.matmul(x_A, self.B) * self.lora_scaling
        return hidden + lora_update

    def importance_scores(self):
        """
        Compute an importance score for each rank-1 component.
        Here we use: score_i = |c_i| * ||A_i|| * ||B_i||
        which is proportional to the Frobenius norm of the component update.
        """
        scores = []
        for i in range(self.rank):
            a_i = self.A[:, i]
            b_i = self.B[i, :]
            score = torch.abs(self.c[i]) * torch.norm(a_i, p=2) * torch.norm(b_i, p=2)
            scores.append(score)
        return torch.stack(scores)

    def prune_components(self, num_to_keep):
        """
        Prune the components by keeping only the top 'num_to_keep' based on the importance score.
        Components not kept have their c set to zero.
        """
        scores = self.importance_scores()
        # Get indices sorted by descending score
        sorted_indices = torch.argsort(scores, descending=True)
        # Create a boolean mask: True for components to keep, False for components to prune
        keep_mask = torch.zeros_like(self.c, dtype=torch.bool)
        keep_mask[sorted_indices[:num_to_keep]] = True
        # Prune by zeroing out the c values for components not kept
        with torch.no_grad():
            self.c.data *= keep_mask.float()
