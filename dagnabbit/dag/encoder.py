import torch
import torch.nn as nn
import torch.nn.functional as F


class DAGEncoder(nn.Module):
    def __init__(self, num_root_nodes, embedding_dim):
        super().__init__()
        self.num_root_nodes = num_root_nodes
        self.operators = nn.ModuleDict(
            {
                "NAND": None,
                "NOR": None,
                "LEAF": None,
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class NeuralOperator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # takes two vectors as inputs, outputs a single vector
        self.linear_1 = nn.Linear(input_dim * 2, input_dim * 4)
        self.linear_2 = nn.Linear(input_dim * 4, input_dim * 1)

        self.activation = nn.GELU()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Takes two vectors as inputs, outputs a single vector
        a: [batch_size, input_dim]
        b: [batch_size, input_dim]
        output: [batch_size, input_dim]
        """
        x = torch.cat([a, b], dim=1)

        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.activation(x)

        x = F.normalize(x, dim=1)

        return x
