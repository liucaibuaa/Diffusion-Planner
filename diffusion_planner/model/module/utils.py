import torch
import torch.nn as nn

class DynamicMLP(nn.module):
  def __init__(self, input_dim, hidden_dims, output_dim):
    super(DynamicMLP, self).__init__()
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]
    for i in range(len(dims) - 1):
      layers.append(nn.Linear(dims[i], dims[i + 1]))
      if i < len(dims) -2:
        layers.append(nn.GELU())
    self.mlp = nn.Sequential(*layers)

    def forward(self, x):
      return self.mlp(x)

