import torch
import torch.nn as nn

class DynamicMLP(nn.Module):
  def __init__(self, in_features, hidden_features, out_features = None, drop = 0.0):
    super(DynamicMLP, self).__init__()
    layers = []
    out_features = out_features or in_features
    dims = [in_features] + [hidden_features] + [out_features]
    for i in range(len(dims) - 1):
      layers.append(nn.Linear(dims[i], dims[i + 1]))
      if i < len(dims) -2:
        layers.append(nn.GELU())
        if drop >0:
          layers.append(nn.Dropout(drop))
    self.mlp = nn.Sequential(*layers)

  def forward(self, x):
    return self.mlp(x)

class DropPath(nn.Module):
  def __init__(self, drop_prob = 0.0):
    super(DropPath, self).__init__()
    self.drop_prob = drop_prob

  def forward(self, x):
    if self.drop_prob == 0. or not self.training:
      return x
    keep_prob = 1- self.drop_prob
    shape = (x.shape[0], ) + (1, )*(x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype = x.dtype, device = x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor





