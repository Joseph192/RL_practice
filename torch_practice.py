import torch
x = torch.tensor([3, 4], device='mps')
print(x.device)