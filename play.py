"""
for doing experiment
"""
import torch

f = torch.ones((16,)).view(2,1,8)
matrix = torch.tensor([1,2,3,4,5,6,7,8],dtype=torch.float32).view(1,8)
res = matrix+f
print(f)
print(res)