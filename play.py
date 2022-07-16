"""
for doing experiment
"""
import torch
import numpy as np

a = torch.ones((8,192,16,16))
b = a.permute((1,0,2,3))
c = b.reshape(192,1,-1)
print(c.size())