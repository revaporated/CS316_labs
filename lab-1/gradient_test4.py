import torch
from torch import tensor
import numpy as np

w = tensor([3.0, 2.0], requires_grad=True)
f = lambda x: x**2
g = lambda u: u[0] + u[1]**2 

result = g(f(w))
result.backward()

print('result is ', result)
print('gradient is ', w.grad)