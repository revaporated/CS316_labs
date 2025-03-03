import torch
from torch import tensor

w = tensor([9.0, 2.0], requires_grad=True)
f = lambda x: x[0] + x[1]**2

result = f(w)
    
print('result is ', result)

result.backward()

print('gradient is ', w.grad)