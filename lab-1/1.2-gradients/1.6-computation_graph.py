import torch
from torch import tensor
import numpy as np

f = lambda x: x**2
g = lambda u: 3*u + u**2
h = lambda z: z**3

parameter = tensor([2.0], requires_grad=True)

result = h(g(f(parameter)))
result.backward()

print('h(g(f(x))) =', result)
print('gradient is', parameter.grad)

# expected

# h(g(f(x))) = tensor([21952.], grad_fn=<PowBackward0>)
# gradient is tensor([103488.])