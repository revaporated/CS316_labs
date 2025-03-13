import torch
from torch import tensor

w = tensor([1.0], requires_grad=True)
f = lambda x: x**2
g = lambda u: torch.sqrt(u)

result = g(f(w))
result.backward()

print('result is', result)
print('gradient is', w.grad)

# expected
#
# result is tensor([1.], grad_fn=<SqrtBackward0>)
# gradient is tensor([1.])