import torch
from torch import tensor

f = lambda x: x**2
g = lambda u: 3*u + u**2
h = lambda z: z**3

parameter = tensor([2.0], requires_grad=True)

r1 = f(parameter)
r1.backward()

print('f(x) =', r1)
print('grad of f(x) =', parameter.grad)
parameter.grad.zero_()

r2 = g(f(parameter))
r2.backward()

print('g(f(x)) =', r2)
print('grad of g(f(x)) =', parameter.grad)
parameter.grad.zero_()  

r3 = h(g(f(parameter)))
r3.backward()

print('h(g(f(x))) =', r3)
print('grad of h(g(f(x))) =', parameter.grad)

# expected

# f(x) = tensor([4.], grad_fn=<PowBackward0>)
# grad of f(x) = tensor([4.])
#
# g(f(x)) = tensor([28.], grad_fn=<AddBackward0>)
# grad of g(f(x)) = tensor([44.])
#
# h(g(f(x))) = tensor([21952.], grad_fn=<PowBackward0>)
# grad of h(g(f(x))) = tensor([103488.])
