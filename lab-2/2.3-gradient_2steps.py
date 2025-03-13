import torch
from torch import tensor

def silly_loss(w):
    return (w - 2) ** 2

if __name__ == '__main__':
    parameters_init = [3.]
    parameters = tensor(parameters_init, dtype=torch.float32, requires_grad=True)

    step_size = 0.01

    print('parameters =', parameters)

    loss_value = silly_loss(parameters)
    print("loss value =", loss_value)

    loss_value.backward()
    print('grad =', parameters.grad)

    parameters = parameters - step_size * parameters.grad
    print('parameters after one step =', parameters)

    loss_value = silly_loss(parameters)
    print("loss value =", loss_value)

    loss_value.backward()
    print('grad =', parameters.grad)

    parameters = parameters - step_size * parameters.grad
    print('parameters after two steps =', parameters)

# expected

# parameters = tensor([3.], requires_grad=True)
# loss value = tensor([1.], grad_fn=<PowBackward0>)
# grad = tensor([2.])
# parameters after one step = tensor([2.9800], grad_fn=<SubBackward0>)
# loss value = tensor([0.9604], grad_fn=<PowBackward0>)
# (pytorch warnings/errors...)
# Traceback (most recent call last):
#   File "C:\Users\e\Documents\school\CS316\lab_assignments\CS316_labs\lab-2\2.3-gradient_2steps.py", line 30, in <module>
#     parameters = parameters - step_size * parameters.grad
# TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'