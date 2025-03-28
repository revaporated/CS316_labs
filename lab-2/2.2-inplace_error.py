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

    parameters -= step_size * parameters.grad
    print('final parameter =', parameters)

# expected

# parameters = tensor([3.], requires_grad=True)
# loss value = tensor(1., grad_fn=<PowBackward0>)
# grad = tensor([2.])
# Traceback (most recent call last):
#   File "C:\Users\e\Documents\school\CS316\lab_assignments\CS316_labs\lab-2\2.2-inplace_error.py", line 21, in <module>
#     parameters -= step_size * parameters.grad
# RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.