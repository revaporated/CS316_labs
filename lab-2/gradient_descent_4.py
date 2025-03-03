import torch
from torch import tensor

def silly_loss(w):
    return (w - 2) ** 2

if __name__ == '__main__':
    parameters_init = [3.]
    parameters = tensor(parameters_init, dtype=torch.float32, requires_grad=True)

    step_size = 0.01

    print('parameters = ', parameters)

    loss_value = silly_loss(parameters)
    print("loss value = ", loss_value)

    loss_value.backward()
    print('grad = ', parameters.grad)

    # tell PyTorch not to keep gradients in here
    with torch.no_grad():
        parameters -= step_size * parameters.grad

    print('parameters after one step = ', parameters)

    loss_value = silly_loss(parameters)
    print("loss value = ", loss_value)

    loss_value.backward()
    print('grad = ', parameters.grad)

    with torch.no_grad():
        parameters -= step_size * parameters.grad
    print('parameters after two steps = ', parameters)