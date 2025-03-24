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

    with torch.no_grad():
        parameters -= step_size * parameters.grad

        # correction to listing 4: zero out the gradient so it doesn't accumulate
        parameters.grad.zero_() 
    print('parameters after one step =', parameters)

    loss_value = silly_loss(parameters)
    print("loss value =", loss_value)

    loss_value.backward()
    print('grad =', parameters.grad)

    with torch.no_grad():
        parameters -= step_size * parameters.grad
    print('parameters after two steps =', parameters)

# expected

# parameters = tensor([3.], requires_grad=True)
# loss value = tensor([1.], grad_fn=<PowBackward0>)
# grad = tensor([2.])
# parameters after one step = tensor([2.9800], requires_grad=True)
# loss value = tensor([0.9604], grad_fn=<PowBackward0>)
# grad = tensor([1.9600])
# parameters after two steps = tensor([2.9604], requires_grad=True)