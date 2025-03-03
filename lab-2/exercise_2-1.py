import torch
from torch import tensor

def least_squares_loss(w):
    return (w - 2) ** 2

if __name__ == '__main__':
    # x = torch.tensor([[1, 2, 3]])
    # y = torch.tensor([[4, 8, 12]])

    x = [1, 2, 3]
    y = [4, 8, 12]

    x_p = tensor(x, dtype=torch.float32, requires_grad=True)
    y_p = tensor(y, dtype=torch.float32, requires_grad=True)

    parameters_init = [3.]
    parameters = tensor(parameters_init, dtype=torch.float32, requires_grad=True)
