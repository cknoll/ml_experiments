"""
Created on 2020-01-09 18:00:00

@author: Carsten Knoll
"""

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init


from ipydex import IPS, activate_ips_on_exception

activate_ips_on_exception()

np.set_printoptions(linewidth=270, precision=4)


def ftensor(arr):
    return torch.from_numpy(arr).float()


def hh_matix(n, u):
    """
    Implement the housholder reflector
    """

    assert len(u.shape) == 1
    k = u.shape[0]
    I = torch.eye(n)

    m = n-k

    I[m:, m:] -= 2 * u.reshape(-1, 1)@u.reshape(1, -1)/u@u

    return I


torch.autograd.set_detect_anomaly(True)


# implement an own nn-module for SV-decomposed representation of a matrix
class SVDMatrix(torch.nn.modules.module.Module):
    """
    implement the method from:

    Jiong Zhang, Qi Lei,Inderjit S. Dhillon: "Stabilizing Gradients for Deep Neural Networks via Efficient SVD
    Parameterization"

    """

    __constants__ = ["n"]

    def __init__(self, n):
        super(SVDMatrix, self).__init__()
        self.n = n
        self.all_params = []
        self.singular_values = Parameter(torch.Tensor(n))
        self.all_params.append(self.singular_values)
        self.H1_stack = []
        self.H2_stack = []

        Pu = torch.eye(n)
        Pv = torch.eye(n)

        for i in range(1, n + 1):
            ui = Parameter(torch.Tensor(n + 1 - i))
            self.all_params.append(ui)
            vi = Parameter(torch.Tensor(i))
            self.all_params.append(vi)

            Hu_i = hh_matix(n, ui)
            Hv_i = hh_matix(n, vi)
            self.H1_stack.append(Hu_i)
            self.H2_stack.append(Hv_i)

            Pu = Pu@Hu_i
            Pv = Pv@Hv_i

        # this is the final matrix (M_{1,1} from Theorem 2 in the paper)
        self.matrix = Pu@torch.diag(self.singular_values)@Pv

        self.reset_parameters()

    def reset_parameters(self):
        for param_tensor in self.all_params:
            # init.kaiming_uniform_(param_tensor, a=np.sqrt(5))
            init.uniform_(param_tensor, -np.sqrt(5), np.sqrt(5))

    def forward(self, input):
        return F.linear(input, self.matrix, None)

    def extra_repr(self):
        return 'n={}'.format(self.n)


class Net(torch.nn.Module):

    def __init__(self, n):
        super(Net, self).__init__()
        # self.matrix = nn.Linear(n, n, bias=False)
        self.matrix = SVDMatrix(n)

    def forward(self, x):
        return self.matrix(x.T).T


if __name__ == "__main__":
    n = 5
    N_data = 100

    np.random.seed(1917)
    A = ftensor(np.random.rand(n, n))

    # generate training data
    X_train = ftensor(np.random.rand(n, N_data))


    Y_train = A@X_train


    # prepare the training
    net = Net(n)
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = nn.MSELoss()


    # do the training

    loss_over_epoch = []

    Nepochs = 2000

    for t in range(Nepochs):
        if t % int(Nepochs/100) == 0:
            print(t)

        prediction = net(X_train)     # input x and predict based on x
        IPS()

        loss = loss_func(prediction, Y_train)     # must be (1. nn output, 2. target)
        loss_over_epoch.append(loss.data.numpy())

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients


    plt.figure()
    plt.semilogy(loss_over_epoch)
    # plt.ioff()
    plt.show()


    IPS()


