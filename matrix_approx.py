import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn


from ipydex import IPS, activate_ips_on_exception

activate_ips_on_exception()

np.set_printoptions(linewidth=270, precision=4)


def ftensor(arr):
    return torch.from_numpy(arr).float()


class Net(torch.nn.Module):

    def __init__(self, n):
        super(Net, self).__init__()
        self.matrix = nn.Linear(n, n, bias=False)

    def forward(self, x):
        return self.matrix(x.T).T


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


