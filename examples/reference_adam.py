import torch.nn as nn
import torch

'''
Simple case: y = sin(x) using competitive gradient descent.
'''

# Settings
lr = 1e-3

# Models
G = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1))

# Dataset
N = 100
x = torch.linspace(0,2*torch.pi,N).reshape(N,1)
y = torch.sin(x)

# Optimizer
optimizer = torch.optim.Adam(G.parameters(), lr=lr)

# Training loop
for i in range(10000):
    optimizer.zero_grad()

    g_out = G(x)

    loss = ((g_out - y)**2).mean()
    loss.backward()
    optimizer.step()

    print(loss.item())

