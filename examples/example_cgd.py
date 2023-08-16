import torch.nn as nn
import torch
import torch_cgd

'''
Simple case: y = sin(x) using competitive gradient descent.
'''

# Settings
lr = 1e-1

# Models
G = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1))
D = nn.Sequential(nn.Linear(1, 40), nn.ReLU(), nn.Linear(40, 1))

# Dataset
N = 100
x = torch.linspace(0,2*torch.pi,N).reshape(N,1)
y = torch.sin(x)

# Optimizer
solver = torch_cgd.solvers.conjugate_gradient(tol=1e-7, atol=1e-20)
optimizer = torch_cgd.CGD_CG(G.parameters(), D.parameters(), lr, solver=solver)
# Training loop
epoch = 0

for i in range(10000):
    optimizer.zero_grad()

    g_out = G(x)
    d_out = D(x)

    loss_d = (d_out* (g_out - y)).mean() # Discriminator: maximize
    loss_g = -loss_d             # Generator: minimize

    optimizer.step(loss_d, loss_g)

    print(loss_d.item(), torch.mean((g_out - y)**2).item())

