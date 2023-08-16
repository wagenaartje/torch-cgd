# torch-cgd 🤺
A fast and memory-efficient implementation of Competitive Gradient Descent (CGD) for PyTorch. The algorithm was originally proposed in [this paper](https://arxiv.org/abs/1905.12103), but a more robust and adaptive version was proposed in [this paper](https://arxiv.org/abs/1910.05852). This implementation is essentially a fork of [devzhk's `cgd-package`](https://github.com/devzhk/cgds-package), but the code has been heavily refactored for readability and customizability. You can install this package with `pip`: 

```
pip install torch-cgd
```

## Get started
You can use CGD for any competitive game of the form $\min_x f(x,y) \min_y g(x,y)$, i.e. games where players are minimizing objectives that are related and conflicting. While the possibilities are endless, you can also use it to replace your conventional loss function such as the `mse` loss with a competitive loss function. This can be beneficial because competitive loss functions can stimulate your network to have a more uniform error over the samples. The following code blocks show an example of this replacement for a network mapping $y=\sin(x)$.


### 1. Original, MSE-based gradient descent
```python
import torch.nn as nn
import torch

# Create the dataset
N = 100
x = torch.linspace(0,2*torch.pi,N).reshape(N,1)
y = torch.sin(x)

# Create the model
G = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1))

# Initialize the optimizer
optimizer = torch.optim.Adam(G.parameters(), lr=1e-3)

# Training loop
for i in range(10000):
    optimizer.zero_grad()

    g_out = G(x)

    loss = ((g_out - y)**2).mean() # Calculate mse
    loss.backward()
    optimizer.step()

    print(i, loss.item())
```


### 2. Competitive gradient descent
We now instead define the loss as $D(x) (G(x) - y)$, where the term within brackets is the error of the generator with respect to the target solution. In other words, the loss represents how well the discriminator is able to estimate the errors of the generator. As a result, a competitive game arises.

```python
import torch.nn as nn
import torch
import torch_cgd

# Create the dataset
N = 100
x = torch.linspace(0,2*torch.pi,N).reshape(N,1)
y = torch.sin(x)

# Create the models (D = discriminator, G = generator)
G = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1))
D = nn.Sequential(nn.Linear(1, 40), nn.ReLU(), nn.Linear(40, 1))

# Initialize the optimizer
solver = torch_cgd.solvers.GMRES(tol=1e-7, atol=1e-20)
optimizer = torch_cgd.ACGD_CG(G.parameters(), D.parameters(), 1e-3, solver=solver)

# Training loop
for i in range(10000):
    optimizer.zero_grad()

    g_out = G(x)
    d_out = D(x)

    loss_d = (d_out* (g_out - y)).mean() # Discriminator: maximize
    loss_g = -loss_d                     # Generator: minimize
    optimizer.step(loss_d)

    mse = torch.mean((g_out - y)**2).item() # Calculate mse
    print(i, mse)
```

## Examples
See the [examples folder](https://github.com/wagenaartje/torch-cgd/tree/main/examples).


## Cite
If you use this code for your research, please cite it as follows:

```
@misc{torch-cgd,
  author = {Thomas Wagenaar},
  title = {torch-cgd: A fast and memory-efficient implementation of competitive gradient descent in PyTorch},
  year = {2023},
  url = {https://github.com/wagenaartje/torch-cgd}
}
```