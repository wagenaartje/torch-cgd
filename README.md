<img src="https://i.imgur.com/vzeaPSt.png">

### torch-cgd
<p><a href="https://badge.fury.io/py/torch-cgd"><img src="https://badge.fury.io/py/torch-cgd.svg" alt="PyPI version" height="18"></a>

A fast and memory-efficient implementation of Adaptive Competitive Gradient Descent (ACGD) for PyTorch. The non-adaptive version of the algorithm was originally proposed in [this paper](https://arxiv.org/abs/1905.12103), but the adaptive version was proposed in [this paper](https://arxiv.org/abs/1910.05852). This repository is essentially a fork of [devzhk's `cgd-package`](https://github.com/devzhk/cgds-package), but the code has been heavily refactored for readability and customizability. You can install this package with `pip`: 

```
pip install torch-cgd
```

## Get started
You can use ACGD for any competitive losses of the form $\min_x \min_y f(x,y)$, in other words those where one player tries to minimize the loss and another player tries to maximize the loss. You can for example use it to replace your conventional loss function such as the `mse` loss with a competitive loss function. This can be beneficial because competitive loss functions can stimulate your network to have a more uniform error over the samples, **leading to considerably lower losses although at a high computational cost.**

### Example
The following code block show an example of this replacement for a network trying to learn the function $y=\sin(x)$. Define the loss as $D(x) (G(x) - y)$, where the term within brackets is the error of the generator with respect to the target solution. In other words, the loss represents how well the discriminator is able to estimate the errors of the generator. As a result, a competitive game arises.

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
optimizer = torch_cgd.ACGD(G.parameters(), D.parameters(), 1e-3, solver=solver)

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

### Choosing the right solver
One of the steps in ACGD involves inverting a matrix, for which many different methods exist. This library offers two different solvers, namely the Conjugate Gradient method (CG) and the Generalized Minimum RESidual method (GMRES). You can initially them, for example, as follows:

```python
solver = torch_cgd.solvers.CG(tol=1e-7, atol=1e-20)
solver = torch_cgd.solvers.GMRES(tol=1e-7, atol=1e-20)
```

Which you can then pass to the ACGD optimizer as follows:

```python
optimizer = torch_cgd.ACGD(..., solver=solver)
```

From my own experience, the best results are obtained with GMRES. Currently, a direct solver is not available yet for ACGD, but it is for CGD. Note that using a direct solver is considerably slower and more memory intensive already for smaller network sizes. 

## More examples
See the [examples folder](https://github.com/wagenaartje/torch-cgd/tree/main/examples).


## Cite
If you use this code for your research, please cite it as follows:

```
@misc{torch-cgd,
  author = {Thomas Wagenaar},
  title = {torch-cgd: A fast and memory-efficient implementation of adaptive competitive gradient descent in PyTorch},
  year = {2023},
  url = {https://github.com/wagenaartje/torch-cgd}
}
```