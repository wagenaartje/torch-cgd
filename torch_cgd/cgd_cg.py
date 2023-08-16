import torch
import torch.autograd as autograd

from . import utils

class Operator:
    def __init__ (self, eta, first, second, first_params, second_params):
        self.eta = eta
        self.first = first
        self.second = second
        self.first_params = first_params
        self.second_params = second_params

    def __matmul__ (self, v):
        v1 = utils.vectorize(autograd.grad(self.first, self.first_params, v.view(-1), retain_graph=True))
        v2 = utils.vectorize(autograd.grad(self.second, self.second_params, v1, retain_graph=True))
        
        result = v.view(-1) - self.eta**2 * v2
        return result.view(-1,1)

class CGD_CG:
    # NOTE!!! We cannot use CG on general sum games, becuase matrix is not always positive definite
    def __init__ (self, x_params, y_params, learning_rate, solver=torch.linalg):
        ''' A CGD solver based on conjugate gradient descent! '''
        self.x_params = list(x_params)
        self.y_params = list(y_params)
        self.eta = learning_rate

        # Count number of parameters
        self.n_x = sum([torch.numel(x) for x in self.x_params])
        self.n_y = sum([torch.numel(y) for y in self.y_params])

        self.solver = solver

    def zero_grad (self) -> None:
        utils.zero_grad(self.x_params)
        utils.zero_grad(self.y_params)

    def step (self, f, g) -> None:
        # NOTE!!! We cannot use CG on general sum games, becuase matrix is not always positive definite
        # x minimizes f
        # y minimizes g

        # First order partial derivatives
        df_dx = autograd.grad(f, self.x_params, retain_graph=True, create_graph=True)
        df_dy = autograd.grad(f, self.y_params, retain_graph=True, create_graph=True)
        dg_dx = autograd.grad(g, self.x_params, retain_graph=True, create_graph=True)
        dg_dy = autograd.grad(g, self.y_params, retain_graph=True, create_graph=True)

        # Flatten first order partial derivatives
        df_dx = utils.vectorize(df_dx)
        df_dy = utils.vectorize(df_dy)
        dg_dx = utils.vectorize(dg_dx)
        dg_dy = utils.vectorize(dg_dy)

        # Construct the linear system
        A1 = Operator(self.eta, dg_dx, df_dy, self.y_params, self.x_params)
        b1 = df_dx - self.eta * utils.vectorize(autograd.grad(df_dy, self.x_params, dg_dy, retain_graph=True))

        # Solve linear system
        dx = - self.eta * self.solver.solve(A1, b1.detach()).view(-1)
        dy = - self.eta * (dg_dy + utils.vectorize(autograd.grad(dg_dx, self.y_params, dx,retain_graph=True)))

        # We could also have calculated dy like below, but less efficient.
        # A2 = Operator(self.eta, df_dy, dg_dx, self.x_params, self.y_params)
        # b2 = dg_dy - self.eta * vectorize(autograd.grad(dg_dx, self.y_params, df_dx, retain_graph=True))
        # dy = - self.eta * self.solver.solve(A2, b2).view(-1)

        # Update parameters
        with torch.no_grad():
            index = 0
            for p in self.x_params:
                size = p.numel()
                p.data.add_(dx[index:index + size].reshape(p.shape))
                index += size

            index = 0
            for p in self.y_params:
                size = p.numel()
                p.data.add_(dy[index:index + size].reshape(p.shape))
                index += size
