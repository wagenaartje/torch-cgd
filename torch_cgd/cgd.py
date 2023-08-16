import torch
import torch.autograd as autograd

from . import utils

# So main problem is that we can't deal with flattened parameters, because that would require concatenating which copies memory.

class CGD:
    def __init__ (self, x_params, y_params, learning_rate, solver=torch.linalg):
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
        # x minimizes f
        # y minimizes g

        # First order partial derivatives
        df_dx = autograd.grad(f, self.x_params, retain_graph=True, create_graph=True)
        dg_dy = autograd.grad(g, self.y_params, retain_graph=True, create_graph=True)

        # Second order partial derivatives
        Df_xy = torch.zeros((self.n_x, self.n_y))
        for i in range(self.n_x):
            result = autograd.grad(utils.ith_element(df_dx, i), self.y_params, retain_graph=True)
            Df_xy[i] = torch.cat([r.flatten() for r in result])

        Dg_yx = torch.zeros((self.n_y, self.n_x))
        for i in range(self.n_y):
            result = autograd.grad(utils.ith_element(dg_dy, i), self.x_params, retain_graph=True)
            Dg_yx[i] = torch.cat([r.flatten() for r in result])

        with torch.no_grad():
            # Flatten first order partial derivatives
            df_dx = torch.cat([r.flatten() for r in df_dx])
            dg_dy = torch.cat([r.flatten() for r in dg_dy])

            # Construct the linear system
            A1 = torch.eye(self.n_x) - self.eta**2 * Df_xy @ Dg_yx
            b1 = df_dx - self.eta * Df_xy @ dg_dy

            A2 = torch.eye(self.n_y) - self.eta**2 * Dg_yx @ Df_xy
            b2 = dg_dy - self.eta * Dg_yx @ df_dx
            
            # Solve linear system
            dx = - self.eta * self.solver.solve(A1, b1)
            dy = - self.eta * self.solver.solve(A2, b2)

            # Update parameters
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
