import torch
import torch.autograd as autograd
import math

from . import solvers
from . import utils

class ACGD:
    def __init__ (self, x_params, y_params, lr=1e-3, beta=0.9, eps=1e-3, solver=None):
        '''
        Implements the Adaptive Competitive Gradient Descent (ACGD) optimizer as given by Algorithm 1 in "Implicit competitive regularization in GANs" by F. Schäfer et al.

        Args:
            x_params: Parameters of the model that minimizes f.
            y_params: Parameters of the model that maximizes f.
            lr: Learning rate.
            beta: Exponential decay rate for the second moment estimates.
            eps: Small constant for numerical stability.
            solver: Linear algebra solver to use to solve linear systems of equations.
        '''

        # Store arguments in class
        self.x_params = list(x_params)
        self.y_params = list(y_params)
        self.eta = lr
        self.beta = beta
        self.eps = eps

        if solver is None:
            self.solver = solvers.GMRES()
        else:
            self.solver = solver

        # Count number of parameters
        self.n_x = sum([torch.numel(x) for x in self.x_params])
        self.n_y = sum([torch.numel(y) for y in self.y_params])

        # Initialize second moment estimates
        self.vx = torch.zeros(self.n_x, device=self.x_params[0].device)
        self.vy = torch.zeros(self.n_y, device=self.y_params[0].device)

        # Initialize timestep
        self.timestep = 0
        self.prev_sol = None

    def zero_grad (self) -> None:
        ''' 
        Sets the gradients of all optimized tensors to zero.
        '''
        utils.zero_grad(self.x_params)
        utils.zero_grad(self.y_params)

    def step (self, f: torch.Tensor) -> None:
        '''
        Performs a single optimization step.

        Args: 
            f: Loss value to minimize and optimize simultaneously.
        '''

        # Increase the timestep
        self.timestep += 1

        # Compute first order partial derivatives
        df_dx = autograd.grad(f, self.x_params, retain_graph=True, create_graph=True)
        df_dy = autograd.grad(f, self.y_params, retain_graph=True, create_graph=True)
        
        # Flatten first order partial derivatives (this operation is expensive if requires_grad=True)
        df_dx = utils.vectorize(df_dx)
        df_dy = utils.vectorize(df_dy)

        # Update second moment estimates
        self.vx = self.beta * self.vx + (1 - self.beta) * df_dx.detach()**2
        self.vy = self.beta * self.vy + (1 - self.beta) * df_dy.detach()**2
        
        # Compute learning rates (with bias correction)
        bias_correction = 1 - self.beta**self.timestep
        eta_x = math.sqrt(bias_correction) * self.eta / (torch.sqrt(self.vx) + self.eps)
        eta_y = math.sqrt(bias_correction) * self.eta / (torch.sqrt(self.vy) + self.eps)

        # Construct the linear system
        if self.solver == torch.linalg:
            # Calculate Hessian matrices
            Df_xy = torch.zeros((self.n_x, self.n_y))
            for i in range(self.n_x):
                Df_xy[i] = utils.vectorize(autograd.grad(utils.ith_element(df_dx, i), self.y_params, retain_graph=True))

            Df_yx = torch.zeros((self.n_y, self.n_x))
            for i in range(self.n_y):
                Df_yx[i] = utils.vectorize(autograd.grad(utils.ith_element(df_dy, i), self.x_params, retain_graph=True))

            A1 = torch.eye(self.n_x) + eta_x.sqrt() * (Df_xy @ (eta_y * (Df_yx @ eta_x.sqrt())))
            b1 = eta_x.sqrt() * (df_dx + Df_xy @ (eta_y * df_dy))
        else:
            A1 = Operator(df_dx, df_dy, self.y_params, self.x_params, eta_y, eta_x)
            b1 = eta_x.sqrt() * (df_dx + utils.vectorize(autograd.grad(df_dy, self.x_params, eta_y * df_dy, retain_graph=True)))

        # Solve the linear system using given solver
        self.prev_sol = self.solver.solve(A1, b1.detach()) #, self.prev_sol)
        dx = - eta_x.sqrt() * self.prev_sol.view(-1)
        dy = eta_y * (df_dy + utils.vectorize(autograd.grad(df_dx, self.y_params, dx)))

        # We could also have calculated dy using its own linear system as given below.
        # However, since we already know what the minimizing model's "action" is, above
        # we can simply compute the maximizing model's "reaction" to it.
        #A2 = Operator(df_dy, df_dx, self.x_params, self.y_params, eta_x, eta_y)
        #b2 = eta_y.sqrt() * (df_dy - vectorize(autograd.grad(df_dx, self.y_params, eta_x * df_dx, retain_graph=True)))
        #dy = + eta_y.sqrt() * self.solver.solve(A2, b2).view(-1)

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

class Operator:
    def __init__ (self, first, second, first_params, second_params, first_eta, second_eta):
        '''
        In iterative matrix inverse algorithms, the to-be-inversed matrix is often multiplied with vectors. Since the matrix might be large, it is often better not to keep it in memory and instead calculate the matrix-vector product on the fly using a series of Hessian-vector products. This class implements such an operator. In the pseudocode (see ACGD class), this operator can represent the matrices

            (I + A^{1/2}_{x,t} D^2_{xy}f A_{y,t} + D^2_{yx}f A^{1/2}_{x,t})
            (I + A^{1/2}_{y,t} D^2_{yx}f A_{x,t} + D^2_{xy}f A^{1/2}_{y,t})

        Args:
            first: The first first derivative of the loss function.
            second: The second first derivative of the loss function.
            first_params: The parameters to take the second derivative of the first first derivative.
            second_params: The parameters to take the second derivative of the second first derivative.
            first_eta: Learning rates of the parameters used for the first derivative.
            second_eta: Learning rates of the parameters used for the second derivative.
        '''

        # Store arguments in class
        self.first = first
        self.second = second
        self.first_params = first_params
        self.second_params = second_params
        self.first_eta = first_eta
        self.second_eta = second_eta

    def __matmul__ (self, v: torch.Tensor) -> torch.Tensor:
        '''
        Performs a matrix-vector product.
        '''

        # From right to left. The equations in the comments assume the matrix used in the pseudocode to calculate dx. 
        # A^{1/2}_{y,t} v
        v0 = self.second_eta.sqrt() * v.view(-1)
        
        # First Hessian-vector product
        # A_{y,t} D^2_{xy}f v0
        v1 = self.first_eta * utils.vectorize(autograd.grad(self.first, self.first_params, v0, retain_graph=True))

        # Second Hessian-vector product
        # A^{1/2}_{x,t} D^2{xy}f v1
        v2 = self.second_eta.sqrt() * utils.vectorize(autograd.grad(self.second, self.second_params, v1, retain_graph=True)) 
        
        # I v + v2
        result = v.view(-1) + v2
        return result.view(-1,1)