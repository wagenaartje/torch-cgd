import torch

class conjugate_gradient:
    def __init__ (self, tol=1e-10, atol=1e-16):
        '''  Conjugate gradient method, implementation based on
            https://en.wikipedia.org/wiki/Conjugate_gradient_method.
            Only works for positive definite matrices A! '''

        self.tol = tol
        self.atol = atol
    
    def solve (self, A, b, x0=None):
        ''' Extremely important that none of the vectors requires grad. '''
        # Transform inputs
        b = torch.reshape(b, (-1, 1))

        # Initial guess
        if x0 is not None:
            x = x0.clone()
        else:
            x = torch.zeros(b.shape, device=b.device)

        # NOTE! Probably should be detaching much more! Also in ACGD.py

        r = b - A @ x
        p = r.clone()

        r_tol = self.tol * b.T @ b

        rdotr = r.T @ r
        
        while True:
            Ap = A @ p
            alpha = rdotr / (p.T @ Ap)
            x = x + alpha * p
            r = r - alpha * Ap

            rdotr_next = r.T @ r
            beta = rdotr_next / rdotr

            if rdotr_next < r_tol or rdotr_next < self.atol:
                break

            p = r + beta * p
            rdotr = rdotr_next

        return x
