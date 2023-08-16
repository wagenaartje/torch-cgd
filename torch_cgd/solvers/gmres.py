import torch

class GMRES:
    def __init__ (self, tol=1e-10, atol=1e-16):
        '''  Conjugate gradient method, implementation based on
            https://en.wikipedia.org/wiki/Conjugate_gradient_method.
            Only works for positive definite matrices A! '''

        self.tol = tol
        self.atol = atol
    
    def solve (self, A, b, x0=None, max_iter=1000):
        ''' Extremely important that none of the vectors requires grad. '''
        # Transform inputs
        b = torch.reshape(b, (-1, 1))

        # Initial guess
        if x0 is not None:
            x = x0.clone()
        else:
            x = torch.zeros(b.shape, device=b.device)

        r = b - A @ x
        r_tol = self.tol * torch.norm(b)

        new_v, rnorm = _safe_normalize(r)
        beta = torch.zeros(max_iter + 1, device=b.device)
        beta[0] = rnorm

        

        V = []
        V.append(new_v)
        H = torch.zeros((max_iter + 1, max_iter + 1), device=b.device)
        cs = torch.zeros(max_iter, device=b.device)  # cosine values at each step
        ss = torch.zeros(max_iter, device=b.device)  # sine values at each step
        
        for j in range(max_iter):
            p = A @ V[j]

            new_v = arnoldi(p, V, H, j + 1)  # Arnoldi iteration to get the j+1 th batch
            V.append(new_v)

            H, cs, ss = apply_given_rotation(H, cs, ss, j)
            beta[j + 1] = ss[j] * beta[j]
            beta[j] = cs[j] * beta[j]
            residual = torch.abs(beta[j + 1])
            if residual < r_tol or residual < self.atol:
                break

        if j == max_iter - 1:
            print('Warning: GMRES did not converge in {} iterations'.format(max_iter))

        # y = torch.linalg.solve_triangular(H[0:j + 1, 0:j + 1], beta[0:j + 1].unsqueeze(-1), upper=True)  # j x j
        # WARNING! Above version does not work on older pytorch versions. So we use deprecated one.
        y, _ = torch.triangular_solve(beta[0:j + 1].unsqueeze(-1), H[0:j + 1, 0:j + 1])
        V = torch.stack(V[:-1], dim=0)[:,:,0]

        sol = x + V.T @ y
        
        return sol
    
def _safe_normalize (x, threshold=None):
    norm = torch.norm(x)
    if threshold is None:
        threshold = torch.finfo(norm.dtype).eps
    normalized_x = x / norm if norm > threshold else torch.zeros_like(x)
    return normalized_x, norm

def arnoldi (vec, V, H, j):
    '''
    Arnoldi iteration to find the j th l2-orthonormal vector
    compute the j-1 th column of Hessenberg matrix
    '''

    for i in range(j):
        H[i, j - 1] = vec.T @ V[i]
        vec = vec - H[i, j-1] * V[i]
    new_v, vnorm = _safe_normalize(vec)
    H[j, j - 1] = vnorm
    return new_v

def apply_given_rotation (H, cs, ss, j):
    # apply previous rotation to the 0->j-1 columns
    for i in range(j):
        tmp = cs[i] * H[i, j] - ss[i] * H[i + 1, j]
        H[i + 1, j] = cs[i] * H[i+1, j] + ss[i] * H[i, j]
        H[i, j] = tmp
    cs[j], ss[j] = cal_rotation(H[j, j], H[j + 1, j])
    H[j, j] = cs[j] * H[j, j] - ss[j] * H[j + 1, j]
    H[j + 1, j] = 0
    return H, cs, ss



def cal_rotation (a, b):
    '''
    Args:
        a: element h in position j
        b: element h in position j+1
    Returns:
        cosine = a / \sqrt{a^2 + b^2}
        sine = - b / \sqrt{a^2 + b^2}
    '''
    c = torch.sqrt(a * a + b * b)
    return a / c, - b / c
