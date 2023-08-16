import torch

def zero_grad (params):
    for p in params:
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()

def ith_element (params, index):
    for p in params:
        if torch.numel(p) < index + 1:
            index -= torch.numel(p)
        else:
            return p.flatten()[index]
        
def vectorize (l):
    return torch.cat([r.flatten() for r in l])