import torch

class LARS(torch.optim.Optimizer):
    """
    Minimal LARS wrapper: wraps an existing optimizer and scales gradients layer-wise.
    Equivalent idea to Apex's LARS for large-batch SSL.
    """
    def __init__(self, optimizer, trust_coefficient=0.001, eps=1e-8, clip=False):
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip

    @property
    def param_groups(self):
        return self.optim.param_groups

    def zero_grad(self, set_to_none: bool = False):
        self.optim.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        # Apply LARC scaling to grads
        for group in self.optim.param_groups:
            weight_decay = group.get("weight_decay", 0.0)
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                p_norm = torch.norm(p.data)
                g_norm = torch.norm(g.data)

                if p_norm > 0 and g_norm > 0:
                    # local_lr = trust * ||p|| / (||g|| + wd*||p|| + eps)
                    denom = g_norm + weight_decay * p_norm + self.eps
                    local_lr = self.trust_coefficient * (p_norm / denom)

                    if self.clip:
                        local_lr = min(local_lr / lr, 1.0)
                    else:
                        local_lr = local_lr / lr

                    g.data.mul_(local_lr)

        return self.optim.step(closure=closure)