import torch
from torch.optim import Optimizer

from .privatizer import PrivUnitG, priv_unit_G_get_p

# Private SGD optimizer
class PrivateUnitSGD(Optimizer):
    def __init__(self, params, lr = 0.01, noise_fn = PrivUnitG, C = 1, epsilon = 1):
        """
        Initialize the private SGD optimizer with a custom noise injection function.
        See https://arxiv.org/abs/2306.15056.

        Args:
            params (iterable): Iterable of model parameters to optimize.
            lr (float, optional): Learning rate (default: 0.01).
            noise_fn (callable, optional): A function that takes the gradient tensor and returns
                a tensor of the same shape representing the injected noise (default: None).
            C (float, optional): Clipping parameter (default: 1).
            epsilon (float, optional): Privacy budget (default: 1).
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        
        p = priv_unit_G_get_p(epsilon)

        defaults = dict(lr=lr, noise_fn=noise_fn, C = C, epsilon = epsilon, p = p)
        super(PrivateUnitSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step with custom noise injection.

        Args:
            closure (callable, optional): A closure that reevaluates the model and computes the loss.
        Returns:
            loss: The current loss value after the optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            noise_fn = group['noise_fn']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                noisy_grad = noise_fn(grad, group["epsilon"], group["C"], group["p"])
                p.data -= group['lr'] * noisy_grad

        return loss