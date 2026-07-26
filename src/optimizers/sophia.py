"""
SophiaG optimizer — a second-order optimizer using diagonal Hessian estimates.

Based on: https://arxiv.org/abs/2305.14342
Original implementation from the AICT project.
"""

import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional


class SophiaG(Optimizer):
    """SophiaG optimizer with gradient preconditioning via Hessian diagonal."""

    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=1e-1, *, maximize=False, capturable=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, rho=rho,
                        weight_decay=weight_decay,
                        maximize=maximize, capturable=capturable)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("capturable", False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]["step"])
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    @torch.no_grad()
    def update_hessian(self):
        """Update the diagonal Hessian estimate for each parameter."""
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults["capturable"] else torch.tensor(0.)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if "hessian" not in state.keys():
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError("SophiaG does not support sparse gradients")
                grads.append(p.grad)
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults["capturable"] else torch.tensor(0.)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if "hessian" not in state.keys():
                    state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state["exp_avg"])
                state_steps.append(state["step"])
                hessian.append(state["hessian"])

                if self.defaults["capturable"]:
                    bs = torch.ones((1,), dtype=torch.float, device=p.device) * bs

            _sophiag(params_with_grad,
                     grads,
                     exp_avgs,
                     hessian,
                     state_steps,
                     bs=bs,
                     beta1=beta1,
                     beta2=beta2,
                     rho=group["rho"],
                     lr=group["lr"],
                     weight_decay=group["weight_decay"],
                     maximize=group["maximize"],
                     capturable=group["capturable"])

        return loss


def _sophiag(params, grads, exp_avgs, hessian, state_steps, *,
             bs, beta1, beta2, rho, lr, weight_decay, maximize, capturable):
    """Core SophiaG update rule."""

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda and bs.is_cuda

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)

        step_t += 1

        # Stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Update biased first moment estimate
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        if capturable:
            step_size = lr
            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            param.addcmul_(exp_avg.sign(), ratio, value=-step_size)
        else:
            step_size_neg = -lr
            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
