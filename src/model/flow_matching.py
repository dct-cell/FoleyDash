import logging
from typing import Callable, Literal

import torch
from torchdiffeq import odeint

log = logging.getLogger()


# Partially from https://github.com/gle-bellier/flow-matching
class FlowMatching:
    def __init__(
        self,
        min_sigma: float = 0.0,
        num_steps: int = 1,
    ):
        # num_steps: number of steps in the euler inference mode
        super().__init__()
        self.min_sigma = min_sigma
        self.num_steps = num_steps

    def get_conditional_flow(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        # which is psi_t(x), eq 22 in flow matching for generative models
        return (1 - t) * x0 + (1 - self.min_sigma) * t * x1

    def get_x1_xt(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        rng: torch.Generator = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = torch.empty_like(x0).normal_(generator=rng)
        xt = self.get_conditional_flow(x0, x1, t)
        return x1, xt

    def to_prior(self, fn: Callable, x0: torch.Tensor) -> torch.Tensor:
        return self.run_t0_to_t1(fn, x0, 0, 1)

    def to_data(self, fn: Callable, x1: torch.Tensor) -> torch.Tensor:
        return self.run_t0_to_t1(fn, x1, 1, 0)

    def run_t0_to_t1(
        self, fn: Callable, x: torch.Tensor, t0: float, t1: float
    ) -> torch.Tensor:
        """
        Take an x_t0 and solve x_t1.
        fn: a function that takes (t, x) and returns the direction x0->x1
        """
        steps = torch.linspace(t0, t1, self.num_steps + 1, device=x.device)
        for i in range(len(steps) - 1):
            t = steps[i]
            s = steps[i + 1]
            x = x + (s - t) * fn(t, s, x)

        # for i in range(self.num_steps):
        #     t = steps[i]
        #     s = steps[i + 1]
        #     x = x + (t1 - t) * fn(t, t1, x)
        #     if i < self.num_steps - 1:
        #         noise = torch.randn_like(x)
        #         x = (1 - s) * x + s * noise
        return x