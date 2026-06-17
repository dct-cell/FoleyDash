"""
Integrate numerical values for some iterations
Typically used for loss computation / logging to tensorboard
Call finalize and create a new Integrator when you want to display/log
"""

import torch

from .logger import TensorboardLogger
from .distribute import world_size, is_rank0


class Integrator:
    def __init__(self, logger: TensorboardLogger, distributed: bool = True):
        self.values = {}
        self.counts = {}

        self.logger = logger
        self.distributed = distributed

    def add_scalar(self, key: str, x: torch.Tensor | int | float):
        if isinstance(x, torch.Tensor):
            x = x.detach()
            if x.dtype in [torch.long, torch.int, torch.bool]:
                x = x.float()

        if key not in self.values:
            self.counts[key] = 1
            self.values[key] = x
        else:
            self.counts[key] += 1
            self.values[key] += x

    def add_dict(self, tensor_dict: dict[str, torch.Tensor]):
        for k, v in tensor_dict.items():
            self.add_scalar(k, v)

    def reset(self):
        self.values = {}
        self.counts = {}

    # Average and output the metrics
    def finalize(self, prefix: str, it: int, ignore_timer: bool = False) -> None:
        # for the metrics
        outputs = {}
        for k, v in self.values.items():
            avg = v / self.counts[k]
            if self.distributed:
                # Inplace operation
                if isinstance(avg, torch.Tensor):
                    avg = avg.cuda()
                else:
                    avg = torch.tensor(avg).cuda()
                torch.distributed.reduce(avg, dst=0)

                if is_rank0:
                    avg = (avg / world_size).cpu().item()
                    outputs[k] = avg
            else:
                # Simple does it 
                outputs[k] = avg

        if (not self.distributed) or is_rank0:
            self.logger.log_metrics(prefix, outputs, it, ignore_timer=ignore_timer)
