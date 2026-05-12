import math

import torch
from typing import Optional
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION, get_constant_schedule_with_warmup


def get_lr_scheduler(
        name: Optional[str],
        optimizer: torch.optim.Optimizer,
        **kwargs,
):
    if name == "cosine":
        if 'total_iters' in kwargs:
            kwargs['T_max'] = kwargs.pop('total_iters')
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **kwargs
        )
    elif name == "cosine_with_restarts":
        if 'total_iters' in kwargs:
            kwargs['T_0'] = kwargs.pop('total_iters')
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, **kwargs
        )
    elif name == "step":

        return torch.optim.lr_scheduler.StepLR(
            optimizer, **kwargs
        )
    elif name == "constant":
        if 'factor' not in kwargs:
            kwargs['factor'] = 1.0

        return torch.optim.lr_scheduler.ConstantLR(optimizer, **kwargs)
    elif name == "linear":

        return torch.optim.lr_scheduler.LinearLR(
            optimizer, **kwargs
        )
    elif name == 'constant_with_warmup':
        total_iters = kwargs.get('total_iters', kwargs.get('max_iterations'))
        if total_iters is None:
            raise ValueError("constant_with_warmup requires total_iters to compute 5% warmup")
        total_iters = max(1, int(total_iters))
        warmup_steps = max(1, int(math.ceil(total_iters * 0.05)))
        configured_warmup_steps = kwargs.get('num_warmup_steps')
        if configured_warmup_steps != warmup_steps:
            print(
                "Overriding num_warmup_steps "
                f"from {configured_warmup_steps} to {warmup_steps} "
                f"(5% of {total_iters} total optimizer steps)"
            )
        kwargs['num_warmup_steps'] = warmup_steps
        kwargs.pop('total_iters', None)
        kwargs.pop('max_iterations', None)
        return get_constant_schedule_with_warmup(optimizer, **kwargs)
    else:
        # try to use a diffusers scheduler
        print(f"Trying to use diffusers scheduler {name}")
        try:
            name = SchedulerType(name)
            schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
            return schedule_func(optimizer, **kwargs)
        except Exception as e:
            print(e)
            pass
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, step, linear or constant"
        )
