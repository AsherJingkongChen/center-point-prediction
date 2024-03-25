from dataclasses import dataclass
from torch import optim


@dataclass
class HyperParameters:
    from torch import nn
    from torch.optim import lr_scheduler
    from torch import Tensor
    from typing import Callable

    HIDDEN_NODES: tuple[int, ...] = (
        5,
        8,
        11,
    )
    ACTIVATION_FUNCTIONS: tuple[nn.Module, ...] = (nn.Tanh, nn.ReLU)
    WEIGHT_INITIALIZERS: tuple[Callable[[Tensor], Tensor], ...] = (
        nn.init.normal_,
        nn.init.xavier_normal_,
        nn.init.kaiming_normal_,
    )
    LOSS_FUNCTIONS: tuple[Callable[[Tensor], Tensor], ...] = (nn.functional.mse_loss,)
    REGULARIZATION_FACTORS: tuple[float, ...] = (0.001, 0.0001)
    OPTIMIZERS: tuple[optim.Optimizer, ...] = (
        optim.SGD,
        lambda *args, **kwargs: optim.SGD(*args, **kwargs, momentum=0.9),
        optim.Adam,
    )
    LEARNING_EPOCHS: tuple[int, ...] = (100, 200, 300)
    LEARNING_RATE_SCHEDULERS: tuple[lr_scheduler.LRScheduler | None, ...] = (
        None,
        lr_scheduler.CosineAnnealingLR,
    )
    EMSEMBLE_COUNTS: tuple[int, ...] = (5,)
    NORMALIZERS: tuple[nn.BatchNorm1d | None, ...] = (None, nn.BatchNorm1d)
