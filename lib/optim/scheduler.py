import math
import torch
import numpy as np
from thop import profile
from thop import clever_format
from scipy.ndimage import map_coordinates

from torch.optim.lr_scheduler import _LRScheduler


class PolyLr(_LRScheduler):
    def __init__(self, optimizer, gamma, max_iteration, minimum_lr=0, warmup_iteration=0, last_epoch=-1):
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.minimum_lr = minimum_lr
        self.warmup_iteration = warmup_iteration

        super(PolyLr, self).__init__(optimizer, last_epoch)

    def poly_lr(self, base_lr, step):
        return (base_lr - self.minimum_lr) * ((1 - (step / self.max_iteration)) ** self.gamma) + self.minimum_lr

    def warmup_lr(self, base_lr, alpha):
        return base_lr * (1 / 10.0 * (1 - alpha) + alpha)

    def get_lr(self):
        if self.last_epoch < self.warmup_iteration:
            alpha = self.last_epoch / self.warmup_iteration
            lrs = [min(self.warmup_lr(base_lr, alpha), self.poly_lr(base_lr, self.last_epoch)) for base_lr in
                    self.base_lrs]
        else:
            lrs = [self.poly_lr(base_lr, self.last_epoch) for base_lr in self.base_lrs]

        return lrs


class CosineLr(_LRScheduler):
    """Cosine annealing + linear warmup. Mirrors PolyLr's constructor so the
    Train_patch.py call site is interchangeable. `gamma` is accepted but
    unused (kept for kwarg-compat with PolyLr)."""
    def __init__(self, optimizer, gamma=None, max_iteration=1, minimum_lr=0,
                 warmup_iteration=0, last_epoch=-1):
        self.max_iteration = max_iteration
        self.minimum_lr = minimum_lr
        self.warmup_iteration = warmup_iteration
        super().__init__(optimizer, last_epoch)

    def cosine_lr(self, base_lr, step):
        t = max(0, step - self.warmup_iteration)
        T = max(1, self.max_iteration - self.warmup_iteration)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(t / T, 1.0)))
        return self.minimum_lr + (base_lr - self.minimum_lr) * cos

    def warmup_lr(self, base_lr, alpha):
        return base_lr * (1 / 10.0 * (1 - alpha) + alpha)

    def get_lr(self):
        if self.last_epoch < self.warmup_iteration:
            alpha = self.last_epoch / max(1, self.warmup_iteration)
            return [self.warmup_lr(base_lr, alpha) for base_lr in self.base_lrs]
        return [self.cosine_lr(base_lr, self.last_epoch) for base_lr in self.base_lrs]