from typing import Iterable
import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert name in self.shadow
                new_avg = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model: torch.nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone()
                p.data = self.shadow[name]

    def restore(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data = self.backup[name]
        self.backup = {}


def maybe_autocast(enabled: bool):
    class Dummy:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    if enabled:
        return torch.cuda.amp.autocast()
    return Dummy()

