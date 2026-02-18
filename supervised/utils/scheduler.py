from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


class CustomLambdaLR(LambdaLR):
    """Linear warm-up scheduler."""

    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, lr_lambda=self.lr_lambda)

    def lr_lambda(self, step):
        return float(step) / max(1, self.warmup_steps)


class CombinedScheduler(_LRScheduler):
    """Two-phase scheduler: linear warm-up followed by cosine annealing."""

    def __init__(self, optimizer, scheduler1, scheduler2, lr_decay=1.0,
                 warmup_steps=100, start_cosine_step=100):
        self.optimizer = optimizer
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.warmup_steps = warmup_steps
        self.start_cosine_step = start_cosine_step
        self.step_num = 0
        self.lr_decay = lr_decay

    def step(self):
        if self.step_num < self.warmup_steps:
            self.scheduler1.step()
        elif self.step_num >= self.start_cosine_step:
            self.scheduler2.step()
            if self.lr_decay < 1.0 and (self.scheduler2.T_cur + 1 == self.scheduler2.T_i):
                self.scheduler2.base_lrs[0] *= self.lr_decay
        self.step_num += 1

    def state_dict(self):
        return {
            "warmup_steps": self.warmup_steps,
            "start_cosine_step": self.start_cosine_step,
            "step_num": self.step_num,
            "lr_decay": self.lr_decay,
            "scheduler1": self.scheduler1.state_dict() if self.scheduler1 else None,
            "scheduler2": self.scheduler2.state_dict() if self.scheduler2 else None,
        }

    def load_state_dict(self, state_dict):
        self.warmup_steps = state_dict["warmup_steps"]
        self.start_cosine_step = state_dict["start_cosine_step"]
        self.step_num = state_dict["step_num"]
        self.lr_decay = state_dict["lr_decay"]
        if self.scheduler1:
            self.scheduler1.load_state_dict(state_dict["scheduler1"])
        if self.scheduler2:
            self.scheduler2.load_state_dict(state_dict["scheduler2"])
