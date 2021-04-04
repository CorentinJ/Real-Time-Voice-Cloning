import numpy as np


class ScheduledOptim():
    ''' A simple wrapper class for learning rate scheduling '''

    def __init__(self, optimizer, decay_steps, n_warmup_steps, current_steps, init_lr):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = current_steps
        self.decay_steps = decay_steps
        self.init_lr = init_lr
        self.lr = init_lr

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def _get_lr(self):
        if self.n_current_steps < self.n_warmup_steps:
            return self.init_lr * float(self.n_current_steps) / float(max(1, self.n_warmup_steps))
        if self.n_current_steps % self.decay_steps == 0:
            return self.lr * 0.5
        else:
            return self.lr

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        self.lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.lr
