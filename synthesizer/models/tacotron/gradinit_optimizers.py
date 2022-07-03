import math

import torch


class RescaleAdam(torch.optim.Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 min_scale=0, grad_clip=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, amsgrad=amsgrad, min_scale=min_scale, grad_clip=grad_clip)
        super(RescaleAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RescaleAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None, is_constraint=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grad_list = []
        alphas = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # State initialization
                amsgrad = group['amsgrad']
                state = self.state[p]
                if len(state) == 0:
                    state['alpha'] = 1.
                    state['init_norm'] = p.norm().item()
                    state['step'] = 0
                    state['cons_step'] = 0
                    # Exponential moving average of gradient values for the weight norms
                    state['exp_avg'] = 0
                    # Exponential moving average of squared gradient values for the weight norms
                    state['exp_avg_sq'] = 0
                    state['cons_exp_avg'] = 0
                    # state['cons_exp_avg_sq'] = 0
                    # if amsgrad:
                    #     # Maintains max of all exp. moving avg. of sq. grad. values
                    #     state['max_exp_avg_sq'] = 0
                # alphas.append(state['alpha'])

                curr_norm = p.data.norm().item()
                if state['init_norm'] == 0 or curr_norm == 0:
                    # pdb.set_trace()
                    continue # typical for biases

                grad = torch.sum(p.grad * p.data).item() * state['init_norm'] / curr_norm
                # grad_list.append(grad)

                if group['grad_clip'] > 0:
                    grad = max(min(grad, group['grad_clip']), -group['grad_clip'])
                # Perform stepweight decay
                # if group['weight_decay'] > 0:
                #     p.mul_(1 - group['lr'] * group['weight_decay'])
                beta1, beta2 = group['betas']
                if is_constraint:
                    state['cons_step'] += 1
                    state['cons_exp_avg'] = state['cons_exp_avg'] * beta1 + grad * (1 - beta1)
                    # state['cons_exp_avg_sq'] = state['cons_exp_avg_sq'] * beta2 + (grad * grad) * (1 - beta2)

                    steps = state['cons_step']
                    exp_avg = state['cons_exp_avg']
                    # exp_avg_sq = state['cons_exp_avg_sq']
                else:
                    # pdb.set_trace()
                    state['step'] += 1
                    state['exp_avg'] = state['exp_avg'] * beta1 + grad * (1 - beta1)

                    steps = state['step']
                    exp_avg = state['exp_avg']

                state['exp_avg_sq'] = state['exp_avg_sq'] * beta2 + (grad * grad) * (1 - beta2)
                exp_avg_sq = state['exp_avg_sq']

                bias_correction1 = 1 - beta1 ** steps
                bias_correction2 = 1 - beta2 ** (state['cons_step'] + state['step'])

                # Decay the first and second moment running average coefficient
                # if amsgrad:
                #     # Maintains the maximum of all 2nd moment running avg. till now
                #     state['max_exp_avg_sq'] = max(state['max_exp_avg_sq'], state['exp_avg_sq'])
                #     # Use the max. for normalizing running avg. of gradient
                #     denom = math.sqrt(state['max_exp_avg_sq'] / bias_correction2) + group['eps']
                # else:
                denom = math.sqrt(exp_avg_sq / bias_correction2) + group['eps']

                step_size = group['lr'] / bias_correction1

                # update the parameter
                state['alpha'] = max(state['alpha'] - step_size * exp_avg / denom, group['min_scale'])
                p.data.mul_(state['alpha'] * state['init_norm'] / curr_norm)

        # print(alphas)
        # print(grad_list)
        # print(max(grad_list), min(grad_list), max(alphas), min(alphas))
        # pdb.set_trace()
        return loss

    def reset_momentums(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                if len(state) == 0:
                    state['alpha'] = 1.
                    state['init_norm'] = p.norm().item()
                    state['step'] = 0
                    # Exponential moving average of gradient values for the weight norms
                    state['exp_avg'] = 0
                    # Exponential moving average of squared gradient values for the weight norms
                    state['exp_avg_sq'] = 0
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = 0
                else:
                    state['step'] = 0
                    # Exponential moving average of gradient values for the weight norms
                    state['exp_avg'] = 0
                    # Exponential moving average of squared gradient values for the weight norms
                    state['exp_avg_sq'] = 0
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = 0
