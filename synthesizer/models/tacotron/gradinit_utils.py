import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from synthesizer.models.tacotron.gradinit_optimizers import RescaleAdam
from synthesizer.models.tacotron.models.modules import Scale, Bias


def get_ordered_params(net):
    param_list = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            param_list.append(m.weight)
            if m.bias is not None:
                param_list.append(m.bias)
        elif isinstance(m, Scale):
            param_list.append(m.weight)
        elif isinstance(m, Bias):
            param_list.append(m.bias)

    return param_list


def set_param(module, name, alg, eta, grad):
    weight = getattr(module, name)
    # remove this parameter from parameter list
    del module._parameters[name]

    # compute the update steps according to the optimizers
    if alg.lower() == 'sgd':
        gstep = eta * grad
    elif alg.lower() == 'adam':
        gstep = eta * grad.sign()
    else:
        raise RuntimeError("Optimization algorithm {} not defined!".format(alg))

    # add the updated parameter as the new parameter
    module.register_parameter(name + '_prev', weight)

    # recompute weight before every forward()
    updated_weight = weight - gstep.data
    setattr(module, name, updated_weight)


def take_opt_step(net, grad_list, alg='sgd', eta=0.1):
    """Take the initial step of the chosen optimizer.
    """
    assert alg.lower() in ['adam', 'sgd']

    idx = 0
    for n, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            grad = grad_list[idx]
            set_param(m, 'weight', alg, eta, grad)
            idx += 1

            if m.bias is not None:
                grad = grad_list[idx]
                set_param(m, 'bias', alg, eta, grad)
                idx += 1
        elif isinstance(m, Scale):
            grad = grad_list[idx]
            set_param(m, 'weight', alg, eta, grad)
            idx += 1
        elif isinstance(m, Bias):
            grad = grad_list[idx]
            set_param(m, 'bias', alg, eta, grad)
            idx += 1


def recover_params(net):
    """Reset the weights to the original values without the gradient step
    """

    def recover_param_(module, name):
        delattr(module, name)
        setattr(module, name, getattr(module, name + '_prev'))
        del module._parameters[name + '_prev']

    for n, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            recover_param_(m, 'weight')
            if m.bias is not None:
                recover_param_(m, 'bias')
        elif isinstance(m, Scale):
            recover_param_(m, 'weight')
        elif isinstance(m, Bias):
            recover_param_(m, 'bias')


def set_bn_modes(net):
    """Switch the BN layers into training mode, but does not track running stats.
    """
    for n, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
            m.track_running_stats = False


def recover_bn_modes(net):
    for n, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True


def get_scale_stats(model, optimizer):
    stat_dict = {}
    # all_s_list = [p.norm().item() for n, p in model.named_parameters() if 'bias' not in n]
    all_s_list = []
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            all_s_list.append(optimizer.state[p]['alpha'])
    stat_dict['s_max'] = max(all_s_list)
    stat_dict['s_min'] = min(all_s_list)
    stat_dict['s_mean'] = np.mean(all_s_list)
    all_s_list = []
    for n, p in model.named_parameters():
        if 'bias' not in n:
            all_s_list.append(optimizer.state[p]['alpha'])
    stat_dict['s_weight_max'] = max(all_s_list)
    stat_dict['s_weight_min'] = min(all_s_list)
    stat_dict['s_weight_mean'] = np.mean(all_s_list)

    return stat_dict


def get_batch(data_iter, data_loader):
    try:
        texts, mels, embeds, idx = next(data_iter)
    except:
        data_iter = iter(data_loader)
        texts, mels, embeds, idx = next(data_iter)
    return data_iter, texts, mels, embeds, idx


def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)


def pad2d(x, max_len, pad_value=0):
    return torch.tensor(np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode="constant", constant_values=pad_value))


def pad3d(x_arg, x_dest):
    x = torch.zeros_like(x_dest)
    x[:, :, :x_arg.shape[-1]] = x_arg[:, :]
    return x


def gradinit(net, dataloader, dataset, args):
    if args.gradinit_resume:
        print("Resuming GradInit model from {}".format(args.gradinit_resume))
        sdict = torch.load(args.gradinit_resume)
        net.load_state_dict(sdict)
        return

    # if isinstance(net, torch.nn.DataParallel):
    #     net_top = net.module
    # else:
    #     net_top = net
    device = torch.device(0)
    bias_params = [p for n, p in net.named_parameters() if 'bias' in n]
    weight_params = [p for n, p in net.named_parameters() if 'weight' in n]

    optimizer = RescaleAdam([{'params': weight_params, 'min_scale': args.gradinit_min_scale, 'lr': args.gradinit_lr},
                             {'params': bias_params, 'min_scale': 0, 'lr': args.gradinit_lr}],
                            grad_clip=args.gradinit_grad_clip)

    net.eval()  # This further shuts down dropout, if any.

    set_bn_modes(net)  # Should be called after net.eval()

    total_loss, total_l0, total_l1, total_residual, total_gnorm = 0, 0, 0, 0, 0
    total_sums, total_sums_gnorm = 0, 0
    cs_count = 0
    total_iters = 0
    obj_loss, updated_loss, residual = -1, -1, -1
    data_iter = iter(dataloader)
    # get all the parameters by order
    params_list = get_ordered_params(net)
    net.train()
    while True:
        eta = args.gradinit_eta

        # continue
        # get the first half of the minibatch
        data_iter, texts0, mels0, embeds0, idx0 = get_batch(data_iter, dataloader)

        # Get the second half of the data.
        data_iter, texts1, mels1, embeds1, idx1 = get_batch(data_iter, dataloader)

        init_inputs = torch.cat([texts0, pad2d(texts1, texts0.shape[-1])] if texts0.shape[-1] > texts1.shape[-1]
                                else [texts1, pad2d(texts0, texts1.shape[-1])]).to(device)
        init_mels = torch.cat([mels0, pad3d(mels1, mels0)] if mels0.shape[-1] > mels1.shape[-1]
                              else [mels1, pad3d(mels0, mels1)]).to(device)
        init_embeds = torch.cat([embeds0, embeds1]).to(device)
        net = net.to(device)
        init_idx = idx0 + idx1

        stop = torch.ones(init_mels.shape[0], init_mels.shape[2], device=device)
        for j, k in enumerate(init_idx):
            stop[j, :int(dataset.metadata[k][4]) - 1] = 0

        # compute the gradient and take one step
        # outputs = net(init_inputs)
        m1_hat, m2_hat, attention, stop_pred = net(init_inputs, init_mels, init_embeds)

        m1_loss = F.mse_loss(m1_hat, init_mels) + F.l1_loss(m1_hat, init_mels)
        m2_loss = F.mse_loss(m2_hat, init_mels)
        stop_loss = F.binary_cross_entropy(stop_pred, stop)  # ?

        init_loss = m1_loss + m2_loss + stop_loss

        all_grads = torch.autograd.grad(init_loss, params_list, create_graph=True)

        # Compute the loss w.r.t. the optimizer
        if args.gradinit_alg.lower() == 'adam':
            # grad-update inner product
            gnorm = sum([g.abs().sum() for g in all_grads])
            loss_grads = all_grads
        else:
            gnorm_sq = sum([g.square().sum() for g in all_grads])
            gnorm = gnorm_sq.sqrt()
            if args.gradinit_normalize_grad:
                loss_grads = [g / gnorm for g in all_grads]
            else:
                loss_grads = all_grads

        total_gnorm += gnorm.item()
        total_sums_gnorm += 1
        if gnorm.item() > args.gradinit_gamma:
            # project back into the gradient norm constraint
            optimizer.zero_grad()
            gnorm.backward()
            optimizer.step(is_constraint=True)

            cs_count += 1
        else:
            # take one optimization step
            take_opt_step(net, loss_grads, alg=args.gradinit_alg, eta=eta)

            total_l0 += init_loss.item()

            data_iter, texts2, mels2, embeds2, idx2 = get_batch(data_iter, dataloader)

            updated_inputs = torch.cat([texts0, pad2d(texts2, texts0.shape[-1])] if texts0.shape[-1] > texts2.shape[-1]
                                       else [texts2, pad2d(texts0, texts2.shape[-1])]).to(device)
            updated_mels = torch.cat([mels0, pad3d(mels2, mels0)] if mels0.shape[-1] > mels2.shape[-1]
                                     else [mels2, pad3d(mels0, mels2)]).to(device)
            updated_embeds = torch.cat([embeds0, embeds2]).to(device)
            updated_idx = idx0 + idx2

            stop = torch.ones(updated_mels.shape[0], updated_mels.shape[2], device=device)
            for j, k in enumerate(updated_idx):
                stop[j, :int(dataset.metadata[k][4]) - 1] = 0

            # compute loss using the updated network
            # net_top.opt_mode(True)
            m1_hat, m2_hat, attention, stop_pred = net(updated_inputs, updated_mels, updated_embeds)
            # net_top.opt_mode(False)
            m1_loss = F.mse_loss(m1_hat, updated_mels) + \
                      F.l1_loss(m1_hat, updated_mels)
            m2_loss = F.mse_loss(m2_hat, updated_mels)
            stop_loss = F.binary_cross_entropy(stop_pred, stop)  # ?

            updated_loss = m1_loss + \
                           m2_loss + \
                           stop_loss

            # If eta is larger, we should expect obj_loss to be even smaller.
            obj_loss = updated_loss / eta

            recover_params(net)
            optimizer.zero_grad()
            obj_loss.backward()
            optimizer.step(is_constraint=False)
            total_l1 += updated_loss.item()

            total_loss += obj_loss.item()
            total_sums += 1

        total_iters += 1
        if (total_sums_gnorm > 0 and total_sums_gnorm % 10 == 0) or total_iters == args.gradinit_iters:
            stat_dict = get_scale_stats(net, optimizer)
            print_str = "Iter {}, obj iters {}, eta {}, constraint count {} loss: {} ({}), init loss: " \
                        "{} ({}), update loss {} ({}), " \
                        "total gnorm: {} ({})\t".format(total_sums_gnorm, total_sums, eta, cs_count,
                                                        round(float(obj_loss), 3),
                                                        round(total_loss / total_sums if total_sums > 0 else -1, 3),
                                                        round(float(init_loss), 3),
                                                        round(total_l0 / total_sums if total_sums > 0 else -1, 3),
                                                        round(float(updated_loss), 3),
                                                        round(total_l1 / total_sums if total_sums > 0 else -1, 3),
                                                        float(gnorm), total_gnorm / total_sums_gnorm)

            for key, val in stat_dict.items():
                print_str += "{}: {:.2e}\t".format(key, val)
            print(print_str)

        if total_iters == args.gradinit_iters:
            break

    recover_bn_modes(net)
    return net


def gradient_quotient(loss, params, eps=1e-5):
    grad = torch.autograd.grad(loss, params, create_graph=True)
    prod = torch.autograd.grad(sum([(g ** 2).sum() / 2 for g in grad]), params,
                               create_graph=True)
    out = sum([((g - p) / (g + eps * (2 * (g >= 0).float() - 1).detach())
                - 1).abs().sum() for g, p in zip(grad, prod)])

    gnorm = sum([(g ** 2).sum().item() for g in grad])
    return out / sum([p.data.numel() for p in params]), gnorm
