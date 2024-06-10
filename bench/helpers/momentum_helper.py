import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Collection, List
from copy import deepcopy


def get_grads_(model, server_model):
    grads = []
    for key in server_model.state_dict().keys():
        if 'num_batches_tracked' not in key:
            grads.append(model.state_dict()[key].data.clone().detach().flatten() - server_model.state_dict()[
                key].data.clone().detach().flatten())
    return torch.cat(grads)


def set_grads_(model, server_model, new_grads):
    start = 0
    for key in server_model.state_dict().keys():
        if 'num_batches_tracked' not in key:
            dims = model.state_dict()[key].shape
            end = start + dims.numel()
            model.state_dict()[key].data.copy_(
                server_model.state_dict()[key].data.clone().detach() + new_grads[start:end].reshape(dims).clone())
            start = end
    return model


def pcgrad_hierarchy(sys_n_client, client_grads, grad_history=None):
    """ Projecting conflicting gradients"""
    # for client_grad in client_grads:
    #     del client_grad
    client_grads_ = torch.stack(client_grads)
    grads = []
    grad_len = grad_history['grad_len']
    start = 0
    for key in grad_len.keys():
        g_len = grad_len[key]
        end = start + g_len
        layer_grad_history = grad_history[key]
        if layer_grad_history is not None:
            pc_v = layer_grad_history.unsqueeze(0)
            client_grads_layer = client_grads_[:, start:end]
            while True:
                num = client_grads_layer.size(0)
                if num > 2:
                    inner_prod = torch.mul(client_grads_layer, pc_v).sum(1)
                    project = inner_prod / (pc_v ** 2).sum().sqrt()
                    _, ind = project.sort(descending=True)
                    pair_list = []
                    if num % 2 == 0:
                        for i in range(num // 2):
                            pair_list.append([ind[i], ind[num - i - 1]])
                    else:
                        for i in range(num // 2):
                            pair_list.append([ind[i], ind[num - i - 1]])
                        pair_list.append([ind[num // 2]])
                    client_grads_new = []
                    for pair in pair_list:
                        if len(pair) > 1:
                            grad_0 = client_grads_layer[pair[0]]
                            grad_1 = client_grads_layer[pair[1]]
                            inner_prod = torch.dot(grad_0, grad_1)
                            if inner_prod < 0:
                                # Sustract the conflicting component
                                grad_pc_0 = grad_0 - inner_prod / (grad_1 ** 2).sum() * grad_1
                                grad_pc_1 = grad_1 - inner_prod / (grad_0 ** 2).sum() * grad_0
                            else:
                                grad_pc_0 = grad_0
                                grad_pc_1 = grad_1
                            grad_pc_0_1 = grad_pc_0 + grad_pc_1
                            client_grads_new.append(grad_pc_0_1)
                        else:
                            grad_single = client_grads_layer[pair[0]]
                            client_grads_new.append(grad_single)
                    client_grads_layer = torch.stack(client_grads_new)
                elif num == 2:
                    grad_pc_0 = client_grads_layer[0]
                    grad_pc_1 = client_grads_layer[1]
                    inner_prod = torch.dot(grad_pc_0, grad_pc_1)
                    if inner_prod < 0:
                        # Sustract the conflicting component
                        grad_pc_0 = grad_pc_0 - inner_prod / (grad_pc_1 ** 2).sum() * grad_pc_1
                        grad_pc_1 = grad_pc_1 - inner_prod / (grad_pc_0 ** 2).sum() * grad_pc_0

                    grad_pc_0_1 = grad_pc_0 + grad_pc_1
                    grad_new = grad_pc_0_1 / sys_n_client
                    break
                else:
                    assert False
            gamma = 0.99
            grad_history[key] = gamma * grad_history[key] + (1 - gamma) * grad_new
            grads.append(grad_new)
        else:
            grad_new = client_grads_[:, start:end].mean(0)
            grad_history[key] = grad_new
            grads.append(grad_new)
        start = end
    grad_new = torch.cat(grads)

    return grad_new, grad_history


class MomentumHelper:
    def __init__(self,
                 cfg,
                 model: nn.Module
                 ):
        self.momentum = cfg.virtual_momentum
        self.velocity = OrderedDict()
        self.grad_history = OrderedDict()
        self.grad_history['grad_len'] = OrderedDict()
        self.pcgrad_momentum = cfg.pcgrad_m
        self.device = torch.device(cfg.gpu_id)
        for k, p in model.named_parameters():
            if p.requires_grad:
                self.velocity[k] = torch.zeros_like(p.data).to(self.device)
                self.grad_history[k] = None
                self.grad_history['grad_len'][k] = p.data.numel()

    def update_with_momentum(self,
                             grad: OrderedDict,
                             velocity: OrderedDict,
                             req_grad_params: Collection):
        for k, p in grad.items():
            if k in req_grad_params:
                torch.add(p, velocity[k], alpha=self.momentum, out=p)
                velocity[k] = deepcopy(p)
        return grad, velocity

    def zeros_like_grad(self):
        ret = OrderedDict()
        for k, p in self.velocity.items():
            ret[k] = torch.zeros_like(p).to(self.device)
        return ret

    def update_with_pcgrad(self,
                           grads: List[OrderedDict],
                           req_grad_params: Collection):
        if len(grads) <= 1:
            return grads[0]
        client_grads = []
        param_index = OrderedDict()
        for grad in grads:
            grad_list = []
            start = 0
            for k, p in grad.items():
                if k in req_grad_params:
                    grad_list.append(grad[k].flatten())
                    if k not in param_index:
                        param_index[k] = (start, start + grad[k].numel(), grad[k].shape)
                        start = start + grad[k].numel()

            client_grads.append(torch.cat(grad_list))

        new_grad, grad_history = pcgrad_hierarchy(len(grads), client_grads, self.grad_history)

        grad = OrderedDict()
        for k, index in param_index.items():
            grad[k] = new_grad[index[0]:index[1]].reshape(index[2])

        self.grad_history = grad_history
        return grad
