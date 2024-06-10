import torch
import torch.nn as nn

from copy import deepcopy
from collections import OrderedDict

style = "state_dict"
correction_dtype = torch.float32


def create_params(model: nn.Module, num_clients):
    if style == "state_dict":
        weight = model.state_dict()
        clnt_params = [{} for _ in range(num_clients)]
        local_corrections = OrderedDict()
        for k, v in weight.items():
            local_corrections[k] = torch.zeros_like(v, dtype=correction_dtype)
        local_corrections = [deepcopy(local_corrections) for _ in range(num_clients)]
        return clnt_params, local_corrections
    else:
        vector = []
        for n, p in model.named_parameters():
            vector.append(p.clone().detach().reshape(-1))
        vector = torch.cat(vector)


def compute_param_offset(model: nn.Module):
    offset = 0
    ret = []
    for n, p in model.named_parameters():
        ret.append((n, (offset, offset + p.numel())))
        offset += p.numel()
    return ret


def preprocess_params(model: nn.Module, clnt_param, local_correction):
    if isinstance(clnt_param, dict) and isinstance(local_correction, dict):
        return clnt_param, local_correction
    else:
        raise NotImplementedError("Not implemented yet.")


def postprocess_params(last_server_model: dict, clnt_trained_model: dict, clnt_param, local_correction):
    if isinstance(clnt_param, dict) and isinstance(local_correction, dict):
        for k in local_correction:
            local_correction[k] += (clnt_trained_model[k] - last_server_model[k].to(clnt_trained_model[k].device)).to(correction_dtype).cpu()
            # clnt_param[k] = clnt_trained_model[k].cpu().detach()
    else:
        raise NotImplementedError("Not implemented yet.")


class FedDynHelper:
    def __init__(self, cfg, model: nn.Module, num_clients: int):
        self.clnt_params, self.local_dual_correction = create_params(model, num_clients)
        self.current_local_correction = None

    def preprocess(self, clnt_idx, model):
        """
        返回clnt_idx的参数与累计误差
        :param clnt_idx:
        :param model: 目标model1形状
        :return:
        """
        clnt_param, local_correction = preprocess_params(model, self.clnt_params[clnt_idx],
                                                         self.local_dual_correction[clnt_idx])
        self.current_local_correction = local_correction
        return clnt_param, local_correction

    def postprocess(self, clnt_idx, clnt_trained_model: dict, last_server_model: dict):
        """
        更新模型参数clnt_params以及local_dual_correction
        :param clnt_idx:
        :param clnt_trained_model: 既\theta_k^{t}的state_dict
        :param last_server_model: 既\theta^{t-1}的state_dict
        :return:
        """
        postprocess_params(last_server_model, clnt_trained_model, self.clnt_params[clnt_idx],
                           self.local_dual_correction[clnt_idx])
        self.current_local_correction = None

    def get_local_param_list(self, clnt_idxs):
        return [self.local_dual_correction[idx] for idx in clnt_idxs]
