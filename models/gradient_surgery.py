import torch
import abc
import torch.nn as nn
from typing import List
from collections import OrderedDict


class StalenessAggregationHelper(abc.ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.gpu_id)

    @abc.abstractmethod
    def aggregate_grad(self,
                       accumulated_grad: OrderedDict,
                       staleness_grads: List[dict],
                       req_grad_params: list):
        pass


class SeqStalenessAggHelper(StalenessAggregationHelper):
    def aggregate_grad(self,
                       accumulated_grad: OrderedDict,
                       staleness_grads: List[dict],
                       req_grad_params: list):
        last_acc_grad_mom = None
        for i in range(self.cfg.max_staleness, 0, -1):
            cur_acc_grad_mom = OrderedDict()
            for grad in staleness_grads:
                if grad["staleness_round"] == i:
                    for name in req_grad_params:
                        if name not in cur_acc_grad_mom:
                            cur_acc_grad_mom[name] = grad["update"][name].clone()
                        else:
                            cur_acc_grad_mom[name] += grad["update"][name]
                    del grad["update"]
            if len(cur_acc_grad_mom) > 0:
                grad_list = []
                for name in req_grad_params:
                    grad_list.append(cur_acc_grad_mom[name].flatten())
                cur_acc_grad_mom = torch.cat(grad_list).to(self.device)
            else:
                cur_acc_grad_mom = None
            if last_acc_grad_mom is not None:
                if cur_acc_grad_mom is None:
                    cur_acc_grad_mom = last_acc_grad_mom
                else:
                    inner_prod = torch.dot(last_acc_grad_mom, cur_acc_grad_mom)
                    if inner_prod < 0:
                        last_acc_grad_mom = last_acc_grad_mom - inner_prod / (
                                cur_acc_grad_mom ** 2).sum() * cur_acc_grad_mom

                    cur_acc_grad_mom = last_acc_grad_mom + cur_acc_grad_mom

            last_acc_grad_mom = cur_acc_grad_mom

        if last_acc_grad_mom is not None:
            last_acc_grad_mom /= len(staleness_grads)
            accumulated_grad_vector = torch.cat([accumulated_grad[name].flatten() for name in req_grad_params])
            inner_prod = torch.dot(accumulated_grad_vector, last_acc_grad_mom)
            if inner_prod < 0:
                last_acc_grad_mom -= inner_prod / (accumulated_grad_vector ** 2).sum() * accumulated_grad_vector

            accumulated_grad_vector += last_acc_grad_mom
            idx = 0
            for name in req_grad_params:
                offset = accumulated_grad[name].numel()
                accumulated_grad[name] = accumulated_grad_vector[idx:idx + offset].clone().view(
                    accumulated_grad[name].shape)
                idx += offset

        return accumulated_grad


class SumStalenessAggHelper(StalenessAggregationHelper):
    def aggregate_grad(self,
                       accumulated_grad: OrderedDict,
                       staleness_grads: List[dict],
                       req_grad_params: list):
        stale_grad_vector = []
        acc_grad_vector = []
        for name in req_grad_params:
            tmp_v = torch.zeros_like(accumulated_grad[name])
            for stale_grad in staleness_grads:
                tmp_v += stale_grad["update"][name] * (1 / (1 + stale_grad["staleness_round"]) ** 0.5)
            if len(staleness_grads) > 0:
                tmp_v /= len(staleness_grads)
            stale_grad_vector.append(tmp_v.flatten())
            acc_grad_vector.append(accumulated_grad[name].flatten())

        stale_grad_vector = torch.cat(stale_grad_vector)
        acc_grad_vector = torch.cat(acc_grad_vector)

        inner_prod = torch.dot(stale_grad_vector, acc_grad_vector)
        if inner_prod < 0:
            stale_grad_vector = stale_grad_vector - inner_prod / (acc_grad_vector ** 2).sum() * acc_grad_vector

        acc_grad_vector = acc_grad_vector + stale_grad_vector
        idx = 0
        for name in req_grad_params:
            offset = accumulated_grad[name].numel()
            accumulated_grad[name] = acc_grad_vector[idx:idx + offset].clone().view(accumulated_grad[name].shape)
            idx += offset

        return accumulated_grad


class NaiveStalenessAggHelper(StalenessAggregationHelper):
    def aggregate_grad(self,
                       accumulated_grad: OrderedDict,
                       staleness_grads: List[dict],
                       req_grad_params: list):
        for name in req_grad_params:
            tmp_v = torch.zeros_like(accumulated_grad[name])
            for stale_grad in staleness_grads:
                tmp_v += stale_grad["update"][name] * (1 / (1 + stale_grad["staleness_round"]) ** 0.5)
            if len(staleness_grads) > 0:
                tmp_v /= len(staleness_grads)
            accumulated_grad[name] = accumulated_grad[name] + tmp_v

        return accumulated_grad
