import math
from collections import OrderedDict

import torch
from typing import List, Collection


class ContrHelper:
    def __init__(self, cfg, num_clients: int):
        self.contr_factor = cfg.contr_factor
        self.contr_t = cfg.contr_t
        self.client_step = [0] * num_clients
        self.client_last_update_round = [None] * num_clients
        self.contr_threshold = 0.01

    def contr_aggregation(self,
                          updates: List[dict],
                          req_grad_params: Collection,
                          cur_round: int,
                          device: torch.device):
        accumulated_grad = OrderedDict()
        cur_client_steps = []
        cur_client_last_update_round = []
        for update in updates:
            cid = update[0]
            state = update[1]
            client_contr_ratio = max(self.contr_threshold, math.cos(min(self.client_step[cid], self.contr_t) / self.contr_t * math.pi / 2))
            cur_client_steps.append((cid, self.client_step[cid], self.client_last_update_round[cid], client_contr_ratio))
            if self.client_last_update_round[cid] is None:
                self.client_step[cid] += 1
            elif cur_round - self.client_last_update_round[cid] > self.contr_t * 2:
                self.client_step[cid] = 0
            else:
                self.client_step[cid] += 1
            self.client_last_update_round[cid] = cur_round

            for name in req_grad_params:
                if name not in accumulated_grad:
                    accumulated_grad[name] = self.contr_factor * client_contr_ratio * state[name].to(device)
                else:
                    accumulated_grad[name] += self.contr_factor * client_contr_ratio * state[name].to(device)

        return accumulated_grad, cur_client_steps, cur_client_last_update_round
