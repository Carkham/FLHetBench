import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import Collection
from copy import deepcopy
from scipy.stats import bernoulli
from typing import Tuple


class FedPMHelper:
    def __init__(self, req_grad_params):
        self.lambda_init = 1
        self.epsilon = 0.01
        self.req_grad_params = req_grad_params

    def compute_n_mask_samples(self, n_roud):
        return 1

    def _with_torch(self,
                    param_dict: dict,
                    k: str,
                    v: torch.Tensor):
        if 'mask' in k:
            theta = torch.sigmoid(v)
            updates_s = torch.bernoulli(theta)
            updates_s = torch.clip(updates_s, self.epsilon, 1 - self.epsilon)

            if param_dict.get(k) is None:
                param_dict[k] = updates_s.cpu()
            else:
                param_dict[k] += updates_s.cpu()
        elif k in self.req_grad_params:
            param_dict[k] = v.cpu()

    def _with_scipy(self,
                    param_dict: dict,
                    k: str,
                    v: torch.Tensor):
        if 'mask' in k:
            theta = torch.sigmoid(v).cpu().numpy()
            updates_s = bernoulli.rvs(theta)
            updates_s = np.where(updates_s == 0, self.epsilon, updates_s)
            updates_s = np.where(updates_s == 1, 1 - self.epsilon, updates_s)

            if param_dict.get(k) is None:
                param_dict[k] = torch.tensor(updates_s)
            else:
                param_dict[k] += torch.tensor(updates_s)
        else:
            param_dict[k] = v.cpu()

    def upload_mask(self,
                    updates: dict,
                    n_samples: int) -> dict:
        param_dict = dict()
        with torch.no_grad():
            for _ in range(n_samples):
                for k, v in updates.items():
                    self._with_torch(param_dict, k, v)
                    # self._with_scipy(param_dict, k, v)
        return param_dict

    def reset_prior(self, model) -> Tuple[dict, dict]:
        alphas = dict()
        betas = dict()
        for k, val in model.named_parameters():
            alphas[k] = torch.ones_like(val, device='cpu') * self.lambda_init
            betas[k] = torch.ones_like(val, device='cpu') * self.lambda_init
        return alphas, betas
