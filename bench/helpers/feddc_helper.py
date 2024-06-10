import numpy as np
import torch
from utils.config import Config

class FedDCHelper:
    def __init__(self, cfg: Config, model_size: int, num_clients: int):
        self.cfg = cfg
        self.num_clients = num_clients
        self.state_gradient_diffs = np.zeros((num_clients + 1, model_size), dtype=np.float32)
        self.parameter_drifts = np.zeros((num_clients, model_size), dtype=np.float32)

        self.delta_g_sum = np.zeros(model_size)

    def before_clnt_trn(self, client, device):
        client.cfg.hist_i = torch.tensor(self.parameter_drifts[int(client.id)], dtype=torch.float32, device=device)

        client.cfg.global_update_last = self.state_gradient_diffs[-1] / self.num_clients
        client.cfg.state_update_diff = torch.tensor(
            -self.state_gradient_diffs[int(client.id)] + client.cfg.global_update_last,
            dtype=torch.float32, device=device)

    def after_clnt_trn(self, client):
        curr_model_param = []
        for n, p in client.model.named_parameters():
            curr_model_param.append(p.data.view(-1).cpu().numpy())
        curr_model_param = np.concatenate(curr_model_param)
        delta_param_curr = curr_model_param - self.cfg.global_model_param.numpy()
        self.parameter_drifts[int(client.id)] += delta_param_curr

        state_g = self.state_gradient_diffs[int(client.id)] - self.cfg.global_update_last
        delta_g_cur = (state_g - self.state_gradient_diffs[int(client.id)]) / self.num_clients
        self.delta_g_sum += delta_g_cur
        self.state_gradient_diffs[int(client.id)] = state_g

    def update_and_get_glb_drifts(self):
        delta_g_cur = self.delta_g_sum / self.num_clients
        self.state_gradient_diffs[-1] += delta_g_cur

        return np.mean(self.parameter_drifts, axis=0)

