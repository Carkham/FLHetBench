import math
import copy
import torch
import numpy as np
import torch.nn as nn
from utils.logger import Logger
from models.modeling import VisionTransformer, CONFIGS
from typing import Dict, List, Union

L = Logger()
logger = L.get_logger()


def compress_student_update_numpy(student_update: Dict[str, np.ndarray],
                                  energy: float):
    compressed_student_update = {}
    update_size = 0
    for name in student_update:
        if len(student_update[name].shape) > 1 and 'embeddings' not in name and 'head' not in name:
            u, sigma, v = np.linalg.svd(student_update[name], full_matrices=False)
            threshold = 0
            if np.sum(np.square(sigma)) == 0:
                compressed_student_update[name] = student_update[name]
                update_size += student_update[name].size
            else:
                for singular_value_num in range(len(sigma)):
                    if np.sum(np.square(sigma[:singular_value_num])) > energy * np.sum(np.square(sigma)):
                        threshold = singular_value_num
                        break
                u = u[:, :threshold]
                sigma = sigma[:threshold]
                v = v[:threshold, :]
                compressed_student_update[name] = [u, sigma, v]
                update_size += u.size + sigma.size + v.size
        elif 'embeddings' not in name:
            compressed_student_update[name] = student_update[name]
            update_size += student_update[name].size
    return compressed_student_update, update_size


def compress_student_update_torch(student_update: Dict[str, torch.Tensor],
                                  energy: float):
    compressed_student_update = {}
    update_size = 0
    with torch.no_grad():
        for name in student_update:
            if len(student_update[name].shape) > 1 and 'embeddings' not in name and 'head' not in name:
                u, sigma, v = torch.linalg.svd(student_update[name], full_matrices=False)
                threshold = 0
                if torch.sum(torch.square(sigma)) == 0:
                    compressed_student_update[name] = student_update[name]
                    update_size += student_update[name].numel()
                else:
                    singular_value_sum_factor = torch.cumsum(torch.square(sigma), dim=0) / torch.sum(
                        torch.square(sigma))
                    threshold = torch.nonzero(singular_value_sum_factor > energy)[0, 0] + 1  # :
                    threshold = threshold - 1 if threshold > len(sigma) else threshold
                    # for singular_value_num in range(len(sigma)):
                    #     if torch.sum(torch.square(sigma[:singular_value_num])) > energy * torch.sum(torch.square(sigma)):
                    #         threshold = singular_value_num
                    #         break
                    u = u[:, :threshold]
                    sigma = sigma[:threshold]
                    v = v[:threshold, :]
                    compressed_student_update[name] = [u, sigma, v]
                    update_size += u.numel() + sigma.numel() + v.numel()
            elif 'embeddings' not in name:
                compressed_student_update[name] = student_update[name]
                update_size += student_update[name].numel()
    return compressed_student_update, update_size


def compress_student_update(student_update: Dict[str, Union[torch.Tensor, np.ndarray]],
                            energy: float):
    for name in student_update:
        if isinstance(student_update[name], torch.Tensor):
            return compress_student_update_torch(student_update, energy)
        if isinstance(student_update[name], np.ndarray):
            return compress_student_update_numpy(student_update, energy)
    raise TypeError("compress update time error")


def aggregate_para_numpy(all_compressed_update: List[Dict[str, np.ndarray]],
                         energy: float):
    aggregated_para = {name: [] for name in all_compressed_update[0]}
    for name in all_compressed_update[0]:
        for i in range(len(all_compressed_update)):
            if len(all_compressed_update[i][name]) == 3:

                aggregated_para[name].append(
                    np.dot(np.dot(all_compressed_update[i][name][0], np.diag(all_compressed_update[i][name][1])),
                           all_compressed_update[i][name][2]))
            else:
                aggregated_para[name].append(all_compressed_update[i][name])
        aggregated_para[name] = np.mean(aggregated_para[name], axis=0)
    # for name in aggregated_para:
    #     if len(aggregated_para[name].shape) > 1:
    #         u, sigma, v = np.linalg.svd(aggregated_para[name], full_matrices=False)
    #         if np.sum(np.square(sigma)) == 0:
    #             continue
    #         else:
    #             threshold = 0
    #             for singular_value_num in range(len(sigma)):
    #                 if np.sum(np.square(sigma[:singular_value_num])) >= energy * np.sum(np.square(sigma)):
    #                     threshold = singular_value_num
    #                     break
    #             u = u[:, :threshold]
    #             sigma = sigma[:threshold]
    #             v = v[:threshold, :]
    #             aggregated_para[name] = [u, sigma, v]
    return aggregated_para


def aggregate_para_torch(all_compressed_update: List[Dict[str, torch.Tensor]],
                         energy: float):
    aggregated_para = {name: [] for name in all_compressed_update[0]}
    with torch.no_grad():
        for name in all_compressed_update[0]:
            for i in range(len(all_compressed_update)):
                if len(all_compressed_update[i][name]) == 3 and not isinstance(all_compressed_update[i][name],
                                                                               torch.Tensor):
                    aggregated_para[name].append(
                        all_compressed_update[i][name][0] @
                        torch.diag(all_compressed_update[i][name][1]) @
                        all_compressed_update[i][name][2]
                    )
                else:
                    aggregated_para[name].append(all_compressed_update[i][name])
            aggregated_para[name] = torch.mean(torch.stack(aggregated_para[name]), dim=0)
    # for name in aggregated_para:
    #     if len(aggregated_para[name].shape) > 1:
    #         u, sigma, v = np.linalg.svd(aggregated_para[name], full_matrices=False)
    #         if np.sum(np.square(sigma)) == 0:
    #             continue
    #         else:
    #             threshold = 0
    #             for singular_value_num in range(len(sigma)):
    #                 if np.sum(np.square(sigma[:singular_value_num])) >= energy * np.sum(np.square(sigma)):
    #                     threshold = singular_value_num
    #                     break
    #             u = u[:, :threshold]
    #             sigma = sigma[:threshold]
    #             v = v[:threshold, :]
    #             aggregated_para[name] = [u, sigma, v]
    return aggregated_para


def aggregate_para(all_compressed_update: List[Dict[str, Union[torch.Tensor, np.ndarray]]],
                   energy: float):
    for name in all_compressed_update[0]:
        if isinstance(all_compressed_update[0][name], torch.Tensor):
            return aggregate_para_torch(all_compressed_update, energy)
        if isinstance(all_compressed_update[0][name], np.ndarray):
            return aggregate_para_numpy(all_compressed_update, energy)
    raise TypeError("compress update time error")


class DistillWrapper(nn.Module):
    def __init__(self, cfg):
        super(DistillWrapper, self).__init__()
        self.teacher_model = VisionTransformer(CONFIGS["ViT-B_16"],
                                               224,
                                               zero_head=True,
                                               num_classes=2,
                                               task_num_classes=cfg.num_classes,
                                               vis=True)
        self.student_model = VisionTransformer(CONFIGS["ViT-B_16"],
                                               224,
                                               zero_head=True,
                                               num_classes=2,
                                               task_num_classes=cfg.num_classes,
                                               vis=True)
        if not cfg.num_classes == 2:
            self.teacher_model.head = nn.Linear(self.teacher_model.head.weight.shape[1], cfg.num_classes)
            self.student_model.head = nn.Linear(self.student_model.head.weight.shape[1], cfg.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion2 = nn.KLDivLoss()
        self.criterion3 = nn.MSELoss()
        self.distill_attn = False

    def forward(self, x, labels=None):
        loss_mse = 0
        if labels is not None:
            logits1, attn_weights1 = self.teacher_model(x)
            logits2, attn_weights2 = self.student_model(x)

            if self.distill_attn:
                for layer in range(len(attn_weights1) - 1):
                    layer_ratio = 12 // 12
                    loss_mse += self.criterion3(attn_weights2[-layer - 1], attn_weights1[-layer * layer_ratio - 1])

            outputs_S1 = torch.log_softmax(logits1, dim=1)
            outputs_S2 = torch.log_softmax(logits2, dim=1)
            outputs_T1 = torch.softmax(logits1, dim=1)
            outputs_T2 = torch.softmax(logits2, dim=1)

            loss1 = self.criterion(logits1, labels)
            loss2 = self.criterion(logits2, labels)
            loss3 = self.criterion2(outputs_S1, outputs_T2) / (loss1 + loss2)
            loss4 = self.criterion2(outputs_S2, outputs_T1) / (loss1 + loss2)

            return loss1 + loss3 + loss2 + loss4
        else:
            return self.student_model(x, labels)


class FedKDHelper:
    def __init__(self, cfg, inter_model, n_client: int):
        self.cfg = cfg
        self.comm_round = 0
        self.device = torch.device(self.cfg.gpu_id)
        self.student_model = inter_model.get_student_state(self.device)
        self.teacher_models = [copy.deepcopy(inter_model.get_teacher_state()) for _ in range(n_client)]

        self.all_compressed_student_update = []
        self.all_compressed_update_size = []
        self.compress_grad = False
        self.is_svd = True
        self.g = np.random.default_rng(42)

    def get_distill_model(self, clnt_idx):
        return self.teacher_models[clnt_idx], self.student_model

    def update_teacher_model(self, clnt_idx, teacher_state):
        # self.teacher_models[clnt_idx] = copy.deepcopy(teacher_state)
        for name, v in teacher_state.items():
            self.teacher_models[clnt_idx][name] = v.clone().detach()
        # self.teacher_models[clnt_idx] = teacher_state

    def compress_student_model(self,
                               student_state: Dict[str, torch.Tensor]):
        student_update = {}
        for name in student_state:
            student_update[name] = \
                (student_state[name].detach().to(self.device) - self.student_model[name].to(self.device)).to(
                    self.device)

        energy = self.cfg.tmin + ((1 + self.comm_round) / self.cfg.rounds) * (self.cfg.tmax - self.cfg.tmin)
        compressed_student_update, update_size = compress_student_update(student_update, energy)
        self.all_compressed_student_update.append(compressed_student_update)
        self.all_compressed_update_size.append(update_size)
        return update_size

    def fail_release(self):
        # remove the last update
        self.all_compressed_student_update = self.all_compressed_student_update[:-1]
        self.all_compressed_update_size = self.all_compressed_update_size[:-1]

    def update_student_model(self):
        logger.info(f"update client num: {len(self.all_compressed_student_update)}")
        if self.is_svd:
            energy = self.cfg.tmin + ((1 + self.comm_round) / self.cfg.rounds) * (self.cfg.tmax - self.cfg.tmin)
            aggregated_para = aggregate_para(self.all_compressed_student_update, energy)
        else:
            aggregated_para = {}
            for name in self.all_compressed_student_update[0]:
                aggregated_para[name] = torch.zeros_like(self.all_compressed_student_update[0][name])
                for update in self.all_compressed_student_update:
                    aggregated_para[name].add_(update[name])
                aggregated_para[name].div_(len(self.all_compressed_student_update))
        agg_sum = 0
        for name in aggregated_para:
            if isinstance(aggregated_para[name], np.ndarray):
                self.student_model[name].add_(torch.FloatTensor(aggregated_para[name]))
            else:
                agg_sum += torch.sum(aggregated_para[name].abs())
                # self.student_model[name].add_(aggregated_para[name].cpu())
                self.student_model[name].add_(aggregated_para[name])
        logger.info(f"agg_sum: {agg_sum}")
        update_size_mean = np.mean(self.all_compressed_update_size)
        self.all_compressed_update_size = []
        self.all_compressed_student_update = []
        return update_size_mean

    def release_student_model(self):
        self.all_compressed_update_size = []
        self.all_compressed_student_update = []
