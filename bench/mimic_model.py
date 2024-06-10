import copy
import gc

import torch
import torch.nn as nn
import numpy as np
import client

from helpers.fedkd_helper import DistillWrapper
from models.biasprompt import BiasPromptViT, BiasPromptResNet

from models.modeling import CONFIGS, VisionTransformer, ResNet
from utils.util import valid
from torch.nn import Linear
from collections import OrderedDict
from utils.logger import Logger

L = Logger()
logger = L.get_logger()
criterion = nn.CrossEntropyLoss()

from utils.config import Config


class MimicModel:
    def __init__(self, cfg: Config):
        self.model = build_backbone(cfg)
        self.size = self._size()
        print(
            "update size: {}\t rate: {}".format(self.size, self.size / sum(p.numel() for p in self.model.parameters())))
        if cfg.feddyn:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=cfg.client_lr,
                                             weight_decay=cfg.feddyn_alpha)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=cfg.client_lr)
        self.cfg = cfg
        self.grad = None
        self.masks = None
        self.scaler = None

    def setup_partial_grad(self, ratio):
        freeze_param_size = self.size * (1 - ratio)
        param_size = 0
        for p in self.model.parameters():
            param_size += p.numel()
            if param_size < freeze_param_size:
                p.requires_grad = False
            else:
                p.requires_grad = True
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=self.cfg.client_lr)

    def get_weights(self):
        return self.model.state_dict()

    def get_grad(self):
        assert self.grad is not None
        ret_grad = self.grad
        self.grad = None
        return ret_grad

    def load_weights(self, w):
        self.model.load_state_dict(w)

    def train(self, dataloader, num_epochs):
        self.model = self.model.to(torch.device(self.cfg.gpu_id))
        self.model.train()
        self.trainloss = 0
        self.optimizer.zero_grad()
        server_model = copy.deepcopy(self.model.state_dict())

        for epoch in range(num_epochs):
            for idx, j in enumerate(dataloader):
                batch = tuple(t for t in j)
                x, y = batch
                x = x.to(torch.device(self.cfg.gpu_id))
                y = y.to(torch.device(self.cfg.gpu_id))
                if self.cfg.fedprox:
                    proximal = 0.0
                    proxy_weights = server_model  # just add here to show
                    for name, p in self.model.named_parameters():
                        proximal += (p - proxy_weights[name].to(torch.device(self.cfg.gpu_id))).norm(2)

                    proximal = proximal * (self.cfg.fedprox_mu / 2)
                    loss = self.model(x, y) + proximal
                elif self.cfg.feddyn:

                    loss = self.model(x, y)
                    local_param_list = []
                    local_grad_vector = []
                    cloud_param_list = []
                    for name, param in self.model.named_parameters():
                        local_param_list.append(param.view(-1))
                        local_grad_vector.append(
                            self.cfg.feddyn_helper.current_local_correction[name].view(-1).to(param.device))
                        cloud_param_list.append(server_model[name].view(-1).to(param.device))
                    local_param_list = torch.cat(local_param_list, 0)
                    local_grad_vector = torch.cat(local_grad_vector, 0).to(local_param_list.device)
                    cloud_param_list = torch.cat(cloud_param_list, 0).to(local_param_list.device)

                    loss_algo = self.cfg.feddyn_alpha * torch.sum(
                        local_param_list * (-cloud_param_list + local_grad_vector))
                    loss = loss + loss_algo
                elif self.cfg.fedkd:
                    loss = self.model(x, y)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=False)

                    loss = self.model(x, y)
                else:

                    loss = self.model(x, y)

                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
                self.trainloss += loss.item()
                self.optimizer.zero_grad(set_to_none=False)
        if self.cfg.fetchsgd:
            assert server_model is not None
            grad_vec = []
            with torch.no_grad():
                for k, p in self.model.named_parameters():
                    if p.requires_grad:
                        if p.grad is None:
                            grad_vec.append(torch.zeros_like(p.data.view(-1)))
                        else:
                            grad_vec.append(torch.sub(p.data.view(-1), server_model[k].view(-1).to(p.device)))
                # concat into a single vector
                grad_vec = torch.cat(grad_vec)
                self.grad = grad_vec.to(torch.device(self.cfg.gpu_id))
        elif self.cfg.virtual_momentum > 0:
            assert server_model is not None
            grad_vec = OrderedDict()
            with torch.no_grad():
                for k, p in self.model.named_parameters():
                    if p.requires_grad:
                        if p.grad is None:
                            # grad_vec[k] = torch.zeros_like(self.model.state_dict()[k])
                            grad_vec[k] = torch.zeros_like(p.data)
                        else:
                            # grad_vec[k] = torch.sub(self.model.state_dict()[k], server_model[k].to(p.device))
                            grad_vec[k] = torch.sub(p.data, server_model[k].to(p.device))
                self.grad = grad_vec
        client.updatetimer += 1

    def eval(self, cfg, dataloader):
        gc.collect()
        torch.cuda.empty_cache()
        self.model = self.model.to(torch.device(self.cfg.gpu_id))
        acc, totalloss = valid(cfg, self.model, dataloader, TestFlag=True)
        print("Eval loss: " + str(totalloss) + " acc: " + str(acc))
        return totalloss, acc

    def get_comp(self, *argv):
        return -1

    def _size(self):
        float_num = 0
        binary_num = 0
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                if 'mask' in k:
                    binary_num += v.numel()
                else:
                    float_num += v.numel()
        return float_num + binary_num / 4


class DistMimicModel(MimicModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.model = DistillWrapper(cfg)
        self.model.teacher_model.load_from(np.load("checkpoint/ViT-B_16.npz"))
        self.model.student_model.load_from(np.load("checkpoint/ViT-B_16.npz"))
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=cfg.client_lr,
                                         momentum=0.)
        self.size = sum(p.numel() for p in self.model.student_model.parameters())

    def load_tea_stu(self, student_w, teacher_w):
        self.model.teacher_model.load_state_dict(teacher_w)
        self.model.student_model.load_state_dict(student_w)

    def load_stu(self, student_w):
        self.model.student_model.load_state_dict(student_w)

    def get_teacher_state(self, device=None):
        if not device:
            return self.model.teacher_model.cpu().state_dict()
        else:
            return self.model.teacher_model.to(device).state_dict()

    def get_student_state(self, device=None):
        if not device:
            return self.model.student_model.cpu().state_dict()
        else:
            return self.model.student_model.to(device).state_dict()


def build_vit_sup_models(cfg, model_root="checkpoint/ViT-B_16.npz",
                         load_pretrain=True) -> nn.Module:
    if cfg.biasprompt:
        model = BiasPromptViT(CONFIGS["BiasPromptViT"],
                              cfg,
                              224,
                              zero_head=True,
                              num_classes=2,
                              task_num_classes=cfg.num_classes)
        if not cfg.num_classes == 2:
            model.head = Linear(model.head.weight.shape[1], cfg.num_classes)
    else:
        model = VisionTransformer(CONFIGS["ViT-B_16"],
                                  224,
                                  zero_head=True,
                                  num_classes=2,
                                  task_num_classes=cfg.num_classes)
        if not cfg.num_classes == 2:
            model.head = Linear(model.head.weight.shape[1], cfg.num_classes)

    if load_pretrain:
        model.load_from(np.load(model_root))

    return model


def build_resnet_sup_models(cfg):
    if cfg.biasprompt:
        model = BiasPromptResNet(
            CONFIGS["Prompt-ResNet50"],
            cfg,
            img_size=224,
            zero_head=True,
            num_classes=2,
            task_num_classes=cfg.num_classes
        )
        if not cfg.num_classes == 2:
            model.head = Linear(model.head.weight.shape[1], cfg.num_classes)
    else:
        model = ResNet(cfg,
                       224,
                       zero_head=True,
                       num_classes=2,
                       task_num_classes=cfg.num_classes)
        if not cfg.num_classes == 2:
            model.head = Linear(model.head.weight.shape[1], cfg.num_classes)

    return model


def build_backbone(cfg):
    if cfg.model == "vit":
        model = build_vit_sup_models(cfg)
    elif cfg.model == "resnet":
        model = build_resnet_sup_models(cfg)
        logger.info("--------------------Using ResNet ------------------------")
    else:
        raise NotImplementedError()

    if cfg.biasprompt:
        if isinstance(model, BiasPromptViT):
            for k, p in model.named_parameters():
                if 'head' not in k and 'pooler' not in k and "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False
        elif isinstance(model, BiasPromptResNet):
            for k, p in model.frozen_layers.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

    return model
