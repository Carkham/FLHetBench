from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from functools import reduce
from operator import add

from torchvision import transforms

from .modeling import VisionTransformer, ResNet


class BiasPromptViT(VisionTransformer):
    def __init__(self, config, cfg, *args, **kwargs):
        super(BiasPromptViT, self).__init__(config, *args, **kwargs)
        self.embeddings = self.transformer.embeddings

        self.prompt_length = cfg.prompt_length

        self.use_prefix_tune = cfg.use_prefix_tune
        self.positions_idx = reduce(add, [list(range(s - 1, e)) for s, e in cfg.positions])
        if not self.use_prefix_tune:
            self.prompt_embeddings = nn.Parameter(
                torch.zeros(len(self.positions_idx), self.prompt_length, config.hidden_size))
        else:
            self.prompt_embeddings = nn.Parameter(
                torch.zeros(len(self.positions_idx), 2, self.prompt_length, config.transformer.num_heads,
                            config.hidden_size // config.transformer.num_heads))
        nn.init.xavier_uniform_(self.prompt_embeddings.data, gain=1)

    def get_prompt(self, layer_idx, batch_size, device):
        if not self.use_prefix_tune:
            return self.prompt_embeddings[self.positions_idx.index(layer_idx)].expand(batch_size, -1, -1)
        else:
            return self.prompt_embeddings[self.positions_idx.index(layer_idx)].expand(batch_size, -1, -1, -1, -1)

    def forward_layer_prompt(self, embedding_output):
        batch_size = embedding_output.shape[0]

        attn_weights = []
        hidden_states = embedding_output
        num_layer = len(self.transformer.encoder.layer)
        for layer_idx in range(num_layer):
            if layer_idx in self.positions_idx:
                prompts = self.get_prompt(layer_idx, batch_size, embedding_output.get_device())
                if not self.use_prefix_tune:
                    if self.positions_idx.index(layer_idx) == 0:  # first
                        hidden_states = torch.cat((
                            hidden_states[:, :1, :],
                            prompts,
                            hidden_states[:, 1:, :]
                        ), dim=1)
                    else:
                        hidden_states = torch.cat((
                            hidden_states[:, :1, :],
                            prompts,
                            hidden_states[:, 1 + prompts.shape[1]:, :]
                        ), dim=1)
                    prompts = None
            else:
                prompts = None
            hidden_states, weights = self.transformer.encoder.layer[layer_idx](hidden_states, prompts)
            if self.transformer.encoder.vis:
                attn_weights.append(weights)
        encoded = self.transformer.encoder.encoder_norm(hidden_states)

        return encoded, attn_weights

    def forward(self, x, labels=None):
        embedding_output = self.embeddings(x)
        # inputs_embeds = torch.cat((prompts, embedding_output), dim=1)
        x, attn_weights = self.forward_layer_prompt(embedding_output)

        logits = self.head(x[:, 0])

        if labels is not None:
            if self.num_classes == 1:
                # loss_fct = CrossEntropyLoss()
                loss_fct = torch.nn.L1Loss()
                loss = loss_fct(logits.view(-1, self.num_classes),
                                labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes),
                                labels.view(-1))

            return loss
        else:
            return logits, attn_weights


class BiasPromptResNet(ResNet):
    def __init__(self, config, cfg, *args, **kwargs):
        super(BiasPromptResNet, self).__init__(cfg, *args, **kwargs)
        self.cfg = cfg
        self.prompt_length = cfg.prompt_length
        self.prompt_config = config

        self.model = self.setup_prompt(config, self.model)
        self.setup_grad(self.model)

    def setup_grad(self, model):
        transfer_type = self.cfg.transfer_type
        # split enc into 3 parts:
        #           prompt_layers  frozen_layers  tuned_layers
        # partial-1  identity       -layer3       layer4
        # partial-2: identity       -layer2      "layer4" "layer3"
        # partial-3: identity       -layer1      "layer4" "layer3" "layer2"
        # linear     identity        all          identity
        # end2end    identity       identity      all

        # prompt-below  conv1        all but conv1
        # prompt-pad   identity        all

        if (
                transfer_type == "prompt" or transfer_type == "BiasPrompt") and self.prompt_config.LOCATION == "below":  # noqa
            self.prompt_layers = nn.Sequential(OrderedDict([
                ("conv1", model.conv1),
                ("bn1", model.bn1),
                ("relu", model.relu),
                ("maxpool", model.maxpool),
            ]))
            self.frozen_layers = nn.Sequential(OrderedDict([
                ("layer1", model.layer1),
                ("layer2", model.layer2),
                ("layer3", model.layer3),
                ("layer4", model.layer4),
                ("avgpool", model.avgpool),
            ]))
            self.tuned_layers = nn.Identity()
        else:
            # partial, linear, end2end, prompt-pad
            self.prompt_layers = nn.Identity()

            if transfer_type == "partial-0":
                # last conv block
                self.frozen_layers = nn.Sequential(OrderedDict([
                    ("conv1", model.conv1),
                    ("bn1", model.bn1),
                    ("relu", model.relu),
                    ("maxpool", model.maxpool),
                    ("layer1", model.layer1),
                    ("layer2", model.layer2),
                    ("layer3", model.layer3),
                    ("layer4", model.layer4[:-1]),
                ]))
                self.tuned_layers = nn.Sequential(OrderedDict([
                    ("layer4", model.layer4[-1]),
                    ("avgpool", model.avgpool),
                ]))
            elif transfer_type == "partial-1":
                # tune last layer
                self.frozen_layers = nn.Sequential(OrderedDict([
                    ("conv1", model.conv1),
                    ("bn1", model.bn1),
                    ("relu", model.relu),
                    ("maxpool", model.maxpool),
                    ("layer1", model.layer1),
                    ("layer2", model.layer2),
                    ("layer3", model.layer3),
                ]))
                self.tuned_layers = nn.Sequential(OrderedDict([
                    ("layer4", model.layer4),
                    ("avgpool", model.avgpool),
                ]))

            elif transfer_type == "partial-2":
                self.frozen_layers = nn.Sequential(OrderedDict([
                    ("conv1", model.conv1),
                    ("bn1", model.bn1),
                    ("relu", model.relu),
                    ("maxpool", model.maxpool),
                    ("layer1", model.layer1),
                    ("layer2", model.layer2),
                ]))
                self.tuned_layers = nn.Sequential(OrderedDict([
                    ("layer3", model.layer3),
                    ("layer4", model.layer4),
                    ("avgpool", model.avgpool),
                ]))

            elif transfer_type == "partial-3":
                self.frozen_layers = nn.Sequential(OrderedDict([
                    ("conv1", model.conv1),
                    ("bn1", model.bn1),
                    ("relu", model.relu),
                    ("maxpool", model.maxpool),
                    ("layer1", model.layer1),
                ]))
                self.tuned_layers = nn.Sequential(OrderedDict([
                    ("layer2", model.layer2),
                    ("layer3", model.layer3),
                    ("layer4", model.layer4),
                    ("avgpool", model.avgpool),
                ]))

            elif transfer_type == "linear" or transfer_type == "side" or transfer_type == "tinytl-bias":
                self.frozen_layers = nn.Sequential(OrderedDict([
                    ("conv1", model.conv1),
                    ("bn1", model.bn1),
                    ("relu", model.relu),
                    ("maxpool", model.maxpool),
                    ("layer1", model.layer1),
                    ("layer2", model.layer2),
                    ("layer3", model.layer3),
                    ("layer4", model.layer4),
                    ("avgpool", model.avgpool),
                ]))
                self.tuned_layers = nn.Identity()

            elif transfer_type == "end2end":
                self.frozen_layers = nn.Identity()
                self.tuned_layers = nn.Sequential(OrderedDict([
                    ("conv1", model.conv1),
                    ("bn1", model.bn1),
                    ("relu", model.relu),
                    ("maxpool", model.maxpool),
                    ("layer1", model.layer1),
                    ("layer2", model.layer2),
                    ("layer3", model.layer3),
                    ("layer4", model.layer4),
                    ("avgpool", model.avgpool),
                ]))

            elif (
                    transfer_type == "prompt" or transfer_type == "BiasPrompt") and self.prompt_config.LOCATION == "pad":  # noqa
                self.frozen_layers = nn.Sequential(OrderedDict([
                    ("conv1", model.conv1),
                    ("bn1", model.bn1),
                    ("relu", model.relu),
                    ("maxpool", model.maxpool),
                    ("layer1", model.layer1),
                    ("layer2", model.layer2),
                    ("layer3", model.layer3),
                    ("layer4", model.layer4),
                    ("avgpool", model.avgpool),
                ]))
                self.tuned_layers = nn.Identity()

        # if transfer_type == "tinytl-bias":
        #     for k, p in self.frozen_layers.named_parameters():
        #         if 'bias' not in k:
        #             p.requires_grad = False
        # else:
        #     for k, p in self.frozen_layers.named_parameters():
        #         p.requires_grad = False
        self.transfer_type = transfer_type

    def setup_prompt(self, prompt_config, model):
        # ONLY support below and pad
        self.prompt_location = prompt_config.LOCATION
        if prompt_config.LOCATION == "below":
            return self._setup_prompt_below(prompt_config, model)
        elif prompt_config.LOCATION == "pad":
            return self._setup_prompt_pad(prompt_config, model)
        else:
            raise ValueError(
                "ResNet models cannot use prompt location {}".format(
                    prompt_config.LOCATION))

    def _setup_prompt_below(self, prompt_config, model):
        if prompt_config.INITIATION == "random":
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, self.prompt_length,
                224, 224
            ))
            nn.init.uniform_(self.prompt_embeddings.data, 0.0, 1.0)
            self.prompt_norm = transforms.Normalize(
                mean=[0.5] * self.prompt_length,
                std=[0.5] * self.prompt_length,
            )

        elif prompt_config.INITIATION == "gaussian":
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, self.prompt_length,
                224, 224
            ))

            nn.init.normal_(self.prompt_embeddings.data)

            self.prompt_norm = nn.Identity()

        elif prompt_config.INITIATION == "xavier":
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, self.prompt_length,
                224, 224
            ))

            nn.init.xavier_uniform_(self.prompt_embeddings.data, gain=1)

            self.prompt_norm = nn.Identity()

        else:
            raise ValueError("Other initiation scheme is not supported")

        # modify first conv layer
        old_weight = model.conv1.weight  # [64, 3, 7, 7]
        model.conv1 = nn.Conv2d(
            self.prompt_length + 3, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        torch.nn.init.xavier_uniform(model.conv1.weight)

        model.conv1.weight[:, :3, :, :].data.copy_(old_weight)
        return model

    def _setup_prompt_pad(self, prompt_config, model):
        if prompt_config.INITIATION == "random":
            self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                1,
                3,
                2 * self.prompt_length,
                224 + 2 * self.prompt_length
            ))
            self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                1,
                3,
                224,
                2 * self.prompt_length
            ))

            nn.init.uniform_(self.prompt_embeddings_tb.data, 0.0, 1.0)
            nn.init.uniform_(self.prompt_embeddings_lr.data, 0.0, 1.0)

            self.prompt_norm = transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )

        elif prompt_config.INITIATION == "gaussian":
            self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                1,
                3,
                2 * self.prompt_length,
                224 + 2 * self.prompt_length
            ))
            self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                1,
                3,
                224,
                2 * self.prompt_length
            ))

            nn.init.normal_(self.prompt_embeddings_tb.data)
            nn.init.normal_(self.prompt_embeddings_lr.data)

            self.prompt_norm = nn.Identity()
        elif prompt_config.INITIATION == "xavier":
            self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                1,
                3,
                2 * self.prompt_length,
                224 + 2 * self.prompt_length
            ))
            self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                1,
                3,
                224,
                2 * self.prompt_length
            ))

            nn.init.xavier_uniform_(self.prompt_embeddings_tb.data, gain=1)
            nn.init.xavier_uniform_(self.prompt_embeddings_lr.data, gain=1)

            self.prompt_norm = nn.Identity()
        else:
            raise ValueError("Other initiation scheme is not supported")
        return model

    def incorporate_prompt(self, x):
        B = x.shape[0]
        if self.prompt_location == "below":
            x = torch.cat((
                x,
                self.prompt_norm(
                    self.prompt_embeddings).expand(B, -1, -1, -1),
            ), dim=1)
            # (B, 3 + num_prompts, crop_size, crop_size)

        elif self.prompt_location == "pad":
            prompt_emb_lr = self.prompt_norm(
                self.prompt_embeddings_lr).expand(B, -1, -1, -1)
            prompt_emb_tb = self.prompt_norm(
                self.prompt_embeddings_tb).expand(B, -1, -1, -1)

            x = torch.cat((
                prompt_emb_lr[:, :, :, :self.prompt_length],
                x, prompt_emb_lr[:, :, :, self.prompt_length:]
            ), dim=-1)
            x = torch.cat((
                prompt_emb_tb[:, :, :self.prompt_length, :],
                x, prompt_emb_tb[:, :, self.prompt_length:, :]
            ), dim=-2)
            # (B, 3, crop_size + num_prompts, crop_size + num_prompts)
        else:
            raise ValueError("not supported yet")
        x = self.prompt_layers(x)
        return x

    def get_features(self, x):
        if self.frozen_layers.training:
            self.frozen_layers.eval()

        if "prompt" not in self.transfer_type and "BiasPrompt" not in self.transfer_type:
            with torch.set_grad_enabled(self.frozen_layers.training):
                x = self.frozen_layers(x)
        else:
            # prompt tuning required frozen_layers saving grad
            x = self.incorporate_prompt(x)
            x = self.frozen_layers(x)

        x = self.tuned_layers(x)  # batch_size x 2048 x 1
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x, labels=None):
        x = self.get_features(x)
        logits = self.head(x)

        if labels is not None:
            if self.num_classes == 1:
                # loss_fct = CrossEntropyLoss()
                loss_fct = torch.nn.L1Loss()
                loss = loss_fct(logits.view(-1, self.num_classes),
                                labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes),
                                labels.view(-1))

            return loss
        else:
            return logits, None
