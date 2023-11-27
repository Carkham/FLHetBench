import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from functools import reduce
from operator import add
from modeling import VisionTransformer


class BiasPrompt(VisionTransformer):
    def __init__(self, config, cfg, *args, **kwargs):
        super(BiasPrompt, self).__init__(config, *args, **kwargs)
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
