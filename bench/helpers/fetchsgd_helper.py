import torch
import torch.nn as nn

from .csvec import CSVec


class FetchSGDHelper:
    def __init__(self, grad_size, cfg):
        self.n_row = cfg.num_rows
        self.n_col = cfg.num_cols
        self.n_blocks = cfg.num_blocks
        self.k = cfg.k
        self.grad_size = grad_size
        self.device = torch.device(cfg.gpu_id)

        shape = (self.n_row, self.n_col)
        self.Vvelocity = torch.zeros(shape, device=self.device)  # 动量
        self.Verror = torch.zeros(shape, device=self.device)  # 误差
        self.rho = cfg.virtual_momentum

    def fetch_process(self, sketched_grad, Vvelocity, Verror):
        torch.add(sketched_grad, Vvelocity, alpha=self.rho, out=Vvelocity)

        # virtual error
        Verror += Vvelocity  # \ita S_u^{t+1} + S_e^t
        sketch = self._args2sketch()
        sketch.accumulateTable(Verror)
        update = sketch.unSketch(k=self.k)  # delta grad

        sketch.zero()
        sketch.accumulateVec(update)
        sketched_update = sketch.table

        # this should work but doesn't (model diverges)
        # Verror -= sketched_update
        # instead, zero out Verror with sketched_update.nonzero()
        nz = sketched_update.nonzero()
        Verror[nz[:, 0], nz[:, 1]] = 0

        # momentum factor masking is annoying for sketched
        # to do it properly, we'd have to:
        # first, pull out the values of momentums where update is nonzero
        # then, sketch those values, and subtract them from momentums
        # this requires an unsketch of all num_workers momentum sketch,
        # which is expensive. So instead, just zero out the momentum sketch
        # anywhere where update is nonzero
        nz = sketched_update.nonzero()
        Vvelocity[nz[:, 0], nz[:, 1]] = 0
        _update = update  # lr
        return _update, Vvelocity, Verror  # lr = 0.001

    def sketch_grad(self, grad_vec):
        sketch = self._args2sketch()
        sketch.accumulateVec(grad_vec)
        return sketch.table

    def _args2sketch(self):
        return CSVec(d=self.grad_size, c=self.n_col, r=self.n_row, device=self.device, numBlocks=self.n_blocks)

    def set_param_vec(self, model_state_dict, param_vec, req_grad_params):
        start = 0
        for k in model_state_dict:
            if k in req_grad_params:
                p = model_state_dict[k]
                end = start + p.numel()
                # p.data.zero_()
                p.add_(param_vec[start:end].cpu().view(p.size()))
                start = end

    def get_param_vec(self, model_state_dict):
        param_vec = []
        for k in model_state_dict:
            p = model_state_dict[k]
            if p.requires_grad:
                param_vec.append(p.data.view(-1).float())
        return torch.cat(param_vec)
