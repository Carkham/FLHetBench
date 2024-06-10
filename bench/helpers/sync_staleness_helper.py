from collections import deque, OrderedDict
from heapq import heappush, heappop
from copy import deepcopy

class SyncStalenessHelper:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_cache = deque()
        self.client_task_model_version = {}
        self.min_pq = []
        self.has_task = False

    def new_task(self, client, cur_round, end_time):
        heappush(self.min_pq, (end_time, cur_round, client))
        self.has_task = True

    def save_model(self, model):
        def _deepcopy(model):
            _model = OrderedDict()
            for name, p in model.items():
                _model[name] = p.clone()
            return _model
        if self.has_task:
            self.model_cache.appendleft(_deepcopy(model))
        else:
            self.model_cache.appendleft(None)
        if len(self.model_cache) > self.cfg.max_staleness + 1:
            self.model_cache.pop()

    def exe_task(self, cur_time, cur_round):
        ext_clients = []
        if len(self.min_pq) > 0:
            head_task = self.min_pq[0]
            while head_task[0] <= cur_time:
                if cur_round - head_task[1] <= self.cfg.max_staleness:
                    model = self.model_cache[cur_round - head_task[1]]
                    client = head_task[2]
                    end_time = head_task[0]
                    ext_clients.append((end_time, cur_round - head_task[1], model, client))
                heappop(self.min_pq)
                if len(self.min_pq) > 0:
                    head_task = self.min_pq[0]
                else:
                    break
        return ext_clients
