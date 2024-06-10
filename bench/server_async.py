import copy
import sys
import time
import traceback
from collections import deque, OrderedDict
from heapq import heappush, heappop
import numpy as np
import torch
import json

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY
from utils.config import Config
from utils.logger import Logger
from server import Server
from copy import deepcopy

L = Logger()
logger = L.get_logger()


def get_client_completion_time(cur_time, client, num_data, batch_size, num_epochs, model_size):
    train_time = client.device.get_train_time(num_data, batch_size, num_epochs)
    download_time = client.device.get_download_time()
    upload_time = client.device.get_upload_time(model_size)  # will be re-calculated after training
    down_end_time = client.timer.get_future_time(cur_time, download_time)
    train_end_time = client.timer.get_future_time(down_end_time, train_time)
    up_end_time = client.timer.get_future_time(train_end_time, upload_time)

    return up_end_time - cur_time


def online(clients, cur_time, time_window, excluding_clients=None):
    # """We assume all users are always online."""
    # return online client according to client's timer
    if excluding_clients is None:
        excluding_clients = []
    online_clients = []
    for c in clients:
        try:
            if c.timer.ready(cur_time, time_window) and c.id not in excluding_clients:
                online_clients.append(c)
        except Exception as e:
            traceback.print_exc()
    return online_clients


class AsyncServer(Server):
    def __init__(self, client_model, clients=[], cfg: Config = None, valids=None, fixed_loader=None, tests=None):
        super().__init__(client_model, clients, cfg, valids, fixed_loader, tests)
        self.virtual_client_clock = {}
        self.min_pq = []
        self.model_cache = deque()
        self.client_task_start_times = {}
        self.client_task_model_version = {}
        self.task_running_clients = set()
        self.current_concurrency = 0
        self.aggregation_denominator = 0
        self.rng = np.random.default_rng(42)
        self.num_epoch = 1
        self.batch_size = 16
        self.rounds_info = {}

    def _new_task(self, cur_time):
        def new_task():
            online_clients = online(self.all_clients, cur_time, 0, self.task_running_clients)
            if len(online_clients) == 0:
                return False
            client = self.rng.choice(online_clients, 1)[0]
            exe_cost = get_client_completion_time(
                cur_time,
                client,
                num_data=len(client.train_data),
                num_epochs=self.num_epoch,
                batch_size=self.batch_size,
                model_size=self.trainable_inter_model.size)
            self.virtual_client_clock[client.id] = exe_cost
            end_time = cur_time + exe_cost
            # logger.info(f"Push client: {client.id}\t\"start\" time:{cur_time:.1f}\t\"end\"\ttime:{end_time:.1f}")
            heappush(self.min_pq, (cur_time, 'start', client))
            heappush(self.min_pq, (end_time, 'end', client))
            self.client_task_start_times[client.id] = cur_time
            self.client_task_model_version[client.id] = self._cur_round
            return True

        # while not new_task():
        #     cur_time = cur_time + self.get_time_window()
        return new_task()

    def tictak_client_tasks(self, num_clients_to_collect):
        """
        Record sampled client execution information in last round. In the SIMULATION_MODE,
        further filter the sampled_client and pick the top num_clients_to_collect clients.
        :param num_clients_to_collect: The number of clients actually needed for next round. K in FedBUFF.
        :return: Tuple: (the List of clients to run, the List of stragglers in the round, a Dict of the virtual clock of each
            client, the duration of the aggregation round, and the durations of each client's task).
        """
        self.model_cache.appendleft(deepcopy(self.model))
        if len(self.model_cache) > self.cfg.max_staleness + 1:
            self.model_cache.pop()
        clients_to_run = []
        durations = []
        final_time = self.get_cur_time()
        if not self.min_pq:
            self._new_task(self.get_cur_time())
        while len(clients_to_run) < num_clients_to_collect:
            event_time, event_type, client = heappop(self.min_pq)
            if event_type == 'start':
                self.task_running_clients.add(client.id)
                self.current_concurrency += 1
                if self.current_concurrency < self.cfg.max_concurrency:
                    self._new_task(event_time)
            else:
                self.current_concurrency -= 1
                if self.current_concurrency <= self.cfg.max_concurrency - 1:
                    self._new_task(event_time)
                if self._cur_round - self.client_task_model_version[client.id] <= self.cfg.max_staleness:
                    clients_to_run.append(client)
                durations.append(event_time - self.client_task_start_times[client.id])
                final_time = event_time
            if not self.min_pq:
                while not self._new_task(self.get_cur_time()):
                    self.pass_time(self.get_time_window())
        self.pass_time(final_time - self.get_cur_time())
        # self.task_running_clients.clear()
        self.rounds_info[self._cur_round] = {
            "start_times": [self.client_task_start_times[c.id] for c in clients_to_run],
            "clients": [c.id for c in clients_to_run],
            "client_clock": [self.virtual_client_clock[c.id] for c in clients_to_run],
            "durations": durations,
            "model_versions": [self.client_task_model_version[c.id] for c in clients_to_run]
        }
        for client in clients_to_run:
            if client.id in self.task_running_clients:
                self.task_running_clients.remove(client.id)
        return clients_to_run, [], self.virtual_client_clock, 0, durations

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None, deadline=-1, round_idx=None):
        deadline = sys.maxsize
        self.selected_clients = clients

        def deep_copy_state(_update, gradiant):
            ret_update = OrderedDict() if isinstance(_update, dict) else _update
            ret_gradiant = OrderedDict() if isinstance(gradiant, dict) else gradiant
            for name in self.req_grad_params:
                if isinstance(_update, dict) and name in _update:
                    ret_update[name] = _update[name].clone().detach()
                if isinstance(gradiant, dict) and name in gradiant:
                    ret_gradiant[name] = gradiant[name].clone().detach()
            return ret_update, ret_gradiant

        for client in self.selected_clients:
            self.client_frequency[int(client.id)] += 1
            self.clients_info[str(client.id)]["freq"] += 1
            self.clients_info[str(client.id)]["selected_round"].append(round_idx)

        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0,
                   'acc': {},
                   'loss': {},
                   'val_acc': {},
                   'val_loss': {},
                   'test_acc': {},
                   'test_loss': {},
                   'ori_d_t': 0,
                   'ori_t_t': 0,
                   'ori_u_t': 0,
                   'act_d_t': 0,
                   'act_t_t': 0,
                   'act_u_t': 0} for c in clients}
        # for c in self.all_clients:
        # c.model.set_params(self.model)
        simulate_time = 0
        simulate_time_list = []
        accs = []
        losses = []
        self.updates = []
        self.gradiants = []
        self.deltas = []
        self.hs = []
        client_count = 0
        preprocess_time = 0
        load_model_time = 0
        clnt_train_time = 0
        deepcopy_time = 0
        postprocess_time = 0

        logger.info("Client loading {}".format({c.id: self.client_task_model_version[c.id] for c in clients}))

        for c in clients:
            _t = time.time()
            c.model = self.trainable_inter_model

            # preprocess for different algorithm
            if self.cfg.feddyn:
                self.feddyn_helper.preprocess(int(c.id), self.inter_model.model)
                self.cfg.server_model = self.model
                preprocess_time += time.time() - _t
                _t = time.time()
            if self.cfg.virtual_momentum > 0:
                self.cfg.server_model = self.model

            if self.cfg.fedkd:
                student_model, teacher_model = self.fedkd_helper.get_distill_model(int(c.id))
                c.model.load_tea_stu(student_model, teacher_model)
            else:
                c.model.load_weights(self.model_cache[self._cur_round - self.client_task_model_version[c.id]])
                load_model_time += time.time() - _t
                _t = time.time()
            if self.cfg.fetchsgd:
                c.set_model_size(self.fetch_helper.n_row * self.fetch_helper.n_col)
            try:
                # set deadline
                c.set_deadline(deadline)
                # training
                logger.debug('client {} starts training...'.format(c.id))
                start_t = self.client_task_start_times[c.id]
                # gradiant here is actually (-1) * grad
                (simulate_time_c, comp, num_samples, update, acc, loss, gradiant, update_size, seed, shape_old,
                 loss_old) = c.train_wo_sim_time(start_t, num_epochs, batch_size, minibatch, self.fixed_loader,
                                                 self.fedkd_helper)
                simulate_time_c = self.virtual_client_clock[c.id]
                clnt_train_time += time.time() - _t
                _t = time.time()
                accs.append(acc)
                losses.append(loss)
                simulate_time = min(deadline, max(simulate_time, simulate_time_c))
                simulate_time_list.append(
                    {"idx": len(simulate_time_list), "cid": str(c.id), "simulate_time": simulate_time_c})
                sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                sys_metrics[c.id][BYTES_WRITTEN_KEY] += update_size
                sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
                sys_metrics[c.id]['acc'] = acc
                sys_metrics[c.id]['loss'] = loss
                # uploading
                # self.updates.append((c.id, num_samples, update))
                update, gradiant = deep_copy_state(update, gradiant)
                deepcopy_time += time.time() - _t
                _t = time.time()

                if self.cfg.fetchsgd:
                    self.updates.append({
                        "result": self.fetch_helper.sketch_grad(gradiant),
                        "client_id": c.id,
                    })
                    gradiant = None
                    update = None
                elif self.cfg.virtual_momentum > 0:
                    self.updates.append({
                        "result": gradiant,
                        "client_id": c.id,
                    })
                    gradiant = None
                    update = None
                elif self.cfg.feddyn:
                    self.feddyn_helper.postprocess(int(c.id), update, self.model)
                    self.updates.append({
                        "result": update,
                        "client_id": c.id,
                    })
                    gradiant = None
                else:
                    self.updates.append({
                        "result": update,
                        "client_id": c.id,
                    })
                    gradiant = None

                postprocess_time += time.time() - _t
                _t = time.time()

                """
                norm_comp = int(comp/self.client_model.flops)
                if norm_comp == 0:
                    logger.error('comp: {}, flops: {}'.format(comp, self.client_model.flops))
                    assert False
                self.clients_info[str(c.id)]["comp"] += norm_comp
                """
                # print('client {} upload successfully with acc {}, loss {}'.format(c.id,acc,loss))
                logger.debug('client {} upload successfully with acc {}, loss {}'.format(c.id, acc, loss))
                self.clients_info[str(c.id)]["suc_freq"] += 1
                self.suc_client_frequency[int(c.id)] += 1
            except Exception as e:
                logger.error('client {} failed: {}'.format(c.id, e))
                traceback.print_exc()
            finally:
                c.model = ""
                client_count += 1

                if self.cfg.compress_algo:
                    sys_metrics[c.id]['before_cprs_u_t'] = round(c.before_comp_upload_time, 3)
                sys_metrics[c.id]['ori_d_t'] = round(c.ori_download_time, 3)
                sys_metrics[c.id]['ori_t_t'] = round(c.ori_train_time, 3)
                sys_metrics[c.id]['ori_u_t'] = round(c.ori_upload_time, 3)

                sys_metrics[c.id]['act_d_t'] = round(c.act_download_time, 3)
                sys_metrics[c.id]['act_t_t'] = round(c.act_train_time, 3)
                sys_metrics[c.id]['act_u_t'] = round(c.act_upload_time, 3)
        logger.info('pre: {} lm: {} ct: {} dc: {}, post: {}'.format(preprocess_time, load_model_time, clnt_train_time,
                                                                    deepcopy_time, postprocess_time))

        try:
            # logger.info('simulation time: {}'.format(simulate_time))
            if int(self.cfg.drop_rate * len(self.updates)) > 0. and self.cfg.target_acc:
                drop_num = int(self.cfg.drop_rate * len(self.updates))
                simulate_time_list = sorted(simulate_time_list, key=lambda x: x["simulate_time"], reverse=True)
                simulate_time_list = simulate_time_list[drop_num:]
                self.updates = [self.updates[info["idx"]] for info in simulate_time_list]
            data_list = [s["simulate_time"] for s in simulate_time_list]
            simulate_time = min(deadline, max(data_list) if len(data_list) != 0 else deadline)
            for info in simulate_time_list:
                self.clients_info[info["cid"]]["wait_time_info"].append({
                    "sim_t": simulate_time,
                    "wait_t": simulate_time - info["simulate_time"]
                })

            sys_metrics['configuration_time'] = simulate_time
            avg_acc = sum(accs) / len(accs)
            avg_loss = sum(losses) / len(losses)
            logger.info('average acc: {}, average loss: {}'.format(avg_acc, avg_loss))
            logger.info('configuration and update stage simulation time: {}'.format(simulate_time))
            logger.info('losses: {}'.format(losses))
        except ZeroDivisionError as e:
            logger.error('training time window is too short to train!')
            # assert False
        except Exception as e:
            logger.error('failed reason: {}'.format(e))
            traceback.print_exc()
            assert False

        # print('If we donot cuda empty, we get the time', time.time() - time1)
        return sys_metrics

    def update_with_staleness(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.updates) / len(self.selected_clients) >= update_frac:
            logger.info('round succeed, updating global model...')
            total = 0
            # To Device
            with torch.no_grad():
                for states in self.updates:
                    # for name in self.model:
                    update = states["result"]
                    cid = states["client_id"]
                    inverted_staleness = 1 / (1 + self._cur_round - self.client_task_model_version[cid]) ** 0.5
                    for name in self.req_grad_params:
                        if total == 0:
                            self.model[name] = update[name].to(self.device) * inverted_staleness
                        else:
                            self.model[name] += update[name].to(self.device) * inverted_staleness
                    total += inverted_staleness
                # for name in self.model:
                for name in self.req_grad_params:
                    self.model[name] /= total
        else:
            logger.info('round failed, global model maintained.')

        self.updates = []

    def update_with_staleness_momentum(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.updates) / len(self.selected_clients) >= update_frac:
            logger.info('round succeed, updating global model...')
            accumulated_grad = self.momentum_helper.zeros_like_grad()
            total = 0
            # To Device
            with torch.no_grad():
                # sum
                for i, states in enumerate(self.updates):
                    # for name in self.model:
                    update = states["result"]
                    cid = states["client_id"]
                    inverted_staleness = 1 / (1 + self._cur_round - self.client_task_model_version[cid]) ** 0.5
                    for name in self.req_grad_params:
                        accumulated_grad[name] += update[name].to(self.device) * inverted_staleness
                    # total += inverted_staleness
                    total += 1
                    self.updates[i] = None
                # avg
                # norm_type = 2.0
                # max_norm = 10
                # grads = [accumulated_grad[name] for name in self.req_grad_params]
                # total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(self.device) for g in grads]), norm_type)
                # clip_coef = max_norm / (total_norm + 1e-6)
                # clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

                weight_update = []
                for name in self.req_grad_params:
                    # accumulated_grad[name].div_(total).mul_(clip_coef_clamped.to(self.device))
                    accumulated_grad[name] /= total
                    weight_update.append(accumulated_grad[name].reshape(-1).clone())

                accumulated_grad, new_velocity = self.momentum_helper.update_with_momentum(
                    accumulated_grad,
                    self.momentum_helper.velocity,
                    self.req_grad_params
                )

                new_Vvelocity = []

                for name in self.req_grad_params:
                    new_Vvelocity.append(new_velocity[name].reshape(-1))

                print(
                    f"Velocity: {torch.sum(torch.abs(torch.cat(new_Vvelocity)))} Update: {torch.sum(torch.abs(torch.cat(weight_update)))}")

                for name in self.req_grad_params:
                    p = self.model[name].to(self.device)
                    p.add_(accumulated_grad[name])
                    self.model[name] = p
                    self.momentum_helper.velocity[name][:] = new_velocity[name]
        else:
            logger.info('round failed, global model maintained.')

        self.updates = []
        self.gradiants = []

    def save_clients_info(self):
        super().save_clients_info()
        import os
        if not os.path.exists('clients_info'):
            os.mkdir('clients_info')
        with open('clients_info/AsyncFL_RoundInfo_{}.json'.format(os.path.basename(self.cfg.config_name),
                                                                  self.cfg.save_name,
                                                                  self.cfg.round_ddl[0]), 'w') as f:
            json.dump(self.rounds_info, f)
        logger.info('save rounds_info.json.')
