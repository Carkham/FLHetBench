import copy

import numpy as np
import timeout_decorator
import traceback

from utils.config import Config
from utils.logger import Logger
from copy import deepcopy
from collections import defaultdict, OrderedDict
import torch
import time
import client
from mimic_model import MimicModel, DistMimicModel
from helpers import FetchSGDHelper, MomentumHelper, FedDynHelper, SyncStalenessHelper, \
    SeqStalenessAggHelper, SumStalenessAggHelper, NaiveStalenessAggHelper, FedKDHelper

L = Logger()
logger = L.get_logger()


class Server:

    def __init__(self, client_model, clients=[], cfg: Config = None, valids=None, fixed_loader=None, tests=None):
        self.best_val_acc = 0
        self._cur_time = 0  # simulation time
        self._cur_round = 0
        self.cfg = cfg
        self.seed = cfg.seed
        self.client_model = client_model
        self.model = client_model.get_weights()
        self.req_grad_params = [name for name, param in client_model.model.named_parameters() if param.requires_grad]
        self.selected_clients = []
        self.client_frequency = [0] * (len(clients) + 1)
        self.suc_client_frequency = [0] * (len(clients) + 1)
        self.client_intervals = {}
        self.all_clients = clients
        self.updates = []
        self.gradiants = []
        self.deltas = []
        self.hs = []
        self.structure_updater = None
        self.fixed_loader = fixed_loader
        self.valids = valids
        self.tests = tests
        self.device = torch.device(self.cfg.gpu_id)

        for c in self.all_clients:
            c.val = valids

        if cfg.fedkd:
            self.inter_model = DistMimicModel(cfg)
        else:
            self.inter_model = MimicModel(cfg)
        # self.trainable_inter_model = copy.deepcopy(self.inter_model)
        self.trainable_inter_model = self.inter_model
        self.trainable_inter_model.cfg = self.cfg

        if cfg.fetchsgd:
            self.fetch_helper = FetchSGDHelper(self.inter_model.size, cfg)
            for c in self.all_clients:
                c.set_model_size(self.fetch_helper.n_row * self.fetch_helper.n_col)

        if cfg.virtual_momentum > 0:
            self.momentum_helper = MomentumHelper(cfg, self.inter_model.model)

        self.fedkd_helper = None
        if cfg.fedkd:
            self.fedkd_helper = FedKDHelper(cfg, self.inter_model, len(self.all_clients))
            self.cfg.fedkd_helper = self.fedkd_helper

        if cfg.feddyn:
            self.feddyn_helper = FedDynHelper(cfg, self.inter_model.model, len(self.all_clients))
            self.cfg.feddyn_helper = self.feddyn_helper

        if cfg.sync_staleness:
            self.sync_staleness_helper = SyncStalenessHelper(cfg)
            if cfg.stale_aggregator == "seq":
                self.staleness_agg_helper = SeqStalenessAggHelper(cfg)
            elif cfg.stale_aggregator == "sum":
                self.staleness_agg_helper = SumStalenessAggHelper(cfg)
            elif cfg.stale_aggregator == "naive":
                self.staleness_agg_helper = NaiveStalenessAggHelper(cfg)
            else:
                raise NotImplementedError("Fuck")

        if cfg.timelyfl:
            self.timelyfl_clients = []

    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        if num_clients < self.cfg.min_selected:
            logger.info('insufficient clients: need {} while get {} online'.format(self.cfg.min_selected, num_clients))
            return False
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None, deadline=-1, round_idx=None):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
            deadline: -1 for unlimited; >0 for each client's deadline
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """

        def deep_copy_state(update, gradiant, device=self.device):
            _update = OrderedDict() if isinstance(update, dict) else update
            _gradiant = OrderedDict() if isinstance(gradiant, dict) else gradiant
            for name in self.req_grad_params:
                if isinstance(update, dict) and name in update:
                    _update[name] = update[name].clone().detach() if device == update[name].device else update[name].to(
                        device).detach()
                if isinstance(gradiant, dict) and name in gradiant:
                    _gradiant[name] = gradiant[name].clone().detach() if device == gradiant[name].device else gradiant[
                        name].to(device).detach()
            return _update, _gradiant

        ## if FedProx used, then update the code

        if (self._cur_round % 1 == 0) and self.cfg.fedprox:
            logger.info("************ proxy weights updated ************")
            self.cfg.proxy_weights = deepcopy(self.model)

        if clients is None:
            clients = self.selected_clients

        sys_metrics = {}
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

        for c in clients:
            # gc.collect()
            # torch.cuda.empty_cache()
            # c.model = deepcopy(self.inter_model)
            c.model = self.trainable_inter_model
            # c.model = createmodel(self.cfg)
            # if (self.client_model_list[0] == ""):
            # self.client_model_list[0] = createmodel(self.cfg)
            # self.client_model_list[0].load_weights(self.model)
            # c.model = self.client_model
            # c.size_count = sum(p.numel() for p in mmm.parameters())
            if self.cfg.feddyn:
                self.feddyn_helper.preprocess(int(c.id), self.inter_model.model)
                self.cfg.server_model = self.model
            if self.cfg.fetchsgd or self.cfg.virtual_momentum > 0:
                self.cfg.server_model = self.model
            if self.cfg.fedkd:
                student_model, teacher_model = self.fedkd_helper.get_distill_model(int(c.id))
                c.model.load_tea_stu(student_model, teacher_model)
            else:
                c.model.load_weights(self.model)

            if self.cfg.fetchsgd:
                c.set_model_size(self.fetch_helper.n_row * self.fetch_helper.n_col)
            try:
                # set deadline
                c.set_deadline(deadline)
                # training
                logger.debug('client {} starts training...'.format(c.id))
                start_t = self.get_cur_time()
                client.updatetimer = self._cur_round
                # gradiant here is actually (-1) * grad
                simulate_time_c, comp, num_samples, update, acc, loss, gradiant, update_size, seed, shape_old, loss_old = c.train(
                    start_t, num_epochs, batch_size, minibatch, self.fixed_loader, self.fedkd_helper)
                # for name in self.model:
                #     update[name] = update[name].cpu()
                # print(torch.cuda.memory_allocated())
                logger.debug('client {} simulate_time: {}'.format(c.id, simulate_time_c))
                logger.debug('client {} num_samples: {}'.format(c.id, num_samples))
                logger.debug('client {} acc: {}, loss: {}'.format(c.id, acc, loss))
                accs.append(acc)
                losses.append(loss)
                simulate_time = min(deadline, max(simulate_time, simulate_time_c))
                simulate_time_list.append(
                    {"idx": len(simulate_time_list), "cid": str(c.id), "simulate_time": simulate_time_c})
                # uploading
                # self.updates.append((c.id, num_samples, update))
                if self.cfg.cpu_updates:
                    update, gradiant = deep_copy_state(update, gradiant, device=torch.device("cpu"))
                else:
                    update, gradiant = deep_copy_state(update, gradiant)

                if self.cfg.fetchsgd:
                    self.updates.append(self.fetch_helper.sketch_grad(gradiant))
                    gradiant = None
                    update = None
                elif self.cfg.virtual_momentum > 0:
                    self.updates.append(gradiant)
                    gradiant = None
                    update = None
                elif self.cfg.feddyn:
                    self.feddyn_helper.postprocess(int(c.id), update, self.model)
                    self.updates.append((int(c.id), update))
                    gradiant = None
                else:
                    self.updates.append(update)
                    gradiant = None

                self.gradiants.append((c.id, num_samples, gradiant))

                """
                norm_comp = int(comp/self.client_model.flops)
                if norm_comp == 0:
                    logger.error('comp: {}, flops: {}'.format(comp, self.client_model.flops))
                    assert False
                self.clients_info[str(c.id)]["comp"] += norm_comp
                """
                # print('client {} upload successfully with acc {}, loss {}'.format(c.id,acc,loss))
                logger.debug('client {} upload successfully with acc {}, loss {}'.format(c.id, acc, loss))
            except timeout_decorator.timeout_decorator.TimeoutError as e:
                # logger.debug('client {} failed: {}'.format(c.id, e))
                # if "interruption" in str(e):
                logger.debug('client {} failed: {}'.format(c.id, e))
                actual_comp = c.get_actual_comp()
                # norm_comp = int(actual_comp/self.client_model.flops)
                # self.clients_info[str(c.id)]["comp"] += norm_comp

                simulate_time = deadline
                if self.cfg.sync_staleness:
                    # logger.info(f"Staleness client {c.id} cached in {self._cur_round} with {c.cost_time}")
                    self.sync_staleness_helper.new_task(c, self._cur_round, c.cost_time)
                if self.cfg.timelyfl:
                    ddl_ori_time = c.timer.get_ori_time(self.get_cur_time(), deadline)
                    self.timelyfl_clients.append({"client": c,
                                                  "ori_time": float(e.value.split("\t")[-1]),
                                                  "ddl_ori_time": ddl_ori_time
                                                  })
            except Exception as e:
                logger.error('client {} failed: {}'.format(c.id, e))
                traceback.print_exc()
            finally:
                c.model = ""
                client_count += 1

        try:
            # logger.info('simulation time: {}'.format(simulate_time))
            if int(self.cfg.drop_rate * len(self.updates)) > 0. and self.cfg.target_acc:
                drop_num = int(self.cfg.drop_rate * len(self.updates))
                simulate_time_list = sorted(simulate_time_list, key=lambda x: x["simulate_time"], reverse=True)
                simulate_time_list = simulate_time_list[drop_num:]
                self.updates = [self.updates[info["idx"]] for info in simulate_time_list]
            data_list = [s["simulate_time"] for s in simulate_time_list]
            simulate_time = min(deadline, max(data_list) if len(data_list) != 0 else deadline)

            if self.cfg.sync_staleness:
                self.sync_staleness_helper.save_model(self.model)
                exe_clients = self.sync_staleness_helper.exe_task(self._cur_time + deadline, self._cur_round)
                for end_time, staleness_round, model, c in exe_clients:
                    c.model = self.trainable_inter_model
                    c.model.load_weights(model)
                    (simulate_time_c, comp, num_samples, update, acc, loss, gradiant, update_size, seed, shape_old,
                     loss_old) = c.train_wo_sim_time(None, num_epochs, batch_size, minibatch, self.fixed_loader,
                                                     self.fedkd_helper)
                    accs.append(acc)
                    losses.append(loss)
                    # logger.info(f"Staleness client {c.id} end in {end_time} with {end_time - self._cur_time}")
                    simulate_time = min(deadline, max(simulate_time, end_time - self._cur_time))
                    update, gradiant = deep_copy_state(update, gradiant)

                    if self.cfg.fetchsgd:
                        self.updates.append({
                            "client_id": int(c.id),
                            "staleness_round": staleness_round,
                            "update": self.fetch_helper.sketch_grad(gradiant)
                        })
                        gradiant = None
                        update = None
                    elif self.cfg.virtual_momentum > 0:
                        self.updates.append({
                            "client_id": int(c.id),
                            "staleness_round": staleness_round,
                            "update": gradiant
                        })
                        gradiant = None
                        update = None
                    elif self.cfg.feddyn:
                        self.feddyn_helper.postprocess(int(c.id), update, self.model)
                        self.updates.append({
                            "client_id": int(c.id),
                            "staleness_round": staleness_round,
                            "update": update
                        })
                        gradiant = None
                    else:
                        self.updates.append({
                            "client_id": int(c.id),
                            "staleness_round": staleness_round,
                            "update": update
                        })
                        gradiant = None

            if self.cfg.timelyfl:
                ratios = []
                for info_dict in self.timelyfl_clients:
                    c = info_dict["client"]
                    ori_time = info_dict["ori_time"]
                    ddl_ori_time = info_dict["ddl_ori_time"]
                    ratio = (ddl_ori_time / ori_time)
                    c.model = deepcopy(self.trainable_inter_model)
                    c.model.load_weights(self.model)
                    c.model.setup_partial_grad(ratio)
                    (simulate_time_c, comp, num_samples, update, acc, loss, gradiant, update_size, seed, shape_old,
                     loss_old) = c.train_wo_sim_time(None, num_epochs, batch_size, minibatch, self.fixed_loader,
                                                     self.fedkd_helper)
                    accs.append(acc)
                    losses.append(loss)
                    # logger.info(f"Staleness client {c.id} end in {end_time} with {end_time - self._cur_time}")
                    simulate_time = min(deadline, max(simulate_time, deadline))

                    cnt_name = 0
                    cnt_train_name = 0
                    _update = {}
                    for name, param in c.model.model.named_parameters():
                        cnt_name += 1
                        if param.requires_grad:
                            cnt_train_name += 1
                            _update = update[name]
                    update, gradiant = deep_copy_state(_update, gradiant)
                    ratios.append((ratio, ddl_ori_time, ori_time, cnt_train_name / cnt_name))

                    if self.cfg.fetchsgd:
                        self.updates.append(self.fetch_helper.sketch_grad(gradiant))
                        gradiant = None
                        update = None
                    elif self.cfg.virtual_momentum > 0:
                        self.updates.append(gradiant)
                        gradiant = None
                        update = None
                    elif self.cfg.feddyn:
                        self.feddyn_helper.postprocess(int(c.id), update, self.model)
                        self.updates.append((int(c.id), update))
                        gradiant = None
                    else:
                        self.updates.append(update)
                        gradiant = None

                    self.gradiants.append((c.id, num_samples, gradiant))
                logger.info(f"TimelyFL optimize partial training clients (ratio ddl_ori ori) {ratios}")

            sys_metrics = {'configuration_time': simulate_time}
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

    def update(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.updates) / len(self.selected_clients) >= update_frac:
            logger.info('round succeed, updating global model...')

            ## update

            total = 1
            # To Device
            with torch.no_grad():
                for states in self.updates:
                    if "client_id" in states:
                        inverted_staleness = 1 / (1 + states["staleness_round"]) ** 0.5
                        states = states["update"]
                    else:
                        inverted_staleness = 1
                    # for name in self.model:
                    for name in self.req_grad_params:
                        self.model[name] = self.model[name].to(self.device)
                        states[name] = states[name].to(self.device)
                        self.model[name] += states[name] * inverted_staleness
                    total += 1
                # for name in self.model:
                for name in self.req_grad_params:
                    self.model[name] /= total
        else:
            logger.info('round failed, global model maintained.')

        self.updates = []

    def update_using_timelyfl(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.updates) / len(self.selected_clients) >= update_frac:
            logger.info('round succeed, updating global model...')

            ## update

            total = {name: 1 for name in self.req_grad_params}
            # To Device
            with torch.no_grad():
                for states in self.updates:
                    # for name in self.model:
                    for name in self.req_grad_params:
                        if name in states:
                            self.model[name] = self.model[name].to(self.device)
                            states[name] = states[name].to(self.device)
                            self.model[name] += states[name]
                            total[name] += 1
                # for name in self.model:
                for name in self.req_grad_params:
                    self.model[name] /= total[name]
        else:
            logger.info('round failed, global model maintained.')

        self.updates = []
        self.timelyfl_clients = []

    def update_using_feddyn(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.updates) / len(self.selected_clients) >= update_frac:
            logger.info('round succeed, updating global model...')

            with torch.no_grad():
                clnt_idxs, updates = [u[0] for u in self.updates], [u[1] for u in self.updates]
                local_param_lists = self.feddyn_helper.get_local_param_list(clnt_idxs)
                for name in self.req_grad_params:
                    self.model[name] = torch.zeros_like(updates[0][name]).to(self.device)
                    _local_param = torch.zeros_like(local_param_lists[0][name])
                    for states in updates:
                        self.model[name] += states[name].to(self.device)
                    for local_param in local_param_lists:
                        _local_param += local_param[name]
                    self.model[name] += _local_param.to(self.device)
                    self.model[name] /= len(updates)
                    # for states in updates:
                    #     if isinstance(self.model[name], list):
                    #         self.model[name].append(states[name])
                    #     else:
                    #         self.model[name] = [states[name]]
                    #     states[name] = None
                    # self.model[name] = torch.stack(self.model[name], dim=0).mean(0).to(self.device)
                    # self.model[name] += torch.stack([local_param[name] for local_param in local_param_lists]).mean(
                    #     0).to(self.device)

        else:
            logger.info('round failed, global model maintained.')

        self.updates = []

    def update_using_fetchsgd(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.updates) / len(self.selected_clients) >= update_frac:
            logger.info('round succeed, updating global model...')
            ## update
            g = torch.zeros_like(self.fetch_helper.Vvelocity).to(self.fetch_helper.device)
            for states in self.updates:
                g = g + states if g is not None else states
            g /= len(self.updates)  # grad

            print(f"sketched_grad: {torch.sum(torch.abs(g))}")
            weight_update, new_Vvelocity, new_Verror = self.fetch_helper.fetch_process(
                g.to(self.fetch_helper.device),
                self.fetch_helper.Vvelocity,
                self.fetch_helper.Verror)

            self.fetch_helper.set_param_vec(self.model, weight_update, self.req_grad_params)
            self.fetch_helper.Vvelocity[:] = new_Vvelocity
            self.fetch_helper.Verror[:] = new_Verror
            print(
                f"Velocity: {torch.sum(torch.abs(new_Vvelocity))} Error: {torch.sum(torch.abs(new_Verror))} Update: {torch.sum(torch.abs(weight_update))}")
        else:
            logger.info('round failed, global model maintained.')

        self.updates = []
        self.gradiants = []

    def update_using_avg_momentum(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        with torch.no_grad():
            if len(self.updates) / len(self.selected_clients) >= update_frac:
                logger.info('round succeed, updating global model...')
                ## update
                if self.cfg.sync_staleness:
                    staleness_updates = []
                    current_updates = []
                    for update in self.updates:
                        if "client_id" in update:
                            staleness_updates.append(update)
                        else:
                            current_updates.append(update)
                    self.updates = current_updates
                    logger.info(
                        f"staleness {len(staleness_updates)} to {len(current_updates)} clients: {[(u['client_id'], u['staleness_round']) for u in staleness_updates]}")

                if len(self.updates) == 0:
                    accumulated_grad = self.momentum_helper.zeros_like_grad()
                else:
                    accumulated_grad = OrderedDict()
                    for name in self.req_grad_params:
                        accumulated_grad[name] = torch.mean(
                            torch.stack([states[name] for states in self.updates], dim=0), dim=0
                        ).to(self.device)

                # for states in self.updates:
                #     for name in self.req_grad_params:
                #         accumulated_grad[name] += states[name].to(self.device)
                #
                # for name in self.req_grad_params:
                #     accumulated_grad[name] /= len(self.updates)

                accumulated_grad, new_velocity = self.momentum_helper.update_with_momentum(
                    accumulated_grad,
                    self.momentum_helper.velocity,
                    self.req_grad_params
                )

                if self.cfg.sync_staleness:
                    accumulated_grad = self.staleness_agg_helper.aggregate_grad(
                        accumulated_grad,
                        staleness_updates,
                        self.req_grad_params
                    )

                sum_v = 0.
                sum_up = 0.
                for name in self.req_grad_params:
                    p = self.model[name].to(self.device)
                    p.add_(accumulated_grad[name])
                    self.model[name] = p
                    self.momentum_helper.velocity[name][:] = new_velocity[name]
                    sum_v += torch.sum(new_velocity[name].abs())
                    sum_up += torch.sum(accumulated_grad[name].abs())
                print(f"Velocity: {sum_v} Update: {sum_up}")
            else:
                logger.info('round failed, global model maintained.')

        self.updates = []
        self.gradiants = []

    def update_using_pcgrad(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        with torch.no_grad():
            if len(self.updates) / len(self.selected_clients) >= update_frac:
                logger.info('round succeed, updating global model...')
                ## update
                accumulated_grad = self.momentum_helper.update_with_pcgrad(
                    self.updates,
                    self.req_grad_params
                )

                if self.momentum_helper.pcgrad_momentum:
                    accumulated_grad, new_velocity = self.momentum_helper.update_with_momentum(
                        accumulated_grad,
                        self.momentum_helper.velocity,
                        self.req_grad_params
                    )

                sum_v = 0.
                sum_up = 0.
                for name in self.req_grad_params:
                    p = self.model[name].to(self.device)
                    p.add_(accumulated_grad[name])
                    self.model[name] = p
                    if self.momentum_helper.pcgrad_momentum:
                        self.momentum_helper.velocity[name][:] = new_velocity[name]
                    sum_v += torch.sum(
                        self.momentum_helper.grad_history[name].abs() if self.momentum_helper.grad_history[
                                                                             name] is not None else torch.tensor(0))
                    sum_up += torch.sum(accumulated_grad[name].abs())
                print(f"Velocity: {sum_v} Update: {sum_up}")
            else:
                logger.info('round failed, global model maintained.')

        self.updates = []
        self.gradiants = []

    def update_using_mom_pcgrad(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        with torch.no_grad():
            if len(self.updates) / len(self.selected_clients) >= update_frac:
                logger.info('round succeed, updating global model...')
                ## update

                accumulated_grad_momentum = self.momentum_helper.zeros_like_grad()

                for name in self.req_grad_params:
                    accumulated_grad_momentum[name] = torch.mean(
                        torch.stack([states[name] for states in self.updates], dim=0), dim=0
                    ).to(self.device)

                accumulated_grad_momentum, new_velocity = self.momentum_helper.update_with_momentum(
                    accumulated_grad_momentum,
                    self.momentum_helper.velocity,
                    self.req_grad_params
                )

                accumulated_grad_pcgrad = self.momentum_helper.update_with_pcgrad(
                    self.updates,
                    self.req_grad_params
                )

                coeff = 0.75
                accumulated_grad = OrderedDict()
                for name in self.req_grad_params:
                    accumulated_grad[name] = coeff * accumulated_grad_momentum[name] + coeff * accumulated_grad_pcgrad[
                        name]

                sum_v = 0.
                sum_up = 0.
                for name in self.req_grad_params:
                    p = self.model[name].to(self.device)
                    p.add_(accumulated_grad[name])
                    self.model[name] = p
                    self.momentum_helper.velocity[name][:] = new_velocity[name]
                    sum_v += torch.sum(
                        self.momentum_helper.grad_history[name].abs() if self.momentum_helper.grad_history[
                                                                             name] is not None else torch.tensor(0))
                    sum_up += torch.sum(accumulated_grad[name].abs())
                print(f"Velocity: {sum_v} Update: {sum_up}")
            else:
                logger.info('round failed, global model maintained.')

        self.updates = []
        self.gradiants = []

    def update_using_fedkd(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.updates) / len(self.selected_clients) >= update_frac:
            logger.info('round succeed, updating global model...')

            update_size = self.fedkd_helper.update_student_model()
            logger.info('round update usv size: {}'.format(update_size))
        else:
            self.fedkd_helper.release_student_model()
            logger.info('round failed, global model maintained.')

        self.updates = []

    def test_model(self, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients
            assert False

        for client in clients_to_test:
            client.model.set_params(self.model)
            c_metrics = client.test(set_to_use)
            # logger.info('client {} metrics: {}'.format(client.id, c_metrics))
            metrics[client.id] = c_metrics

        return metrics

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.all_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model.set_params(self.model)
        model_sess = self.client_model.sess
        return self.client_model.saver.save(model_sess, path)

    def close_model(self):
        self.client_model.close()

    def get_cur_time(self):
        return self._cur_time

    def pass_time(self, sec):
        self._cur_time += sec

    def get_time_window(self):
        tw = np.random.normal(self.cfg.time_window[0], self.cfg.time_window[1])
        while tw < 0:
            tw = np.random.normal(self.cfg.time_window[0], self.cfg.time_window[1])
        return tw

    def final_train_eval(self, model):
        best_model_name = 'best_' + str(self.cfg.round_ddl[0]) + '_s' + str(self.cfg.seed) + '.pt'
        m = torch.load(best_model_name)
        model.load_weights(m)
        for c in self.all_clients:
            totalloss, acc = model.eval(c.dataloader, printing=False)
            client.greatstring += (str(c.id) + " " + str(acc) + "\n")

    def my_eval(self, dataloader, model, load_best=False, is_test=False):
        best_model_name = str(self.cfg.save_name) + '_' + str(self.cfg.dataset_name) + '_' + str(
            self.cfg.round_ddl[0]) + '_s' + str(self.cfg.seed) + '.pt'
        # model_name = str(self.cfg.round_ddl[0]) + '_s' + str(self.cfg.seed) + '.pt'
        if load_best:
            m = torch.load(best_model_name)
            model.load_weights(m)
        else:
            if self.cfg.fedkd:
                model.load_weights(self.fedkd_helper.student_model)
            else:
                model.load_weights(self.model)
        totalloss, acc = model.eval(self.cfg, dataloader)
        logger.info(("(Test)" if is_test else "(Val)") + "Eval loss: " + str(totalloss) + " acc: " + str(acc))
        if acc > self.best_val_acc and not load_best and not is_test:
            torch.save(model.get_weights(), best_model_name)
            self.best_val_acc = acc
            logger.info("saving best")
        return acc
