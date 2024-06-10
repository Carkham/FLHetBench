import random
import warnings
import timeout_decorator
import sys
import numpy as np
import json
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.logger import Logger
from device import Device, create_device
from timer import Timer

from mimic_data import ImageDataset
from helpers import FedKDHelper
from datasets.arrow_dataset import Dataset
from mimic_model import MimicModel

L = Logger()
logger = L.get_logger()

updatetimer = 1


def is_none_suc(devices: dict):
    for uid, device in devices.items():
        timer = Timer(ubt=device, google=True)
        if timer.isSuccess:
            return False
    return True


class Client:
    d = None
    pred_d = None
    idx = 0
    try:
        with open('data/state_traces.json', 'r', encoding='utf-8') as f:
            d = json.load(f)

    except FileNotFoundError as e:
        print(e)
        d = None
        logger.warn('no user behavior trace was found, running in no-trace mode')

    def __init__(self, client_id, group=None, train_data=None, model=None, device=None, cfg=None):
        self.model: MimicModel = model
        self.id = client_id  # integer
        self.group = group
        self.train_data = train_data
        self.eval_data = None
        self.deadline = 1  # < 0 for unlimited
        self.cfg = cfg
        if isinstance(train_data, Dataset):
            self.dataloader = DataLoader(train_data, batch_size=cfg.client_bs)
        else:
            assert len(train_data[0]) == len(train_data[1])
            batch_size = min(len(train_data[0]), cfg.client_bs)
            self.dataloader = DataLoader(dataset=ImageDataset(train_data[0], train_data[1]), batch_size=batch_size,
                                         shuffle=True)

        self.device = device  # if device is none, it will use real time as train time, and set upload/download time as 0
        if self.device is None:
            logger.warn(
                'client {} with no device init, upload time will be set as 0 and speed will be the gpu speed'.format(
                    self.id))
            self.upload_time = 0

        # timer
        d = Client.d
        if d is None:
            cfg.state_hete = False
        # uid = random.randint(0, len(d))
        if cfg.state_hete:
            if cfg.state_path:
                if Client.pred_d is None:
                    with open(cfg.state_path, "r") as f:
                        Client.pred_d = json.load(f)
                uid = Client.pred_d[Client.idx]
                Client.idx += 1
                self.timer = Timer(ubt=d[str(uid)], google=True)
            elif not cfg.real_world:
                uid = random.sample(list(d.keys()), 1)[0]
                self.timer = Timer(ubt=d[str(uid)], google=True)
                while not self.timer.isSuccess:
                    uid = random.sample(list(d.keys()), 1)[0]
                    self.timer = Timer(ubt=d[str(uid)], google=True)
            else:
                uid = self.id
                self.timer = Timer(ubt=d[str(uid)], google=True)
        else:
            # no behavior heterogeneity, always available
            self.timer = Timer(None)
            self.deadline = sys.maxsize  # deadline is meaningless without user trace

        real_device_model = self.timer.model

        if not self.device:
            self.device = create_device(cfg, 0.0)

        if self.cfg.device_hete:
            self.device.set_device_model(real_device_model)
        else:
            self.device.set_device_model("Redmi Note 8")

    def train(self, start_t=None, num_epochs=1, batch_size=10, minibatch=None, fixed_loader=None, distill_helper=None,
              clnt_idx=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
            start_t: strat time of the training, only used in train_with_simulate_time
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            acc, loss, grad, update_size
        """

        def train_with_simulate_time(self: Client,
                                     start_t,
                                     num_epochs=1,
                                     batch_size=10,
                                     minibatch=None,
                                     fixed_loader=None,
                                     fedkd_helper: FedKDHelper = None,
                                     clnt_idx: int = None):
            if minibatch is None:
                num_data = min(len(self.train_data), self.cfg.max_sample)
            else:
                frac = min(1.0, minibatch)
                num_data = max(1, int(frac * len(self.train_data)))

            train_time = self.device.get_train_time(num_data, batch_size, num_epochs)
            logger.debug('client {}: num data:{}'.format(self.id, num_data))
            logger.debug('client {}: train time:{}'.format(self.id, train_time))
            """
            # compute num_data
            if minibatch is None:
                num_data = min(len(self.train_data["x"]), self.cfg.max_sample)
                xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
                data = {'x': xs, 'y': ys}
            else:
                frac = min(1.0, minibatch)
                num_data = max(1, int(frac*len(self.train_data["x"])))
                xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
                data = {'x': xs, 'y': ys}
            """
            data = self.dataloader
            download_time = self.device.get_download_time()
            upload_time = self.device.get_upload_time(self.model.size)  # will be re-calculated after training

            down_end_time = self.timer.get_future_time(start_t, download_time)
            logger.debug("client {} download-time-need={}, download-time-cost={} end at {}, "
                         .format(self.id, download_time, down_end_time - start_t, down_end_time))

            train_end_time = self.timer.get_future_time(down_end_time, train_time)
            logger.debug("client {} train-time-need={}, train-time-cost={} end at {}, "
                         .format(self.id, train_time, train_end_time - down_end_time, train_end_time))

            up_end_time = self.timer.get_future_time(train_end_time, upload_time)
            logger.debug("client {} upload-time-need={}, upload-time-cost={} end at {}, "
                         .format(self.id, upload_time, up_end_time - train_end_time, up_end_time))

            # total_cost = up_end_time - start_t
            # logger.debug("client {} task-time-need={}, task-time-cost={}"
            #             .format(self.id, download_time+train_time+upload_time, total_cost))

            self.ori_download_time = download_time  # original
            self.ori_train_time = train_time
            self.before_comp_upload_time = upload_time
            self.ori_upload_time = upload_time
            self.ori_total_cost = download_time + train_time + upload_time

            self.act_download_time = down_end_time - start_t  # actual
            self.act_train_time = train_end_time - down_end_time
            self.act_upload_time = up_end_time - train_end_time  # maybe decrease for the use of conpression algorithm

            self.update_size = self.model.size

            '''
            if not self.timer.check_comm_suc(start_t, download_time):
                self.actual_comp = 0.0
                download_available_time = self.timer.get_available_time(start_t, download_time)
                failed_reason = 'download interruption: download_time({}) > download_available_time({})'.format(download_time, download_available_time)
                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
            if train_time > train_time_limit:
                # data sampling
                comp = self.model.get_comp(data, num_epochs, batch_size)
                self.actual_comp = int(comp*available_time/train_time)    # will be used in get_actual_comp
                failed_reason = 'out of deadline: download_time({}) + train_time({}) + upload_time({}) > deadline({})'.format(download_time, train_time, upload_time, self.deadline)
                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
            elif train_time > available_time:
                # client interruption
                comp = self.model.get_comp(data, num_epochs, batch_size)
                self.actual_comp = int(comp*available_time/train_time)    # will be used in get_actual_comp
                failed_reason = 'client interruption: train_time({}) > available_time({})'.format(train_time, available_time)
                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
            if not self.timer.check_comm_suc(start_t + download_time + train_time, upload_time):
                comp = self.model.get_comp(data, num_epochs, batch_size)
                self.actual_comp = comp
                upload_available_time = self.timer.get_available_time(start_t + download_time + train_time, upload_time)
                failed_reason = 'upload interruption: upload_time({}) > upload_available_time({})'.format(upload_time, upload_available_time)
                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
            '''
            if (down_end_time - start_t) > self.deadline:
                # download too long
                self.actual_comp = 0.0
                self.update_size = 0
                failed_reason = f'failed when downloading\t{self.ori_total_cost}'
                self.cost_time = down_end_time
                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
            elif (train_end_time - start_t) > self.deadline:
                # failed when training
                train_time_limit = self.deadline - self.act_download_time
                if train_time_limit <= 0:
                    train_time_limit = 0.001
                available_time = self.timer.get_available_time(start_t + self.act_download_time, train_time_limit)
                comp = self.model.get_comp(data, num_epochs, batch_size)
                self.actual_comp = int(comp * available_time / train_time)  # will be used in get_actual_comp
                self.update_size = 0
                self.cost_time = train_end_time
                failed_reason = f'failed when training\t{self.ori_total_cost}'
                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
            elif (up_end_time - start_t) > self.deadline:
                self.actual_comp = self.model.get_comp(data, num_epochs, batch_size)
                if self.cfg.fedkd:
                    self.model.train(data, num_epochs)
                    # print(torch.cuda.memory_allocated())
                    self.lasttime = updatetimer
                    comp, update, acc, loss, grad, loss_old = -1, self.model.get_weights(), -1, self.model.trainloss, -1, -1
                else:
                    failed_reason = f'failed when uploading\t{self.ori_total_cost}'
                    self.cost_time = up_end_time
                    raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
            else:
                if minibatch is None:
                    if self.cfg.no_training:
                        comp = self.model.get_comp(data, num_epochs, batch_size)
                        update, acc, loss, grad, loss_old = -1, -1, -1, -1, -1
                    else:
                        self.model.train(data, num_epochs)
                        # print(torch.cuda.memory_allocated())
                        self.lasttime = updatetimer
                        comp, update, acc, loss, grad, loss_old = -1, self.model.get_weights(), -1, self.model.trainloss, -1, -1
                        # comp, update, acc, loss, grad, loss_old = self.model.train(data, num_epochs, batch_size)
                else:
                    # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
                    num_epochs = 1
                    if self.cfg.no_training:
                        comp = self.model.get_comp(data, num_epochs, num_data)
                        self.lasttime = updatetimer
                        update, acc, loss, grad, loss_old = -1, -1, -1, -1, -1
                    else:
                        # comp, update, acc, loss, grad, loss_old = self.model.train(data, num_epochs, batch_size)
                        self.model.train(data, self.id)
                        self.lasttime = updatetimer
                        comp, update, acc, loss, grad, loss_old = -1, self.model.get_weights(), -1, -1, -1, -1
                        # comp, update, acc, loss, grad, loss_old = self.model.train(data, num_epochs, batch_size)
            num_train_samples = len(data)
            simulate_time_c = train_time + upload_time
            self.actual_comp = comp

            # Federated Learning Strategies for Improving Communication Efficiency
            seed = None
            shape_old = None

            if self.cfg.virtual_momentum > 0:
                grad = self.model.get_grad()

            if self.cfg.fedkd:
                # local teacher update
                fedkd_helper.update_teacher_model(int(self.id), self.model.get_teacher_state())
                # share student update
                self.update_size = fedkd_helper.compress_student_model(
                    self.model.get_student_state(fedkd_helper.device))
                update, grad = None, None
                if fedkd_helper.is_svd:
                    # if compress with svd then compute upload_time
                    upload_time = self.device.get_upload_time(self.update_size)
                    self.ori_upload_time = upload_time
                    up_end_time = self.timer.get_future_time(train_end_time, upload_time)
                    self.act_upload_time = up_end_time - train_end_time

            total_cost = self.act_download_time + self.act_train_time + self.act_upload_time
            if total_cost > self.deadline:
                fedkd_helper.fail_release()
                # failed when uploading
                self.actual_comp = self.model.get_comp(data, num_epochs, batch_size)
                failed_reason = f'failed when uploading\t{self.ori_total_cost}'
                print(failed_reason + self.id)
                # Note that, to simplify, we did not change the update_size here, actually the actual update size is less.
                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)

            return total_cost, comp, num_train_samples, update, acc, loss, grad, self.update_size, seed, shape_old, loss_old

        return train_with_simulate_time(self, start_t, num_epochs, batch_size, minibatch, fixed_loader, distill_helper,
                                        clnt_idx)

    def train_wo_sim_time(self, start_t=None, num_epochs=1, batch_size=10, minibatch=None, fixed_loader=None,
                          distill_helper=None,
                          clnt_idx=None):
        if minibatch is None:
            num_data = min(len(self.train_data), self.cfg.max_sample)
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac * len(self.train_data)))

        train_time = self.device.get_train_time(num_data, batch_size, num_epochs)
        logger.debug('client {}: num data:{}'.format(self.id, num_data))
        logger.debug('client {}: train time:{}'.format(self.id, train_time))
        """
        # compute num_data
        if minibatch is None:
            num_data = min(len(self.train_data["x"]), self.cfg.max_sample)
            xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
            data = {'x': xs, 'y': ys}
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac*len(self.train_data["x"])))
            xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
            data = {'x': xs, 'y': ys}
        """
        data = self.dataloader
        self.update_size = self.model.size
        self.model.train(data, num_epochs)
        # print(torch.cuda.memory_allocated())
        self.lasttime = updatetimer
        comp, update, acc, loss, grad, loss_old = -1, self.model.get_weights(), -1, self.model.trainloss, -1, -1

        if self.cfg.virtual_momentum > 0:
            grad = self.model.get_grad()

        if self.cfg.fedkd:
            # local teacher update
            distill_helper.update_teacher_model(int(self.id), self.model.get_teacher_state())
            # share student update
            self.update_size = distill_helper.compress_student_model(
                self.model.get_student_state(distill_helper.device))
            update, grad = None, None
        return None, comp, None, update, acc, loss, grad, self.update_size, None, None, loss_old

    def test(self, set_to_use='test'):
        """Tests self.model on self.test_data.
        
        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data
        return self.model.test(data)

    def set_model_size(self, model_size):
        if hasattr(self.model, "size"):
            self.model.size = model_size
        self.device.model_size = model_size / 1024

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data)

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data)

        return train_size

        # @property

    # def model(self):
    #     """Returns this client reference to model being trained"""
    #     return self._model

    # @model.setter
    # def model(self, model):
    #     warnings.warn('The current implementation shares the model among all clients.'
    #                   'Setting it on one client will effectively modify all clients.')
    #     self._model = model

    def set_deadline(self, deadline=-1):
        if deadline < 0 or not (self.cfg.state_hete or self.cfg.device_hete):
            self.deadline = sys.maxsize
        else:
            self.deadline = deadline
        logger.debug('client {}\'s deadline is set to {}'.format(self.id, self.deadline))

    '''
    def set_upload_time(self, upload_time):
        if upload_time > 0:
            self.upload_time = upload_time
        else:
            logger.error('invalid upload time: {}'.format(upload_time))
            assert False
        logger.debug('client {}\'s upload_time is set to {}'.format(self.id, self.upload_time))
    
    def get_train_time_limit(self):
        if self.device != None:
            self.upload_time = self.device.get_upload_time()
            logger.debug('client {} upload time: {}'.format(self.id, self.upload_time))
        if self.upload_time < self.deadline :
            # logger.info('deadline: {}'.format(self.deadline))
            return self.deadline - self.upload_time
        else:
            return 0.01
    '''

    def upload_suc(self, start_t, num_epochs=1, batch_size=10, minibatch=None):
        """Test if this client will upload successfully

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
            start_t: strat time of the training, only used in train_with_simulate_time
        Return:
            result: test result(True or False)
        """
        if minibatch is None:
            num_data = min(len(self.train_data["x"]), self.cfg.max_sample)
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac * len(self.train_data["x"])))
        if self.device == None:
            download_time = 0.0
            upload_time = 0.0
        else:
            download_time = self.device.get_download_time()
            upload_time = self.device.get_upload_time()
        train_time = self.device.get_train_time(num_data, batch_size, num_epochs)
        train_time_limit = self.deadline - download_time - upload_time
        if train_time_limit < 0:
            train_time_limit = 0.001
        available_time = self.timer.get_available_time(start_t + download_time, train_time_limit)

        logger.debug('client {}: train time:{}'.format(self.id, train_time))
        logger.debug('client {}: available time:{}'.format(self.id, available_time))

        # compute num_data
        if minibatch is None:
            num_data = min(len(self.train_data["x"]), self.cfg.max_sample)
            xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
            data = {'x': xs, 'y': ys}
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac * len(self.train_data["x"])))
            xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
            data = {'x': xs, 'y': ys}

        if not self.timer.check_comm_suc(start_t, download_time):
            return False
        if train_time > train_time_limit:
            return False
        elif train_time > available_time:
            return False
        if not self.timer.check_comm_suc(start_t + download_time + train_time, upload_time):
            return False
        else:
            return True

    def get_device_model(self):
        if self.device == None:
            return 'None'
        return self.device.device_model

    def get_actual_comp(self):
        '''
        get the actual computation in the training process
        '''
        return self.actual_comp

    def __lt__(self, new):
        return int(self.id) < int(new.id)
