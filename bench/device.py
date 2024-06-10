# simulate device type
# current classify as big/middle/small device
# device can also be
import os

import numpy as np
import json
import random
import pickle as pkl

from typing import List
from utils.logger import Logger
from utils.device_util.device_util import Device_Util

# -1 - self define device, 0 - small, 1 - mid, 2 - big

L = Logger()
logger = L.get_logger()


class Device():
    du = Device_Util()
    speed_distri = None
    rng = np.random.default_rng(42)
    try:
        with open('speed_distri.json', 'r') as f:
            speed_distri = json.load(f)
    except FileNotFoundError as e:
        speed_distri = None
        logger.warn('no user\'s network speed trace was found, set all communication time to 0.0s')

    # support device type
    def __init__(self, cfg, model_size=0):
        self.device_model = None  # later set according to the trace
        self.cfg = cfg

        self.model_size = model_size / 1024  # change to kb because speed data use 'kb/s'
        if cfg.state_hete == False and cfg.device_hete == False:
            # make sure the no_trace mode perform the same as original leaf
            self.model_size = 0
        if Device.speed_distri == None:
            # treat as no_trace mode
            self.model_size = 0
            self.upload_speed_u = 1.0
            self.upload_speed_sigma = 0.0
            self.download_speed_u = 1.0
            self.download_speed_sigma = 0.0
        else:
            if cfg.device_hete == False:
                # assign a fixed speed distribution, the middle one
                # guid = list(Device.speed_distri.keys())[cfg.seed%len(Device.speed_distri)]
                keys = list(Device.speed_distri.keys())
                guid = \
                    sorted(keys,
                           key=lambda x: 1 / Device.speed_distri[x]['down_u'] + 1 / Device.speed_distri[x]['up_u'])[0]
                # logger.info(guid)
            else:
                guid = random.sample(list(Device.speed_distri.keys()), 1)[0]
            self.download_speed_u = Device.speed_distri[guid]['down_u']
            self.download_speed_sigma = Device.speed_distri[guid]['down_sigma']
            self.upload_speed_u = Device.speed_distri[guid]['up_u']
            self.upload_speed_sigma = Device.speed_distri[guid]['up_sigma']

        Device.du.set_model(cfg.model)
        Device.du.set_dataset("femnist")

    def set_device_model(self, real_device_model):
        self.device_model = Device.du.transfer(real_device_model)

    def _correct_time(self, model_size, speed):
        if self.cfg.correct:
            return (model_size * 4) / (speed / 8)
        else:
            return model_size / speed

    def get_upload_time(self, model_size):
        if self.cfg.state_random_t_cost:
            return 0
        if self.model_size == 0.0:
            return 0.0

        upload_speed = np.random.normal(self.upload_speed_u, self.upload_speed_sigma)
        while upload_speed < 0:
            upload_speed = np.random.normal(self.upload_speed_u, self.upload_speed_sigma)
        upload_time = self._correct_time(model_size, upload_speed) / 1000
        return upload_time

    def get_download_time(self):
        if self.cfg.state_random_t_cost:
            return self.rng.integers(0, self.cfg.state_speed_ddl)
        if self.model_size == 0.0:
            return 0.0

        download_speed = np.random.normal(self.download_speed_u, self.download_speed_sigma)
        while download_speed < 0:
            download_speed = np.random.normal(self.download_speed_u, self.download_speed_sigma)
        download_time = self._correct_time(self.model_size, download_speed)
        return download_time

    def get_train_time(self, num_sample, batch_size, num_epoch):
        if self.cfg.state_random_t_cost:
            return 0
        # TODO - finish train time predictor

        # current implementation: 
        # use real data withour prediction, 
        # so now it does not support other models
        if self.device_model == None:
            assert False
        return Device.du.get_train_time(self.device_model, num_sample, batch_size, num_epoch)


def filter_mobi_dev(devices) -> List[dict]:
    def is_match(speed_results):
        return len(speed_results) > 1 and 1000 < np.mean(speed_results) < 10000

    return [v for v in devices if is_match(v["tcp_speed_results"])]


class MobiPerfDevice(Device):
    try:
        with open("mobiperf_tcp_down_2018.json", "rb") as f:
            speed_trace = json.load(f)
            speed_trace = filter_mobi_dev(speed_trace)
    except FileNotFoundError as e:
        speed_trace = None
        logger.warn('no user\'s network speed trace was found, set all communication time to 0.0s')
    guids = set()
    pre_sampled = None
    idx = 0

    def __init__(self, cfg, model_size=0):
        super().__init__(cfg, model_size)
        if self.speed_trace is None:
            self.model_size = 0
            self.upload_speed_u = 1.0
            self.upload_speed_sigma = 0.0
            self.download_speed_u = 1.0
            self.download_speed_sigma = 0.0
        else:
            if MobiPerfDevice.pre_sampled is not None and isinstance(MobiPerfDevice.pre_sampled[0], dict):
                self.download_speed_traces = MobiPerfDevice.pre_sampled[MobiPerfDevice.idx]["tcp_speed_results"]
                self.download_speed_u = np.mean(self.download_speed_traces)
                self.upload_speed_traces = MobiPerfDevice.pre_sampled[MobiPerfDevice.idx]["tcp_speed_results"]
                self.upload_speed_u = np.mean(self.upload_speed_traces)
                MobiPerfDevice.idx += 1
            else:
                if MobiPerfDevice.pre_sampled is not None:
                    guid = MobiPerfDevice.pre_sampled[MobiPerfDevice.idx]
                    MobiPerfDevice.idx += 1
                else:
                    guid = random.sample(range(4), 1)[0]
                self.download_speed_traces = MobiPerfDevice.speed_trace[guid]["tcp_speed_results"]
                self.download_speed_traces = [v for v in self.download_speed_traces if v > 1000]
                self.download_speed_u = np.mean(self.download_speed_traces)
                self.upload_speed_traces = MobiPerfDevice.speed_trace[guid]["tcp_speed_results"]
                self.upload_speed_traces = [v for v in self.upload_speed_traces if v > 1000]
                self.upload_speed_u = np.mean(self.upload_speed_traces)

                # statics
                self.guids.add(guid)
            self.train_time = random.randint(0, 5)

    def get_train_time(self, num_sample, batch_size, num_epoch):
        if self.cfg.ignore_train_time:
            return 0
        else:
            return num_epoch * ((num_sample - 1) // batch_size + 1) * self.train_time / 40

    def get_upload_time(self, model_size):
        upload_speed = random.choice(self.upload_speed_traces)
        # upload_time = model_size / upload_speed / 1000
        upload_time = self._correct_time(model_size, upload_speed) / 1000
        return float(upload_time)

    def get_download_time(self):
        download_speed = random.choice(self.download_speed_traces)
        # download_time = self.model_size / download_speed
        download_time = self._correct_time(self.model_size, download_speed)
        return float(download_time)

    @staticmethod
    def random_sample(data, k, n, mu=4000):
        generator = np.random.default_rng(seed=42)

        def softmax(f):
            # return np.ones_like(f)
            if not isinstance(f, np.ndarray) or len(f) == 1:
                return [1]
                # instead: first shift the values of f so that the highest number is 0:
            f = - np.sqrt(f)  # f becomes [-666, -333, 0]
            try:
                return np.exp(f) / np.sum(np.exp(f))  # safe to do, gives the correct answer
            except Exception as e:
                raise EOFError()

        id_map = {str(np.mean(data[i]["tcp_speed_results"])): i for i in range(len(data))}

        data = [np.mean(data[i]["tcp_speed_results"]) for i in range(len(data))]

        _data = generator.choice(data, size=k)
        while np.max(_data) < mu or np.min(_data) > mu:
            _data = generator.choice(data, size=k)

        data = np.array(_data)

        x = generator.choice(data, 1)[0]
        samples = [x]
        appear_time = np.zeros_like(data)
        for i in range(1, n):
            if np.mean(samples) < mu:
                x = generator.choice(data[data > mu], 1, p=softmax(appear_time[data > mu]))[0]
                appear_time[data == x] += 1
            elif np.mean(samples) > mu:
                x = generator.choice(data[data < mu], 1, p=softmax(appear_time[data < mu]))[0]
                appear_time[data == x] += 1
            else:
                x = generator.choice(data, 1, p=softmax(appear_time))[0]
                appear_time[data == x] += 1
            samples.append(x)
        return [id_map[str(s)] for s in samples]


def create_device(cfg, model_size=0):
    return MobiPerfDevice(cfg, model_size)


def pre_sample(cfg, clients_num):
    if MobiPerfDevice.pre_sampled is None:
        if cfg.device_path is not None:
            if os.path.exists(cfg.device_path):
                with open(cfg.device_path, "r") as f:
                    MobiPerfDevice.pre_sampled = json.load(f)
                data = [np.mean(p["tcp_speed_results"]) for p in MobiPerfDevice.pre_sampled]
                logger.info("Pre-sampled mean={} std={}".format(np.mean(data), np.std(data)))
                return
            else:
                raise ValueError(f"Device path {cfg.device_path} does not exist")
        else:
            MobiPerfDevice.pre_sampled = MobiPerfDevice.random_sample(MobiPerfDevice.speed_trace, cfg.avail_device_num,
                                                                      clients_num)
        data = [np.mean(MobiPerfDevice.speed_trace[p]["tcp_speed_results"]) for p in MobiPerfDevice.pre_sampled]
        logger.info("Pre-sampled mean={} std={}".format(np.mean(data), np.std(data)))
