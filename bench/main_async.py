"""Script to run the baselines."""
import numpy as np
import random
import time
import traceback
from collections import defaultdict
import torch

# args
from utils.args import parse_args

# eventlet.monkey_patch()
args = parse_args()
config_name = args.config

# logger
from utils.logger import Logger

L = Logger()
L.set_log_name('train_records/train_record' + '_' + args.save_name + '_' + str(args.dataset_name) + '_' + str(
    int(args.deadline)) + "_s" + str(
    args.seed) + '.log')
logger = L.get_logger()

from client import Client
from server_async import AsyncServer

from utils.config import Config
from device import Device, create_device, pre_sample, MobiPerfDevice
from mimic_model import MimicModel

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

task = 9
classes = 2
per_client = 1
from mimic_data import NUM_CLASSES


def main():
    # read config from file
    cfg = Config(config_name, args)

    ## update configure file
    _update_config(cfg, args)
    # cfg.round_ddl[0] = args.deadline

    cfg.num_classes = NUM_CLASSES[cfg.dataset_name]

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(cfg.seed)
    torch.set_num_threads(3)

    '''
    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    sys_clients_per_round = args.sys_clients_per_round if args.sys_clients_per_round != -1 else tup[2]
    '''

    num_rounds = cfg.num_rounds
    eval_every = cfg.eval_every
    sys_clients_per_round = cfg.sys_clients_per_round

    # Suppress tf warnings
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Create 2 models

    # Create client model, and share params with server model
    # tf.reset_default_graph()
    client_model = createmodel(cfg)

    # Create clients
    logger.info('======================Setup Clients==========================')
    clients, valids, fixed_valids, tests = setup_clients(cfg, client_model)

    attended_clients = set()
    # Create server
    server = AsyncServer(client_model, clients=clients, cfg=cfg, valids=valids, fixed_loader=fixed_valids, tests=tests)

    # Initial status
    logger.info('===================== Random Initialization =====================')
    # print_stats(0, server, clients, client_num_samples, args, stat_writer_fn)

    # Simulate training
    if num_rounds == -1:
        import sys
        num_rounds = sys.maxsize

    if cfg.aggregation_strategy == "deadline":
        import sys
        total_training_time = cfg.round_ddl[0] * num_rounds
        num_rounds = sys.maxsize
        logger.info(f"****** Activate deadline-based modes: {total_training_time} ******")

    rounds_info = {}

    for i in range(num_rounds):
        if server.get_cur_time() >= total_training_time:
            server.my_eval(valids, createmodel(cfg))
            if cfg.test_curve and not cfg.target_acc:
                server.my_eval(tests, createmodel(cfg), load_best=False, is_test=True)
            server.my_eval(tests, createmodel(cfg), load_best=True, is_test=True)
            logger.info("Reach {} after {} round".format(server.get_cur_time(), i + 1))

            return
        logger.info('===================== Round {} of {} ====================='.format(i + 1, num_rounds))

        server._cur_round = i

        # 1. selection stage
        logger.info('--------------------- selection stage ---------------------')
        # 1.1 select clients
        cur_time = server.get_cur_time()
        time_window = server.get_time_window()
        logger.info('current time: {}\ttime window: {}\t'.format(cur_time, time_window))
        # In async mode, the sys_clients_per_round param is treated as the async buffer size. In sync, this is the number
        (clients_to_run, round_stragglers, virtual_client_clock, round_duration,
         flatten_client_duration) = server.tictak_client_tasks(sys_clients_per_round)

        # c_ids, c_groups, c_num_samples = server.get_clients_info(server.selected_clients)
        # attended_clients.update(c_ids)
        # c_ids.sort()
        # logger.info("selected num: {}".format(len(c_ids)))
        # logger.debug("selected client_ids: {}".format(c_ids))

        # 1.2 decide deadline for each client
        deadline = np.random.normal(cfg.round_ddl[0], cfg.round_ddl[1])
        while deadline <= 0:
            deadline = np.random.normal(cfg.round_ddl[0], cfg.round_ddl[1])
        deadline = int(deadline)

        # 1.3.1 show how many clients will upload successfully
        '''
        suc_c = 0
        for c in online_clients:
            c.set_deadline(deadline)
            if c.upload_suc(server.get_cur_time(), num_epochs=cfg.num_epochs, batch_size=cfg.batch_size, minibatch=cfg.minibatch):
                suc_c += 1
        logger.info('{} clients will upload successfully at most'.format(suc_c))
        '''

        # 2. configuration stage
        logger.info('--------------------- configuration stage ---------------------')
        # 2.1 train(no parallel implementation)

        sys_metrics = server.train_model(num_epochs=cfg.num_epochs, batch_size=cfg.batch_size, minibatch=cfg.minibatch,
                                         clients=clients_to_run, deadline=deadline, round_idx=i)

        # 2.2 update simulation time
        # server.pass_time(sys_metrics['configuration_time'])
        # already update in tictak_client_tasks func

        # 3. update stage
        logger.info('--------------------- report stage ---------------------')
        # 3.1 update global model
        if cfg.virtual_momentum > 0:
            server.update_with_staleness_momentum(cfg.update_frac)
            logger.info('round success by using avg momentum')
        else:
            server.update_with_staleness(cfg.update_frac)

        # 3.2 total simulation time for this round
        # logger.info("simulating round {} used {} seconds".format(i+1, time.time()-round_start_time))

        # 4. Test model(if necessary)
        if (i % eval_every == 0):
            server.my_eval(valids, createmodel(cfg))
            if cfg.test_curve and not cfg.target_acc:
                server.my_eval(tests, createmodel(cfg), load_best=False, is_test=True)
            if cfg.target_acc:
                cur_acc = server.my_eval(tests, createmodel(cfg), load_best=False, is_test=True)
                if cur_acc >= cfg.target_acc:
                    logger.info("Reach {} after {}".format(cur_acc, server.get_cur_time()))

                    return
            if cfg.max_rounds and i + 1 > cfg.max_rounds:
                logger.info("Fail! Reach {} after {}".format(cur_acc, server.get_cur_time()))
                return


def _update_config(cfg, args):
    """update configure file cfg with configure file args.
    """
    ## update all the configure files with manual input

    for k, v_ in vars(args).items():
        if k == 'config':
            continue
        if k == 'time_window':
            cfg.__setattr__(k, [v_, 0.0])
            continue
        if k == 'fedavgm' and args.biasprompt and not args.pcgrad:
            cfg.__setattr__(k, True)
            continue
        print('Updating configure file', k, v_)
        cfg.__setattr__(k, v_)


def online(clients, cur_time, time_window):
    # """We assume all users are always online."""
    # return online client according to client's timer
    online_clients = []
    for c in clients:
        try:
            if c.timer.ready(cur_time, time_window):
                online_clients.append(c)
        except Exception as e:
            traceback.print_exc()
    L = Logger()
    logger = L.get_logger()
    logger.info('{} of {} clients online'.format(len(online_clients), len(clients)))
    return online_clients


def createmodel(cfg) -> MimicModel:
    return MimicModel(cfg)


def create_clients(users, groups, train_data, model, cfg):
    L = Logger()
    logger = L.get_logger()
    sys_n_client = min(cfg.max_client_num, len(users))
    users = random.sample(users, sys_n_client)
    logger.info('Clients in Total: %d' % (len(users)))
    if len(groups) == 0:
        groups = [[] for _ in users]
    # clients = [Client(u, g, train_data[u], test_data[u], model, random.randint(0, 2), cfg) for u, g in zip(users, groups)]
    # clients = [Client(u, g, train_data[u], test_data[u], model, Device(random.randint(0, 2), cfg)) for u, g in zip(users, groups)]
    cnt = 0
    clients = []

    pre_sample(cfg, len(users))
    for u, g in zip(users, groups):
        c = Client(u, g, train_data[u], "dummy_model", create_device(cfg, model_size=model.size), cfg)
        if len(c.train_data) == 0:
            continue
        clients.append(c)
        cnt += 1
        if cnt % 500 == 0:
            logger.info('set up {} clients'.format(cnt))
    device_model = MobiPerfDevice
    if device_model is not None:
        logger.info('{}/{} total devices'.format(len(device_model.guids), cfg.avail_device_num))
        d = [c.device.download_speed_u for c in clients]
        d_m, d_s = np.mean(d), np.var(d)
        logger.info('Speed mean = {} \t Speed var = {}'.format(d_m, d_s))
    from timer import Timer
    Timer.save_cache()
    model2cnt = defaultdict(int)
    for c in clients:
        model2cnt[c.get_device_model()] += 1
    logger.info('device setup result: {}'.format(model2cnt))
    return clients


from mimic_data import get_dataloader


def setup_clients(cfg, model=None, use_val_set=False):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    # eval_set = 'test' if not use_val_set else 'val'
    # train_data_dir = os.path.join('..', 'data', cfg.dataset, 'data', 'train')
    # test_data_dir = os.path.join('..', 'data', cfg.dataset, 'data', eval_set)

    users, groups, train_data, valids, fixed_valids, tests = get_dataloader(cfg, task, per_client)

    clients = create_clients(users, groups, train_data, model, cfg)

    return clients, valids, fixed_valids, tests


if __name__ == '__main__':
    # nohup python main.py -dataset shakespeare -model stacked_lstm &
    start_time = time.time()
    main()
    # logger.info("used time = {}s".format(time.time() - start_time))
