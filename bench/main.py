"""Script to run the baselines."""
import os
import random
import time
import traceback
import torch
import numpy as np

from collections import defaultdict
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
from server import Server

from utils.config import Config
from device import Device, create_device, pre_sample, MobiPerfDevice
from mimic_model import MimicModel
from mimic_data import get_dataloader, NUM_CLASSES

task = 9
classes = 2
per_client = 1


def main():
    # read config from file
    cfg = Config(config_name, args)

    assert cfg.model in ["vit", "resnet"]
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
    client_model = create_model(cfg)

    # Create clients
    logger.info('======================Setup Clients==========================')
    clients, valids, fixed_valids, tests = setup_clients(cfg, client_model)
    # fixed_val_data = next(iter(fixed_valids))
    # fixed_val_datas, fixed_val_labels = tuple(v.to(torch.device(cfg.gpu_id)) for v in fixed_val_data)

    # print(sorted([c.num_train_samples for c in clients]))

    attended_clients = set()
    # Create server
    server = Server(client_model, clients=clients, cfg=cfg, valids=valids, fixed_loader=fixed_valids, tests=tests)

    client_ids, client_groups, client_num_samples = server.get_clients_info(clients)

    # Initial status
    logger.info('===================== Random Initialization =====================')

    # Simulate training
    if num_rounds == -1:
        import sys
        num_rounds = sys.maxsize

    for i in range(num_rounds):

        ## just continue the training
        # if i < 3700:
        #     continue
        # round_start_time = time.time()
        logger.info('===================== Round {} of {} ====================='.format(i + 1, num_rounds))

        server._cur_round = i

        # 1. selection stage
        logger.info('--------------------- selection stage ---------------------')
        # 1.1 select clients
        cur_time = server.get_cur_time()
        time_window = server.get_time_window()
        logger.info('current time: {}\ttime window: {}\t'.format(cur_time, time_window))
        online_clients = online(clients, cur_time, time_window)
        server.select_clients(i, online_clients, num_clients=sys_clients_per_round)
        if len(server.selected_clients) < server.cfg.min_selected:
            # insufficient clients to select, round failed
            logger.info('round failed in selection stage!')
            server.pass_time(time_window)
            if i % eval_every == 0:
                server.my_eval(valids, create_model(cfg))
                if cfg.target_acc:
                    cur_acc = server.my_eval(tests, create_model(cfg), load_best=False, is_test=True)
                    if cur_acc >= cfg.target_acc:
                        logger.info("Reach {} after {}".format(cur_acc, server.get_cur_time()))
                        return
                    if cfg.max_rounds and i + 1 > cfg.max_rounds:
                        logger.info("Fail! Reach {} after {}".format(cur_acc, server.get_cur_time()))
                        return
            continue

        c_ids, c_groups, c_num_samples = server.get_clients_info(server.selected_clients)
        attended_clients.update(c_ids)
        c_ids.sort()
        logger.info("selected num: {}".format(len(c_ids)))
        logger.debug("selected client_ids: {}".format(c_ids))

        # 1.2 decide deadline for each client
        deadline = np.random.normal(cfg.round_ddl[0], cfg.round_ddl[1])
        while deadline <= 0:
            deadline = np.random.normal(cfg.round_ddl[0], cfg.round_ddl[1])
        deadline = int(deadline)
        # if cfg.state_hete:
        if True:
            logger.info('selected deadline: {}'.format(deadline))

        # 1.3 update simulation time
        server.pass_time(time_window)

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

        sys_metrics = server.train_model(num_epochs=cfg.client_n_epoch, batch_size=cfg.client_bs,
                                         minibatch=cfg.minibatch, deadline=deadline, round_idx=i)

        # 2.2 update simulation time
        server.pass_time(sys_metrics['configuration_time'])

        # 3. update stage
        logger.info('--------------------- report stage ---------------------')
        # 3.1 update global model
        if cfg.fetchsgd:
            server.update_using_fetchsgd(cfg.update_frac)
            logger.info('round success by using fetchsgd')
        elif cfg.virtual_momentum > 0:
            server.update_using_avg_momentum(cfg.update_frac)
            logger.info('round success by using avg momentum')
        elif cfg.fedkd:
            server.update_using_fedkd(cfg.update_frac)
            logger.info('round success by using FedKD')
        elif cfg.feddyn:
            server.update_using_feddyn(cfg.update_frac)
            logger.info('round success by using FedDyn')
        elif cfg.timelyfl:
            server.update_using_timelyfl(cfg.update_frac)
            logger.info('round success by using timelyfl')
        else:
            server.update(cfg.update_frac)

        # 3.2 total simulation time for this round
        # logger.info("simulating round {} used {} seconds".format(i+1, time.time()-round_start_time))

        # 4. Test model(if necessary)
        if i % eval_every == 0:
            server.my_eval(valids, create_model(cfg))
            if cfg.target_acc:
                cur_acc = server.my_eval(tests, create_model(cfg), load_best=False, is_test=True)
                if cur_acc >= cfg.target_acc:
                    logger.info("Reach {} after {}".format(cur_acc, server.get_cur_time()))
                    return
                if cfg.max_rounds and i + 1 > cfg.max_rounds:
                    logger.info("Fail! Reach {} after {}".format(cur_acc, server.get_cur_time()))
                    return
                    #
        """
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            if cfg.no_training:
                continue
            logger.info('--------------------- test result ---------------------')
            logger.info('attended_clients num: {}/{}'.format(len(attended_clients), len(clients)))
            # logger.info('attended_clients: {}'.format(attended_clients))
            # test_num = min(len(clients), 100)
            test_num = len(clients)
            if (i + 1) % (10*eval_every) == 0 or (i + 1) == num_rounds:
                test_num = len(clients)
                with open('attended_clients_{}.json'.format(config_name), 'w') as fp:
                    json.dump(list(attended_clients), fp)
                    logger.info('save attended_clients.json')
                
                # Save server model
                ckpt_path = os.path.join('checkpoints', cfg.dataset)
                if not os.path.exists(ckpt_path):
                    os.makedirs(ckpt_path)
                save_path = server.save_model(os.path.join(ckpt_path, '{}_{}.ckpt'.format(cfg.model, cfg.config_name)))
                logger.info('Model saved in path: %s' % save_path)
                
            test_clients = random.sample(clients, test_num) 
            sc_ids, sc_groups, sc_num_samples = server.get_clients_info(test_clients)
            logger.info('number of clients for test: {} of {} '.format(len(test_clients),len(clients)))
            another_stat_writer_fn = get_stat_writer_function(sc_ids, sc_groups, sc_num_samples, args)
            # print_stats(i + 1, server, test_clients, client_num_samples, args, stat_writer_fn)
            print_stats(i, server, test_clients, sc_num_samples, args, another_stat_writer_fn)
            
            if (i + 1) % (10*eval_every) == 0 or (i + 1) == num_rounds:
                
        """
    # Close models
    server.my_eval(valids, create_model(cfg))
    server.my_eval(tests, create_model(cfg), load_best=True, is_test=True)
    # server.close_model()


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


def create_model(cfg):
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
        logger.info('{}/{} total devices'.format(len(set(device_model.guids)), len(device_model.guids)))
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


def setup_clients(cfg, model=None):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """

    users, groups, train_data, valids, fixed_valids, tests = get_dataloader(dataset_name=cfg.dataset_name,
                                                                            data_path=cfg.data_path,
                                                                            n_client=cfg.sys_n_client,
                                                                            iid=cfg.iid,
                                                                            sub_dataset_size=cfg.sub_dataset_size,
                                                                            beta=cfg.non_iid_beta,
                                                                            seed=cfg.seed)

    clients = create_clients(users=users, groups=groups, train_data=train_data, model=model, cfg=cfg)

    return clients, valids, fixed_valids, tests


if __name__ == '__main__':
    # nohup python main.py -dataset shakespeare -model stacked_lstm &
    start_time = time.time()
    main()
    # logger.info("used time = {}s".format(time.time() - start_time))
