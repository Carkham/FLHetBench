import os
from .logger import Logger
import traceback
import sys
import yaml

L = Logger()
logger = L.get_logger()

DEFAULT_CONFIG_FILE = 'default.cfg'


# configuration for FedAvg
def load_prompt_config(path=None):
    if path is not None:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return data
    else:
        return None


class Config:

    def __init__(self, config_file='default.cfg', args=None):
        self.aggregation_strategy = None
        self.config_name = config_file
        self.data_path = './data/'
        self.dataset_name = 'COVID'
        self.model = 'vit'

        # Deadline-based and Readiness-based strategy
        self.num_rounds = -1  # -1 for unlimited
        self.max_rounds = None
        self.eval_every = 3  # -1 for eval when quit
        self.sys_clients_per_round = 10
        self.sys_n_client = None
        self.min_selected = 1
        self.max_sample = 100  # max sample num for training in a round

        # client
        self.client_lr = 0.1
        self.client_bs = 10
        self.client_n_epoch = 1

        self.seed = 0
        self.minibatch = None  # always None for FedAvg
        self.round_ddl = [100, 0]
        self.update_frac = 0.05
        self.max_client_num = 1000  # total client num, -1 for unlimited
        self.upload_time = [10.0, 1.0]  # now its no use in config
        '''
        # speed is no more used and replaced by training time provided by device_util
        self.big_speed = [150.0, 1.0]
        self.mid_speed = [100.0, 1.0]
        self.small_speed = [50.0, 1.0]  
        '''
        self.time_window = [
            20.0, 0.0
        ]  # time window for selection stage ## changed to 30 to check
        self.user_trace = False
        self.state_hete = False
        self.device_hete = False
        self.no_training = False
        self.real_world = False
        self.correct = True
        # grad_compress,  structure_k, fedprox and qffl are mutually-exclusive
        self.n_gpu = 1
        self.gpu_id = "cuda:0"
        self.num_classes = 2  ## default to 2 but changed to others according to the datasets names

        # fedprox
        self.fedprox = False
        self.fedprox_mu = 0

        # fedkd
        self.fedkd = False
        self.tmin = 0.95
        self.tmax = 0.98

        # biasprompt
        self.biasprompt = False
        self.positions = [[1, 12]]
        self.prompt_length = 50
        self.use_prefix_tune = False

        # fetchsgd
        self.fetchsgd = False
        self.k = 50000
        self.num_cols = 500000
        self.num_rows = 5
        self.num_blocks = 20

        # sever momentum
        self.virtual_momentum = 0.

        # FedDyn
        self.feddyn = False
        self.feddyn_alpha = 1e-2

        # async FL and FedBUFF
        self.max_staleness = 5
        self.max_concurrency = 100
        # self.async_buffer = 10

        # TimelyFL
        self.timelyfl = False
        self.async_buffer = 10

        logger.info('read config from {}'.format(config_file))
        self.read_config(config_file, args)
        self.log_config()

    def read_config(self, filename=DEFAULT_CONFIG_FILE, args=None):
        if not os.path.exists(filename):
            logger.error(
                'ERROR: config file {} does not exist!'.format(filename))
            assert False
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                try:
                    line = line.strip().split()
                    # System
                    if line[0] == 'sys_clients_per_round':
                        self.sys_clients_per_round = int(line[1])
                    elif line[0] == 'sys_n_client':
                        self.sys_n_client = int(line[1])

                    # Client
                    elif line[0] == 'client_lr':
                        self.client_lr = float(line[1])
                    elif line[0] == 'client_bs':
                        self.client_bs = int(line[1])
                    elif line[0] == 'client_n_epoch':
                        self.client_n_epoch = int(line[1])

                    elif line[0] == 'num_rounds':
                        self.num_rounds = int(line[1])
                    elif line[0] == 'eval_every':
                        self.eval_every = int(line[1])
                    elif line[0] == 'max_client_num':
                        self.max_client_num = int(line[1])
                        if self.max_client_num < 0:
                            self.max_client_num = sys.maxsize
                    elif line[0] == 'min_selected':
                        self.min_selected = int(line[1])
                    elif line[0] == 'n_gpu':
                        self.n_gpu = int(line[1])
                    elif line[0] == 'gpu_id':
                        self.gpu_id = str(line[1])
                    elif line[0] == 'seed':
                        self.seed = int(line[1])
                    elif line[0] == 'model':
                        self.model = str(line[1])
                    elif line[0] == 'round_ddl':
                        self.round_ddl = [float(line[1]), float(line[2])]
                    elif line[0] == 'update_frac':
                        self.update_frac = float(line[1])
                    elif line[0] == 'upload_time':
                        self.upload_time = [float(line[1]), float(line[2])]
                    elif line[0] == 'time_window':
                        self.time_window = [float(line[1]), float(line[2])]
                    elif line[0] == 'state_hete':
                        self.state_hete = line[1].strip() == 'True'
                        if not self.state_hete:
                            logger.info(
                                'no behavior heterogeneity! assume client is availiable at any time.'
                            )
                    elif line[0] == 'device_hete':
                        self.device_hete = line[1].strip() == 'True'
                        if not self.device_hete:
                            logger.info(
                                'no hardware heterogeneity! assume all clients are same.'
                            )
                    elif line[0] == 'no_training':
                        self.no_training = line[1].strip() == 'True'
                        if self.no_training:
                            logger.info('no actual training process')
                    elif line[0] == 'realworld':
                        self.real_world = line[1].strip() == 'True'
                    elif line[0] == 'max_sample':
                        self.max_sample = int(line[1])
                    elif line[0] == 'fedprox':
                        self.fedprox = line[1].strip() == 'True'
                    elif line[0] == 'fedprox_mu':
                        self.fedprox_mu = float(line[1].strip())
                    elif line[0] == 'user_trace':
                        # to be compatibale with old version
                        self.user_trace = line[1].strip() == 'True'

                    # added ourself
                    elif line[0] == 'data_path':
                        self.data_path = str(line[1])
                    elif line[0] == 'dataset_name':
                        self.dataset_name = str(line[1])
                    elif line[0] == 'save_name':
                        self.save_name = str(line[1])
                    elif line[0] == 'sub_dataset_size':
                        self.sub_dataset_size = int(line[1])

                    # biasprompt
                    elif line[0] == 'biasprompt':
                        self.biasprompt = line[1].strip() == 'True'
                    elif line[0] == 'positions':
                        self.positions = eval(line[1].strip())
                    elif line[0] == 'prompt_length':
                        self.prompt_length = int(line[1].strip())

                    # fedkd
                    elif line[0] == 'fedkd':
                        self.fedkd = line[1].strip() == 'True'
                    elif line[0] == 't_min':
                        self.tmin = float(line[1])
                    elif line[0] == 't_max':
                        self.tmax = float(line[1])

                    # fetchsgd
                    elif line[0] == 'fetchsgd':
                        self.fetchsgd = line[1].strip() == 'True'
                    elif line[0] == 'num_rows':
                        self.num_rows = int(line[1])
                    elif line[0] == 'k':
                        self.k = int(line[1])
                    elif line[0] == 'num_cols':
                        self.num_cols = int(line[1])
                    elif line[0] == 'num_blocks':
                        self.num_blocks = int(line[1])

                    # server momentum
                    elif line[0] == 'virtual_momentum':
                        self.virtual_momentum = float(line[1])

                    # feddyn
                    elif line[0] == 'feddyn':
                        self.feddyn = line[1].strip() == 'True'
                    elif line[0] == 'feddyn_alpha':
                        self.feddyn_alpha = float(line[1])

                    # timelyfl
                    elif line[0] == 'timelyfl':
                        self.timelyfl = line[1].strip() == 'True'

                except Exception as e:
                    traceback.print_exc()
        if self.real_world and 'realworld' not in self.dataset:
            logger.error(
                '\'real_world\' is valid only when dataset is set to \'realworld\', current dataset {}'
                .format(self.dataset))
            self.real_world = False
        if self.user_trace == True:
            self.device_hete = True
            self.state_hete = True
        self.state_path = args.state_path
        self.device_path = args.device_path
        # print("Manual input args: ")
        # print("Deadline: " + str(args.deadline))
        # print("Rounds " + str(args.rounds))
        # print("Seed " + str(args.seed))
        # print("GPU " + args.gpu)
        # print("Val set " + str(args.val))
        # print("dataset_name " + str(args.dataset_name))
        # print("data_aug_phase " + str(args.data_aug_phase))
        # print("naive_meta_learning " + str(args.naive_meta_learning))
        self.gpu_id = args.gpu
        self.seed = args.seed

        self.aggregation_strategy = args.aggregation_strategy
        self.num_rounds = args.num_rounds
        self.round_ddl[0] = args.deadline

        self.target_acc = args.target_acc
        self.drop_rate = args.drop_rate
        if (self.target_acc or self.drop_rate) and self.aggregation_strategy != "readiness":
            raise ValueError("--target_acc/--drop_rate must be set with readiness-based strategy")
        if self.aggregation_strategy == "readiness":
            self.round_ddl[0] = sys.maxsize

        if args.prompt_length is not None:
            self.prompt_length = args.prompt_length
        if self.biasprompt or self.fetchsgd:
            self.virtual_momentum = 0.9

        self.model = args.sys_model

    def log_config(self):
        configs = vars(self)
        logger.info('================= Config =================')
        for key in configs.keys():
            logger.info('\t{} = {}'.format(key, configs[key]))
        logger.info('================= ====== =================')
