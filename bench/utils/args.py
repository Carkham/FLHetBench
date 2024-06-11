import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config',
                        help='path to config file;',
                        type=str,
                        # required=True,
                        default='default.cfg')

    # move to config
    '''
    parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    # required=True,
                    default='shakespeare')
    parser.add_argument('-model',
                    help='name of model;',
                    type=str,
                    # required=True,
                    default='stacked_lstm')
    parser.add_argument('--num-rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=-1)
    parser.add_argument('--eval-every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=-1)
    parser.add_argument('--clients-per-round',
                    help='number of clients trained per round;',
                    type=int,
                    default=-1)
    parser.add_argument('--batch-size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    parser.add_argument('--seed',
                    help='seed for random client sampling and batch splitting',
                    type=int,
                    default=0)
    '''
    parser.add_argument('--save_name',
                        help='The name used to save the model and others;',
                        type=str,
                        default='FedAVG')
    parser.add_argument('--gpu',
                        type=str,
                        default="cuda:0")
    parser.add_argument('--seed',
                        help='seed for random client sampling and batch splitting',
                        type=int,
                        default=1234)
    parser.add_argument('--cpu_updates',
                        action="store_true",
                        default=False,
                        help="Set true for saving GPU memory")

    # Minibatch doesn't support num_epochs, so make them mutually exclusive
    epoch_capability_group = parser.add_mutually_exclusive_group()
    # num epochs will be determined by config file
    epoch_capability_group.add_argument('--num-epochs',
                                        help='number of epochs when clients train on data;',
                                        type=int,
                                        default=1)

    # move to config
    '''
    parser.add_argument('-lr',
                    help='learning rate for local optimizers;',
                    type=float,
                    default=-1,
                    required=False)
    parser.add_argument("-gpu_fraction", 
                        help="per process gpu memory fraction", 
                        type=float,
                        default=0.2,
                        required=False)
    '''
    parser.add_argument('--aggregation_strategy',
                        type=str,
                        choices=["deadline", "readiness"],
                        default="deadline",
                        help="Deadline-based or Readiness-based aggregation strategy"
                        )

    # Readiness-based strategy
    parser.add_argument('--target_acc',
                        type=float,
                        default=None,
                        help="Use to record tart performance time")
    parser.add_argument('--drop_rate',
                        type=float,
                        default=0.,
                        help="Fraction of how many stragglers to drop")
    parser.add_argument('--max_rounds',
                        type=int,
                        default=10000,
                        help="Max communication rounds for readiness-based strategy")

    # Deadline-based strategy
    parser.add_argument('--num_rounds',
                        type=int,
                        default=10000,
                        help="Total communication rounds")
    parser.add_argument('--deadline',
                        type=float,
                        default=30,
                        help="Round deadline")

    # Heterogeneity parameters
    parser.add_argument('--state_path',
                        type=str,
                        default=None,
                        help="Heterogeneous state data path")
    parser.add_argument('--device_path',
                        type=str,
                        default=None,
                        help="Heterogeneous device data path")
    parser.add_argument('--ignore_train_time',
                        action='store_true',
                        default=False,
                        help="Whether to ignore the train time")

    # Dataset parameters
    parser.add_argument('--dataset_name',
                        type=str,
                        # required=True,
                        default='COVID',
                        help='name of dataset;',)
    parser.add_argument('--data_path',
                        type=str,
                        default='data/',
                        help='path of dataset;',)
    parser.add_argument('--iid',
                        action="store_true",
                        default=False,
                        help="whether sampled with iid distribution")
    parser.add_argument('--non_iid_beta',
                        type=float,
                        default=0.6,
                        help="parameter beta for non-iid sampling with dirichlet distribution")
    parser.add_argument('--sub_dataset_size',
                        type=int,
                        default=None,
                        help="subset dataset size")

    # System parameters
    parser.add_argument('--sys-n_client', type=int, default=100, help="if is not None data is split as non-iid")
    parser.add_argument('--sys-clients_per_round', type=int, default=20, help="If not none, replace the config's one")
    parser.add_argument('--sys-model', type=str, default='vit', help='Model name')

    # Client parameters
    parser.add_argument('--client-lr', type=float, default=0.001, help='Learning rate in clients')
    parser.add_argument('--client-bs', type=int, default=16, help='Batch size in clients')
    parser.add_argument('--client-n_epoch', type=int, default=1, help='Number of local training epochs in clients')

    # fedprox
    parser.add_argument("--fedprox",
                        action='store_true',
                        default=False,
                        help="Use of FedProx or not.")
    parser.add_argument('--fedprox_mu',
                        type=float,
                        default=0.1)

    # fedkd
    parser.add_argument('--fedkd',
                        action="store_true",
                        default=False)

    # feddyn
    parser.add_argument('--feddyn',
                        action="store_true",
                        default=False)

    # timelyfl
    parser.add_argument('--timelyfl',
                        action="store_true",
                        default=False)

    # fetchsgd
    parser.add_argument('--fetchsgd',
                        action='store_true',
                        default=False,
                        help="Whether aggregate with FetchSGD")

    # BiasPrompt
    parser.add_argument('--biasprompt',
                        action="store_true",
                        default=False)
    parser.add_argument('--prompt_length',
                        type=int,
                        default=50,
                        help="prompt_length")

    # Server momentum
    parser.add_argument('--virtual_momentum',
                        action="store_true",
                        default=False)

    parser.add_argument('--sync_staleness',
                        action="store_true",
                        default=False)

    parser.add_argument('--stale_aggregator',
                        choices=["seq", "sum", "naive"],
                        default="seq")

    parser.add_argument('--time_window',
                        type=float,
                        default=20.0)

    # Async FL
    parser.add_argument('--max_staleness',
                        type=int,
                        default=5)
    return parser.parse_args()
