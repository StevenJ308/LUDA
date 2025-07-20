import argparse
import importlib
from utils import *

# MODEL_DIR = None
MODEL_DIR = r'D:\LwPK_m\checkpoint\cub200\luda\ft_cos-avg_cos-data_init-start_0\Epo_80-Bs_128-time_15-20-40' \
            r'\session0_max_acc.pth '
DATA_DIR = 'data/'
# PROJECT = 'luda'
PROJECT = 'cec'


def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    # about
    parser.add_argument('-epochs_base', type=int, default=80)
    parser.add_argument('-lr_neg', type=float, default=0.01)  # cifar 0.1   cub 0.005 imagenet 0.1
    parser.add_argument('-lr_base', type=float, default=0.005)  # cifar 0.1   cub 0.005 imagenet 0.1
    parser.add_argument('-schedule', type=str, default='Milestone',
                        choices=['Step', 'Milestone', 'Cosine'])  # cifar  COSINE    CUB  MILESTONE  imagenet COSINE
    parser.add_argument('-milestones', nargs='+', type=int, default=[40, 80, 120])
    parser.add_argument('-step', type=int, default=20)
    parser.add_argument('-decay', type=float, default=0.0005)  # ViT 0.3 ResNet 0.0005
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)  # CIFAR 0.1  cub 0.25  imagenet 0.1
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')
    parser.add_argument('-batch_size_base', type=int, default=32)
    parser.add_argument('-batch_size_new', type=int, default=0,
                        help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos'])  # ft_dot means using linear classifier, ft_cos means using cosine
    # classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos'])

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-num_workers', type=int, default=0)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')

    # about manifold mixup
    parser.add_argument('-balance', type=float, default=0.01)
    parser.add_argument('-loss_iter', type=int, default=10)
    parser.add_argument('-alpha', type=float, default=2.0)
    parser.add_argument('-eta', type=float, default=0.1)

    # about LwPK
    parser.add_argument('-use_margin', type=bool, default=True)
    parser.add_argument('-neg_margin', type=float, default=0.15)
    parser.add_argument('-pos_margin', type=float, default=-0.1)
    parser.add_argument('-n_epochs', type=int, default=150)
    parser.add_argument('-lr_scale', type=float, default=0.5)
    parser.add_argument('-use_ba_scale', type=bool, default=False)
    parser.add_argument('-use_ng_scale', type=bool, default=False)

    # for episode learning
    parser.add_argument('-train_episode', type=int, default=50)
    parser.add_argument('-episode_shot', type=int, default=1)
    parser.add_argument('-episode_way', type=int, default=15)
    parser.add_argument('-episode_query', type=int, default=15)

    # for cec
    parser.add_argument('-lrg', type=float, default=0.1)  # lr for graph attention network
    parser.add_argument('-low_shot', type=int, default=1)
    parser.add_argument('-low_way', type=int, default=8)

    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()