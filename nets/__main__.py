import argparse
import logging
import random
import sys

import fire
import numpy as np
import torch
from numpy.random.mtrand import RandomState

from nets import train_model, trainer, trainer_convergence, test_models
from nets.full_trainer import train as full_train
from nets.trainer_convergence import train as conv_train
from nets.trainer_first_epoch import train as train_first
from transformations import layer_based

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
parser.add_argument('--model', type=str,
                    help='A model name')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed to change random weights')
parser.add_argument('--epoch', type=int, default=20,
                    help='Number of epochs to train and test for')
parser.add_argument('--full', type=bool, default=False,
                    help='Number of epochs to train and test for')
parser.add_argument('--convergence', type=bool, default=False,
                    help='Number of epochs to train and test for')
parser.add_argument('--lr', type=float, default=0.0, help='Learning rate to start/continue learning with')
parser.add_argument('--first', type=bool, default=False,
                    help='Set true to only train the first epoch and evaluate fractions')
parser.add_argument('--step', type=int, default=0, help='Step size for learning rate scheduler')
parser.add_argument('--batch_fix', type=bool, default=False,
                    help='disables all other flags and stored the models performance values in the database')

args, remaining_args = parser.parse_known_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level),
                    format='%(asctime)-15s %(levelname)s:%(name)s:%(message)s')
for disable_logger in ['s3transfer', 'botocore', 'boto3', 'urllib3', 'peewee', 'PIL']:
    logging.getLogger(disable_logger).setLevel(logging.WARNING)


def score_model_console():
    print('Start scoring model proces..')
    logger.info(f'Models configured:{args.model}')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    layer_based.random_state = RandomState(args.seed)
    random.seed(0)
    logger.info(f'Run with seed {args.seed}')
    test_models.batch_fix = args.batch_fix
    if args.first:
        train_model(model=args.model, train_func=train_first)
    if args.full:
        train_model(model=args.model, train_func=full_train)
    if args.convergence:
        if args.lr != 0:
            trainer_convergence.lr = args.lr
        if args.step != 0:
            trainer_convergence.step_size = args.step
        if args.epoch != 20:
            trainer_convergence.epochs = args.epoch
        train_model(model=args.model, train_func=conv_train)
    else:
        if args.step != 0:
            trainer.step_size = args.step
        if args.lr != 0:
            trainer.lr = args.lr
        trainer.epochs = args.epoch
        logger.info(f'Train for {args.epoch} epochs')
        train_model(model=args.model)


logger.info(f"Running {' '.join(sys.argv)}")
fire.Fire(command='score_model_console')

# if __name__ == '__main__':
#     train_model('CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_full')
