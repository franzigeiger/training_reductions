import argparse
import logging
import random
import sys

import fire
import numpy as np
import torch
from numpy.random.mtrand import RandomState

from nets import train_model, trainer
from nets.full_trainer import train as full_train
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
    if args.full:
        train_model(model=args.model, train_func=full_train)
    else:
        trainer.epochs = args.epoch
        logger.info(f'Train for {args.epoch} epochs')
        train_model(model=args.model)


logger.info(f"Running {' '.join(sys.argv)}")
fire.Fire(command='score_model_console')
