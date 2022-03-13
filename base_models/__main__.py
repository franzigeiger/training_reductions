import argparse
import fire
import logging
import numpy as np
import random
import sys
import torch
from numpy.random.mtrand import RandomState

import base_models.trainer_images as image_train
from base_models import train_model, trainer, trainer_convergence, test_models, global_data, store_model, \
    train_full_on_version
from base_models.full_trainer import train as full_train
from base_models.trainer_convergence import train as conv_train
from base_models.trainer_first_epoch import train as train_first
from base_models.transfer_models import train_other
from transformations import layer_based

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
parser.add_argument('--model', type=str,
                    help='A model name')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed to change random weights.')
parser.add_argument('--epoch', type=int, default=20,
                    help='Number of epochs to train and test for.')
parser.add_argument('--full', type=bool, default=False,
                    help='Train all weights regardless of the configurations.')
parser.add_argument('--convergence', type=bool, default=False,
                    help='Run training until convergence, disables number of epochs.')
parser.add_argument('--convergence_2', type=bool, default=False,
                    help='Train until convergence with a more patient setting. Can be used when '
                         'training on less images and --images > 0.')
parser.add_argument('--save', type=bool, default=False,
                    help='Save only weights without training')
parser.add_argument('--full_continued', type=bool, default=False,
                    help='Continue full training on a certain checkpoint')
parser.add_argument('--source', type=str, default=False,
                    help='The checkpoint weight file name for continued training.')
parser.add_argument('--prune', type=bool, default=False,
                    help='Prune weights of the input model.')
parser.add_argument('--images', type=int, default=0,
                    help='Number of images to train on.')
parser.add_argument('--lr', type=float, default=0.0, help='Learning rate to start/continue learning with')
parser.add_argument('--optimizer', type=str, default='', help='Optimizer to train with')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay to train with')
parser.add_argument('--first', type=bool, default=False,
                    help='Set true to only train the first epoch and store weights on epoch '
                         'fractions.')
parser.add_argument('--step', type=int, default=0, help='Step size for learning rate scheduler')
parser.add_argument('--batch_fix', type=bool, default=False,
                    help='Disables all other flags and stored the models performance values in '
                         'the database')
parser.add_argument('--other', type=str, default='',
                    help='Train a model that is not CORnet-S but initialize it from the '
                         'configurations of the CORnet-S model under --name.')
parser.add_argument('--version', type=str, default='version',
                    help='Run another model with a mapping of of the specified version')

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
    random.seed(args.seed)
    logger.info(f'Run with seed {args.seed}')
    test_models.batch_fix = args.batch_fix
    if args.convergence_2:
        global_data.convergence_2 = True
    if args.seed != 0:
        global_data.seed = args.seed
    if args.first:
        train_model(model=args.model, train_func=train_first)
    elif args.full:
        train_model(model=args.model, train_func=full_train)
    elif args.full_continued:
        trainer_convergence.epochs = args.epoch
        train_full_on_version(model=args.model, source=args.source, train_func=conv_train)
    elif args.prune:
        trainer_convergence.epochs = args.epoch
        from base_models.pruner import train as train_prune
        train_model(model=args.model, train_func=train_prune)
    elif args.convergence:
        if args.lr != 0:
            trainer_convergence.lr = args.lr
        if args.optimizer != '':
            trainer_convergence.optimizer = args.optimizer
        if args.weight_decay != 0:
            trainer_convergence.weight_decay = args.weight_decay
        if args.step != 0:
            trainer_convergence.step_size = args.step
        if args.epoch != 20:
            trainer_convergence.epochs = args.epoch
        train_model(model=args.model, train_func=conv_train)
    elif args.save:
        train_model(model=args.model, train_func=store_model.train)
    elif args.other != '':
        if args.lr != 0:
            trainer_convergence.lr = args.lr
        if args.step != 0:
            trainer_convergence.step_size = args.step
        if args.epoch != 20:
            trainer_convergence.epochs = args.epoch
        trainer.epochs = args.epoch
        train_other(net=args.other, template=args.model, version=args.version, train_func=conv_train)
    else:
        if args.step != 0:
            trainer.step_size = args.step
        if args.lr != 0:
            trainer.lr = args.lr
            image_train.lr = args.lr
        trainer.epochs = args.epoch
        image_train.epochs = args.epoch
        logger.info(f'Train for {args.epoch} epochs')
        train_model(model=args.model, images=args.images)


logger.info(f"Running {' '.join(sys.argv)}")
fire.Fire(command='score_model_console')
