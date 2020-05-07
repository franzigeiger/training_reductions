import argparse
import logging
import sys

import fire
from numpy.random.mtrand import RandomState

from benchmark.run_benchmark import score_models as score_models_full
from benchmark.run_decoder_train_benchmark import score_models as score_models_train
from benchmark.run_single_layer_benchmark import score_models as score_models_single
from nets import test_models
from runtime.compression import measure_performance
from transformations import layer_based

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='DEBUG')
parser.add_argument('--model', type=str,
                    help='A model name')
parser.add_argument('--benchmark', type=str,
                    help='A benchmark name to execute')
parser.add_argument('--file_name', type=str,
                    help='A file to store results')
parser.add_argument('--batchnorm', type=bool,
                    help='Set if we apply changes also to batchnorm')
parser.add_argument('--pool', type=str, default='NET',
                    help='Pool to use: SINGLE|NET|TRAIN')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed to change random weights')
parser.add_argument('--epoch', type=float, default=6,
                    help='Number of epoch to test for')
parser.add_argument('--performance', type=bool, default=False,
                    help='disables all other flags and stored the models performance values in the database')
parser.add_argument('--batch_fix', type=bool, default=False,
                    help='disables all other flags and stored the models performance values in the database')
args, remaining_args = parser.parse_known_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level),
                    format='%(asctime)-15s %(levelname)s:%(name)s:%(message)s')
for disable_logger in ['s3transfer', 'botocore', 'boto3', 'urllib3', 'peewee', 'PIL']:
    logging.getLogger(disable_logger).setLevel(logging.WARNING)


def score_model_console():
    print('Start scoring model proces..')
    logger.info(f'Benchmarks configured:{args.benchmark}')
    logger.info(f'Models configured:{args.model}')
    test_models.seed = args.seed
    layer_based.random_state = RandomState(args.seed)
    logger.info(f'Run with seed {args.seed}')
    if args.performance:
        measure_performance(args.model)
    if args.pool == 'SINGLE':
        score_models_single(model=args.model, benchmark=args.benchmark, filename=args.file_name)
    elif args.pool == 'NET':
        score_models_full(model=args.model, benchmark=args.benchmark)
    elif args.pool == 'TRAIN':
        model = args.model
        if args.batch_fix:
            model = model + '_BF'

        if args.epoch % 1 == 0:
            model = model + f'_epoch_{int(args.epoch):02d}'
        else:
            model = model + f'_epoch_{args.epoch:.1f}'
        # model = model + f'_epoch_{args.epoch:02d}'  # if arg
        score_models_train(model=model, benchmark=args.benchmark)


logger.info(f"Running {' '.join(sys.argv)}")
fire.Fire(command='score_model_console')
