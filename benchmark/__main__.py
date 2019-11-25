import argparse
import logging
import sys

import fire

from benchmark.run_benchmark import score_models

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
parser.add_argument('--model', type=str,
                    help='A model name')
parser.add_argument('--benchmark', type=str,
                    help='A benchmark name to execute')
parser.add_argument('--file_name', type=str,
                    help='A file to store results')
parser.add_argument('--batchnorm', type=bool,
                    help='Set if we apply changes also to batchnorm')
args, remaining_args = parser.parse_known_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level),
                    format='%(asctime)-15s %(levelname)s:%(name)s:%(message)s')
for disable_logger in ['s3transfer', 'botocore', 'boto3', 'urllib3', 'peewee', 'PIL']:
    logging.getLogger(disable_logger).setLevel(logging.WARNING)


def score_model_console():
    print('Start scoring model proces..')
    logger.info(f'Benchmarks configured:{args.benchmark}')
    logger.info(f'Models configured:{args.model}')
    score_models(model=args.model, benchmark= args.benchmark, filename=args.file_name)


logger.info(f"Running {' '.join(sys.argv)}")
fire.Fire(command='score_model_console')