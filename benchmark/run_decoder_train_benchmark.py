import datetime
import logging
import os
import sys

from brainscore import score_model

from benchmark.database import create_connection, store_score
from nets.test_models import cornet_s_brainmodel

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName('DEBUG'),
                    format='%(asctime)-15s %(levelname)s:%(name)s:%(message)s')
for disable_logger in ['s3transfer', 'botocore', 'boto3', 'urllib3', 'peewee', 'PIL']:
    logging.getLogger(disable_logger).setLevel(logging.WARNING)


def run_benchmark(benchmark_identifier, model_name):
    print(f'>>>>>Start running model {model_name} on benchmark {benchmark_identifier}')
    # model = brain_translated_pool[model_name]
    model = cornet_s_brainmodel(model_name, True)
    score = score_model(model_identifier=model.identifier, model=model, benchmark_identifier=benchmark_identifier)
    return score, model.identifier


def score_models(model, benchmark):
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    if 'public' in benchmarks:
        path = f'{dir_path}/../scores.sqlite'
    else:
        path = f'{dir_path}/../scores.sqlite'
    db = create_connection(path)
    os.environ["RESULTCACHING_DISABLE"] = "model_tools,candidate_models,brainscore.score_model"
    base_model = model.split('_')[0]
    try:
        d = datetime.datetime.now()
        raw_score, model = run_benchmark(benchmark, model)
        print(raw_score)
        result = raw_score.sel(aggregation='center')
        result = result.values
        store_score(db, (base_model, benchmark, d, result.item(0), model, False))
    except Exception as e:
        logging.error(f'Could not run model {model} because of following error')
        logging.error(e, exc_info=True)
    finally:
        db.close()
        d = datetime.datetime.today()
        logger.info(f'\nJob finished at {d.isoformat()}\n')


if __name__ == '__main__':
    d = datetime.datetime.today()
    filename = f'results_{d.isoformat()}.txt'
    benchmarks = [
        'movshon.FreemanZiemba2013.V1-pls',
        # 'movshon.FreemanZiemba2013.V2-pls',
        # 'dicarlo.Majaj2015.V4-pls',
        # 'dicarlo.Majaj2015.IT-pls',
        # 'dicarlo.Rajalingham2018-i2n',
        # 'fei-fei.Deng2009-top1'
    ]
    for benchmark in benchmarks:
        # score_models('CORnet-S', benchmark, filename)
        # score_models('alexnet_jumbler', benchmark, filename)
        # score_models('alexnet_kernel_jumbler', benchmark, filename)
        score_models('CORnet-S_train_gabor_fit_second_kernel_conv_epoch_05', benchmark)
        # score_models('alexnet_norm_dist_kernel', benchmark, filename)
        # score_models('CORnet-S_norm_dist', benchmark, filename)
        # score_models('resnet101', benchmark, filename)
        # score_models('densenet169', benchmark, filename)
