import datetime
import logging
import os
import sys
import traceback

from submission import score_model

# from submission import brain_translated_pool
from benchmark.database import create_connection, store_score
from model_impls.decoder_train_pool import brain_translated_pool

logger = logging.getLogger(__name__)

def run_benchmark(benchmark_identifier, model_name):
    print(f'>>>>>Start running model {model_name} on benchmark {benchmark_identifier}')
    model = brain_translated_pool[model_name]
    score = score_model(model_identifier=model_name, model=model, benchmark_identifier=benchmark_identifier)
    return score


def score_models(model, benchmark, filename):
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    path = f'{dir_path}/../scores.sqlite'
    db = create_connection(path)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    d = datetime.datetime.today()
    base_model = model.split('_')[0]
    raw_scores = []
    try:
        # repeat = models[model]
        # iterations = 4 if repeat else 1
        # for i in range(iterations):
        d = datetime.datetime.now()
        raw_score = run_benchmark(benchmark, model)
        result = raw_score.sel(aggregation='center')
        result = result.values
        store_score(db, (base_model,benchmark,d, result.item(0), model, False))
        raw_scores.append(result.item(0))
    except Exception as e:
        logging.error(f'Could not run model {model} because of following error')
        logging.error(e, exc_info=True)
        with open(f'error_{model}_{benchmark}.txt', 'w') as f:
            traceback.print_exc(file=f)

    finally:
        file = open(filename, 'a')
        file.write(benchmark)
        file.write('\n')
        file.write(str((model, raw_scores)))
        file.close()
        db.close()
        d = datetime.datetime.today()
        logger.info(f'\nJob finished at {d.isoformat()}\n')



if __name__ == '__main__':
    d = datetime.datetime.today()
    filename = f'results_{d.isoformat()}.txt'
    benchmarks = [
        # 'movshon.FreemanZiemba2013.V1-pls',
        # 'movshon.FreemanZiemba2013.V2-pls',
        # 'dicarlo.Majaj2015.V4-pls',
        # 'dicarlo.Majaj2015.IT-pls',
        'dicarlo.Rajalingham2018-i2n',
        'fei-fei.Deng2009-top1'
    ]
    for benchmark in benchmarks:
        # score_models('CORnet-S', benchmark, filename)
        # score_models('alexnet_jumbler', benchmark, filename)
        # score_models('alexnet_kernel_jumbler', benchmark, filename)
        score_models('CORnet-S_train_random', benchmark, filename)
        # score_models('alexnet_norm_dist_kernel', benchmark, filename)
        # score_models('CORnet-S_norm_dist', benchmark, filename)
        # score_models('resnet101', benchmark, filename)
        # score_models('densenet169', benchmark, filename)
