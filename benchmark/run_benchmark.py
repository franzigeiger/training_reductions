import datetime
import logging
import logging
import os
import statistics
import sys

from submission import score_model

# from submission import brain_translated_pool
from models.pool import brain_translated_pool


def run_benchmark(benchmark_identifier, model_name):
    print(f'>>>>>Start running model {model_name} on benchmark {benchmark_identifier}')
    model = brain_translated_pool[model_name]
    score = score_model(model_identifier=model_name, model=model, benchmark_identifier=benchmark_identifier)
    return score


def score_models():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    data = {}
    raw_data = {}
    d = datetime.datetime.today()
    file = open(f'results_{d.isoformat()}.txt', 'w')
    file.write(f'Job started at {d.isoformat()}\n')
    models = {
        'CORnet-S_jumbler': True,
        'CORnet-S_norm_dist': False,
        'CORnet-S_uniform_dist': False,
        'CORnet-S_random': True,
        'CORnet-S': False
    }
    benchmarks = [
        # 'movshon.FreemanZiemba2013.V1-pls',
        # 'movshon.FreemanZiemba2013.V2-pls',
        # 'dicarlo.Majaj2015.V4-pls',
        # 'dicarlo.Majaj2015.IT-pls',
        'dicarlo.Rajalingham2018-i2n',
        'fei-fei.Deng2009-top1'
    ]
    try:
        for model, repeat in models.items():
            scores = []
            raw_all = []
            for benchmark in benchmarks:
                raw_scores = []
                iterations = 4 if repeat else 1
                for i in range(iterations):
                    raw_score = run_benchmark(benchmark, model)
                    os.environ["RESULTCACHING_DISABLE"] = "1"
                    result = raw_score.sel(aggregation='center')
                    result = result.values
                    raw_scores.append(result.item(0))
                os.environ["RESULTCACHING_DISABLE"] = '0'
                score = statistics.mean(raw_scores)
                file.write(f'Raw values for {model} on {benchmark}: \n {str(raw_scores)} \n')
                scores.append(score)
                raw_all.append(raw_scores)
            file.write(f'Results for model{model}')
            print(f'Result for model{model} following values')
            print(str(scores))
            file.write(str(scores))
            file.write('\n')
            data[model] = scores
            raw_data[model] = raw_all
    finally:
        file.write(str(data))
        file.write('\n')
        file.write(str(raw_data))
        d = datetime.datetime.today()
        file.write(f'\nJob finished at {d.isoformat()}\n')
        file.close()


if __name__ == '__main__':
    score_models()
