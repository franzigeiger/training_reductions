import os

import numpy as np
import torch
from ptflops import get_model_complexity_info

from benchmark.database import create_connection, store_analysis
from nets import get_model, trained_models, run_model_training
from nets.trainer_performance import train


def measure_performance(identifier):
    model = get_model(identifier, False, trained_models[identifier])
    config = trained_models[identifier]
    fixed_weights = 0
    train_weights = 0
    for name, m in model.named_parameters():
        if type(m) == torch.nn.Conv2d:
            size = 1
            for dim in np.shape(m.weight.data.cpu().numpy()): size *= dim
            if any(value in name for value in config['layers']):
                train_weights += size
            else:
                fixed_weights += size
    time = run_model_training(identifier, False, config, train)
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    path = f'{dir_path}/../scores.sqlite'
    db = create_connection(path)
    store_analysis(db, (identifier, time['time'], fixed_weights, train_weights, 0, flops))
