import sys

from nets.global_data import best_models_brain_avg_all
from runtime.compression import measure_performance

if __name__ == '__main__':
    if len(sys.argv) > 1 is not None:
        measure_performance(sys.argv[1], '6 layers(V1 & V2 - V2.conv3), Imagenet focus', True)
    for k, v in best_models_brain_avg_all.items():
        measure_performance(k, v, do_epoch=True, do_analysis=True)
