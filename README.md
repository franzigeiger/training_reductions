# Wiring up vision

This repository contains all sources for work published
in [Wiring Up Vision: Minimizing Supervised Synaptic Updates Needed to Produce a Primate Ventral Stream](https://www.biorxiv.org/content/10.1101/2020.06.08.140111v1)

It allows to:
- Initialize weights of [CORnet-S](https://www.biorxiv.org/content/10.1101/408385v1) with different methods, the ones used in the paper are:
        - Gabor initialization (use a gabor function and a multivariate gaussian distribution over its parameters to initialize CNN kernels)
        - Cluser initalization (use cluster centers and cluster variance from trained weights to sample new kernels.)
- Train CORnet-S under a variety of conditions:
        - Training with less images
        - Train with different routines (until convergence, only a few epochs,..)
        - Freezing any set of layers of the model
- Train transfer models (Resnet50 and Mobilienet) on weight initialization distributions extracted from CORnet-S
- Evaluate a trained model on benchmarks provided by brain-score.   


## Installation

Run `pip install .` to install all required dependencies.

## Paper plots

To create the plots shown in the paper (and some further analysis for rebuttals) run `analysis/paper_plots.py`.

## Weight compression

## Training a model

Start a model training with

```python base_models --name <model name> --convergence```

The caller allows to specify a variety of flags determining which type of training should be executed. For instance
`--convergence` enables training a model from scratch until convergence.

The model name must be a name specified in `base_models.__init__.py`. The entries in this file define the model
initialization and the layers to train.

### Benchmark

To evaluate performance of the various model versions we have a benchmarking suite, which can run the various brainscore
benchmarks.

```python benchmark --model <model_name> --benchmark <benchmark name>```

The name is also the identifier for the weight file. The benchmark name must be a benchmark that brainscore offers. For
the analysis here we used those benchmarks:

```    
        'movshon.FreemanZiemba2013.V1-pls',
        'movshon.FreemanZiemba2013.V2-pls',
        'dicarlo.Majaj2015.V4-pls',
        'dicarlo.Majaj2015.IT-pls',
        'dicarlo.Rajalingham2018-i2n',
        'fei-fei.Deng2009-top1'
```

Benchmark results are stored in the sqlLite database `scores.sqlite` and `scores_public.sqlite` for public benchmark
runs.

Scores are stored in the table called `raw_scores`.

## Pretrained weights

We provide CORnet-S weights for our best model with 3 different seeds in `/weights`.

# Learn more

For further details check out these ressources:
- The official [brain-score website](http://www.brain-score.org/)
- [brain-score repository](https://github.com/brain-score/brain-score)
- [Submit your own model to brain-score](https://github.com/brain-score/sample-model-submission)
- Other [candidate-models](https://github.com/brain-score/candidate_models) evaluated on brainscore

# Citation

To cite this work you can use this citation:

```
@inproceedings{
geiger2022wiring,
title={Wiring Up Vision: Minimizing Supervised Synaptic Updates Needed to Produce a Primate Ventral Stream},
author={Franziska Geiger and Martin Schrimpf and Tiago Marques and James J. DiCarlo},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=g1SzIRLQXMM}
}
```

