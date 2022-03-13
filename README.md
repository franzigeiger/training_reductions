# Wiring up vision

This repository contains all sources for work published
in [Wiring Up Vision: Minimizing Supervised Synaptic Updates Needed to Produce a Primate Ventral Stream](https://www.biorxiv.org/content/10.1101/2020.06.08.140111v1)


## Installation

Run `pip install .` to install all required dependencies.

## Paper plots

To create the plots shown in the paper (and some further analysis for rebuttals) run `analysis/paper_plots.py`.

This will create the 

## Weight compression



## Training a model

Start a model training with

```python base_models --name <model name> --convergence```

The caller allows to specify a variety of flags determining which type of training should be executed. For instance
`--convergence` enables training a model from scratch until convergence.

The model name must be a name specified in `base_models.__init__.py`. The entries in this file define the model
initialization and the layers to train.

### Benchmark

To evaluate performance of the various model versions we have a benchmarking suite, which can run the various brainscore benchmarks.


```python benchmark --model <model_name> --benchmark <>```

The name is also the identifier for the weight file. The benchmark name must be a benchmark that brainscore offers. For the analysis here we used those benchmarks:

```    
        'movshon.FreemanZiemba2013.V1-pls',
        'movshon.FreemanZiemba2013.V2-pls',
        'dicarlo.Majaj2015.V4-pls',
        'dicarlo.Majaj2015.IT-pls',
        'dicarlo.Rajalingham2018-i2n',
        'fei-fei.Deng2009-top1'
```

Benchmark results are stored in the sqlLite database `scores.sqlite` and `scores_public.sqlite` for public benchmark runs.

Scores are stored in the table called `raw_scores`.


## Pretrained weights

We provide CORnet-S weights for our best model with 3 different seeds in `/weights`.

