from benchmark.database import load_scores
from plot.plot_data import get_connection, plot_data_base


def plot_single_layer_perturbation():
    norm_dist_models = []
    jumbler_models = []
    fixed_models = []
    fixed_small_models = []
    for i in range(1, 18):
        norm_dist_models.append(f'CORnet-S_norm_dist_L{i}')
        jumbler_models.append(f'CORnet-S_jumbler_L{i}')
        fixed_models.append(f'CORnet-S_fixed_value_L{i}')
        fixed_small_models.append(f'CORnet-S_fixed_value_small_L{i}')
    conn = get_connection()
    result_norm = load_scores(conn, norm_dist_models, ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_jumbler = load_scores(conn, jumbler_models, ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_fixed = load_scores(conn, fixed_models, ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_base = load_scores(conn, ['CORnet-S'], ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_fixed_small = load_scores(conn, fixed_small_models,
                                     ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])

    it_data = {}
    it_data['norm_dist'] = []
    it_data['jumbler'] = []
    it_data['fixed_value'] = []
    it_data['fixed_small_value'] = []
    it_data['base'] = []
    behavior_data = {}
    behavior_data['norm_dist'] = []
    behavior_data['jumbler'] = []
    behavior_data['fixed_value'] = []
    behavior_data['fixed_small_value'] = []
    behavior_data['base'] = []
    labels = []
    for i in range(1, 18):
        labels.append(f'L{i}')
        it_data['norm_dist'].append(result_norm[f'CORnet-S_norm_dist_L{i}'][0])
        it_data['jumbler'].append(result_jumbler[f'CORnet-S_jumbler_L{i}'][0])
        it_data['fixed_value'].append(result_fixed[f'CORnet-S_fixed_value_L{i}'][0])
        it_data['base'].append(result_base[f'CORnet-S'][0])
        it_data['fixed_small_value'].append(result_fixed_small[f'CORnet-S_fixed_value_small_L{i}'][0])
        behavior_data['norm_dist'].append(result_norm[f'CORnet-S_norm_dist_L{i}'][1])
        behavior_data['jumbler'].append(result_jumbler[f'CORnet-S_jumbler_L{i}'][1])
        behavior_data['fixed_value'].append(result_fixed[f'CORnet-S_fixed_value_L{i}'][1])
        behavior_data['base'].append(result_base[f'CORnet-S'][1])
        behavior_data['fixed_small_value'].append(result_fixed_small[f'CORnet-S_fixed_value_small_L{i}'][1])

    plot_data_base(it_data, 'IT Benchmark single layer', labels, 'Conv Layers', 'Score', [0.0, 0.6])
    plot_data_base(behavior_data, 'Behavior Benchmark single layer', labels, 'Conv Layers', 'Score', [0.0, 0.6])


def plot_high_low_variance():
    high_var_models = []
    high_var_trained_models = []
    low_var_models = []
    low_var_trained_models = []
    for i in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        high_var_models.append(f'CORnet-S_high_variance_{i}')
        high_var_trained_models.append(f'CORnet-S_trained_high_variance_{i}')
        low_var_models.append(f'CORnet-S_low_variance_{i}')
        low_var_trained_models.append(f'CORnet-S_trained_low_variance_{i}')
    conn = get_connection()
    result_high_var = load_scores(conn, high_var_models, ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_high_var_trained = load_scores(conn, high_var_trained_models,
                                          ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_low_var = load_scores(conn, low_var_models, ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_low_var_trained = load_scores(conn, low_var_trained_models,
                                         ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_base_random = load_scores(conn, ['CORnet-S_random'],
                                     ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_base = load_scores(conn, ['CORnet-S'], ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])

    it_data = {}
    it_data['high_variance_untrained'] = []
    it_data['high_variance_trained'] = []
    it_data['low_variance_untrained'] = []
    it_data['low_variance_trained'] = []
    it_data['base_untrained'] = []
    it_data['base_trained'] = []
    behavior_data = {}
    behavior_data['high_variance_untrained'] = []
    behavior_data['high_variance_trained'] = []
    behavior_data['low_variance_trained'] = []
    behavior_data['low_variance_untrained'] = []
    behavior_data['base_untrained'] = []
    behavior_data['base_trained'] = []
    labels = []
    for i in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        labels.append(i)
        it_data['high_variance_trained'].append(result_high_var_trained[f'CORnet-S_trained_high_variance_{i}'][0])
        it_data['high_variance_untrained'].append(result_high_var[f'CORnet-S_high_variance_{i}'][0])
        it_data['low_variance_untrained'].append(result_low_var[f'CORnet-S_low_variance_{i}'][0])
        it_data['low_variance_trained'].append(result_low_var_trained[f'CORnet-S_trained_low_variance_{i}'][0])
        it_data['base_untrained'].append(result_base_random[f'CORnet-S_random'][0])
        it_data['base_trained'].append(result_base[f'CORnet-S'][0])
        behavior_data['high_variance_untrained'].append(result_high_var[f'CORnet-S_high_variance_{i}'][1])
        behavior_data['high_variance_trained'].append(result_high_var_trained[f'CORnet-S_trained_high_variance_{i}'][1])
        behavior_data['low_variance_untrained'].append(result_low_var[f'CORnet-S_low_variance_{i}'][1])
        behavior_data['low_variance_trained'].append(result_low_var_trained[f'CORnet-S_trained_low_variance_{i}'][1])
        behavior_data['base_untrained'].append(result_base_random[f'CORnet-S_random'][1])
        behavior_data['base_trained'].append(result_base[f'CORnet-S'][1])

    plot_data_base(it_data, 'it_benchmark_variance', labels, 'Zero values in %', 'Score', [0.0, 0.6])
    plot_data_base(behavior_data, 'behavior_benchmark_variance', labels, 'Zero values in %', 'Score', [0.0, 0.6])


def plot_high_low_variance_separate():
    high_var_models = []
    high_var_trained_models = []
    low_var_models = []
    low_var_trained_models = []
    for i in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        high_var_models.append(f'CORnet-S_high_variance_{i}')
        high_var_trained_models.append(f'CORnet-S_trained_high_variance_{i}')
        low_var_models.append(f'CORnet-S_low_variance_{i}')
        low_var_trained_models.append(f'CORnet-S_trained_low_variance_{i}')
    conn = get_connection()
    result_high_var = load_scores(conn, high_var_models, ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_high_var_trained = load_scores(conn, high_var_trained_models,
                                          ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_low_var = load_scores(conn, low_var_models, ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_low_var_trained = load_scores(conn, low_var_trained_models,
                                         ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_base_random = load_scores(conn, ['CORnet-S_random'],
                                     ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_base = load_scores(conn, ['CORnet-S'], ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])

    it_data = {}
    # it_data['high_variance_untrained'] = []
    it_data['high_variance'] = []
    # it_data['low_variance_untrained'] = []
    it_data['low_variance'] = []
    it_data['base'] = []
    # it_data['base_trained'] = []
    behavior_data = {}
    behavior_data['high_variance'] = []
    # behavior_data['high_variance'] = []
    behavior_data['low_variance'] = []
    # behavior_data['low_variance_untrained'] = []
    # behavior_data['base_untrained'] = []
    behavior_data['base'] = []
    labels = []
    for i in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        labels.append(i)
        it_data['high_variance'].append(result_high_var_trained[f'CORnet-S_trained_high_variance_{i}'][0])
        # it_data['high_variance'].append(result_high_var[f'CORnet-S_high_variance_{i}'][0])
        # it_data['low_variance'].append(result_low_var[f'CORnet-S_low_variance_{i}'][0])
        it_data['low_variance'].append(result_low_var_trained[f'CORnet-S_trained_low_variance_{i}'][0])
        # it_data['base'].append(result_base_random[f'CORnet-S_random'][0])
        it_data['base'].append(result_base[f'CORnet-S'][0])
        # behavior_data['high_variance'].append(result_high_var[f'CORnet-S_high_variance_{i}'][1])
        behavior_data['high_variance'].append(result_high_var_trained[f'CORnet-S_trained_high_variance_{i}'][1])
        # behavior_data['low_variance'].append(result_low_var[f'CORnet-S_low_variance_{i}'][1])
        behavior_data['low_variance'].append(result_low_var_trained[f'CORnet-S_trained_low_variance_{i}'][1])
        # behavior_data['base'].append(result_base_random[f'CORnet-S_random'][1])
        behavior_data['base'].append(result_base[f'CORnet-S'][1])

    plot_data_base(it_data, 'IT Benchmark Variance Trained', labels, 'Zero kernels in %', 'Score', [0.0, 0.6])
    plot_data_base(behavior_data, 'Behavior Benchmark Variance Trained', labels, 'Zero kernels in %', 'Score',
                   [0.0, 0.6])


def plot_high_low_nullify():
    high_var_models = []
    high_var_trained_models = []
    low_var_models = []
    low_var_trained_models = []
    for i in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        high_var_models.append(f'CORnet-S_high_zero_{i}')
        high_var_trained_models.append(f'CORnet-S_trained_high_zero_{i}')
        low_var_models.append(f'CORnet-S_low_zero_{i}')
        low_var_trained_models.append(f'CORnet-S_trained_low_zero_{i}')
    conn = get_connection()
    result_high_var = load_scores(conn, high_var_models, ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_high_var_trained = load_scores(conn, high_var_trained_models,
                                          ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_low_var = load_scores(conn, low_var_models, ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_low_var_trained = load_scores(conn, low_var_trained_models,
                                         ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_base_random = load_scores(conn, ['CORnet-S_random'],
                                     ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_base = load_scores(conn, ['CORnet-S'], ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])

    it_data = {}
    it_data['high_zero_untrained'] = []
    it_data['low_zero_untrained'] = []
    it_data['high_zero_trained'] = []
    it_data['low_zero_trained'] = []
    it_data['base_untrained'] = []
    it_data['base_trained'] = []
    behavior_data = {}
    behavior_data['high_zero_untrained'] = []
    behavior_data['low_zero_untrained'] = []
    behavior_data['high_zero_trained'] = []
    behavior_data['low_zero_trained'] = []
    behavior_data['base_untrained'] = []
    behavior_data['base_trained'] = []
    labels = []
    for i in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        labels.append(i)
        it_data['high_zero_untrained'].append(result_high_var[f'CORnet-S_high_zero_{i}'][0])
        it_data['high_zero_trained'].append(result_high_var_trained[f'CORnet-S_trained_high_zero_{i}'][0])
        it_data['low_zero_untrained'].append(result_low_var[f'CORnet-S_low_zero_{i}'][0])
        it_data['low_zero_trained'].append(result_low_var_trained[f'CORnet-S_trained_low_zero_{i}'][0])
        it_data['base_untrained'].append(result_base_random[f'CORnet-S_random'][0])
        it_data['base_trained'].append(result_base[f'CORnet-S'][0])
        behavior_data['high_zero_untrained'].append(result_high_var[f'CORnet-S_high_zero_{i}'][1])
        behavior_data['high_zero_trained'].append(result_high_var_trained[f'CORnet-S_trained_high_zero_{i}'][1])
        behavior_data['low_zero_untrained'].append(result_low_var[f'CORnet-S_low_zero_{i}'][1])
        behavior_data['low_zero_trained'].append(result_low_var_trained[f'CORnet-S_trained_low_zero_{i}'][1])
        behavior_data['base_untrained'].append(result_base_random[f'CORnet-S_random'][1])
        behavior_data['base_trained'].append(result_base[f'CORnet-S'][1])

    plot_data_base(it_data, 'it_benchmark_zero', labels, 'Zero values in %', 'Score', [0.0, 0.6])
    plot_data_base(behavior_data, 'behavior_benchmark_zero', labels, 'Zero values in %', 'Score', [0.0, 0.6])


def plot_high_low_nullify_separate():
    high_var_models = []
    high_var_trained_models = []
    low_var_models = []
    low_var_trained_models = []
    for i in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        high_var_models.append(f'CORnet-S_high_zero_{i}')
        high_var_trained_models.append(f'CORnet-S_trained_high_zero_{i}')
        low_var_models.append(f'CORnet-S_low_zero_{i}')
        low_var_trained_models.append(f'CORnet-S_trained_low_zero_{i}')
    conn = get_connection()
    result_high_var = load_scores(conn, high_var_models, ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_high_var_trained = load_scores(conn, high_var_trained_models,
                                          ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_low_var = load_scores(conn, low_var_models, ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_low_var_trained = load_scores(conn, low_var_trained_models,
                                         ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_base_random = load_scores(conn, ['CORnet-S_random'],
                                     ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_base = load_scores(conn, ['CORnet-S'], ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])

    it_data = {}
    it_data['high_zero'] = []
    it_data['low_zero'] = []
    # it_data['high_zero_trained'] = []
    # it_data['low_zero_trained'] = []
    it_data['base'] = []
    # it_data['base_trained'] = []
    behavior_data = {}
    behavior_data['high_zero'] = []
    behavior_data['low_zero'] = []
    # behavior_data['high_zero_trained'] = []
    # behavior_data['low_zero_trained'] = []
    behavior_data['base'] = []
    # behavior_data['base_trained'] = []
    labels = []
    for i in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        labels.append(i)
        # it_data['high_zero'].append(result_high_var[f'CORnet-S_high_zero_{i}'][0])
        it_data['high_zero'].append(result_high_var_trained[f'CORnet-S_trained_high_zero_{i}'][0])
        # it_data['low_zero'].append(result_low_var[f'CORnet-S_low_zero_{i}'][0])
        it_data['low_zero'].append(result_low_var_trained[f'CORnet-S_trained_low_zero_{i}'][0])
        # it_data['base'].append(result_base_random[f'CORnet-S_random'][0])
        it_data['base'].append(result_base[f'CORnet-S'][0])
        # behavior_data['high_zero'].append(result_high_var[f'CORnet-S_high_zero_{i}'][1])
        behavior_data['high_zero'].append(result_high_var_trained[f'CORnet-S_trained_high_zero_{i}'][1])
        # behavior_data['low_zero'].append(result_low_var[f'CORnet-S_low_zero_{i}'][1])
        behavior_data['low_zero'].append(result_low_var_trained[f'CORnet-S_trained_low_zero_{i}'][1])
        # behavior_data['base'].append(result_base_random[f'CORnet-S_random'][1])
        behavior_data['base'].append(result_base[f'CORnet-S'][1])

    plot_data_base(it_data, 'IT Benchmark Zero Trained', labels, 'Zero values in %', 'Score', [0.0, 0.6],
                   base_line=result_base[f'CORnet-S'][0])
    plot_data_base(behavior_data, 'Behavior Benchmark Zero Trained', labels, 'Zero values in %', 'Score', [0.0, 0.6],
                   base_line=result_base[f'CORnet-S'][1])



if __name__ == '__main__':
    # plot_high_low_variance()
    # plot_high_low_nullify()
    # plot_single_layer_perturbation()
    plot_high_low_nullify_separate()
    # plot_high_low_variance_separate()
