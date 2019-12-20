from benchmark.database import load_scores
from plot.plot_data import get_connection, plot_data_base


def plot_single_layer_perturbation():

    norm_dist_models=[]
    jumbler_models=[]
    fixed_models=[]
    for i in range(1,18):
        norm_dist_models.append(f'CORnet-S_norm_dist_L{i}')
        jumbler_models.append(f'CORnet-S_jumbler_L{i}')
        fixed_models.append(f'CORnet-S_fixed_value_L{i}')
    conn =get_connection()
    result_norm = load_scores(conn, norm_dist_models, ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_jumbler= load_scores(conn, jumbler_models, ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_fixed= load_scores(conn, fixed_models, ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])
    result_base = load_scores(conn, ['CORnet-S'], ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n'])

    it_data={}
    it_data['norm_dist'] = []
    it_data['jumbler'] = []
    it_data['fixed_value'] = []
    it_data['base'] = []
    behavior_data={}
    behavior_data['norm_dist'] = []
    behavior_data['jumbler'] = []
    behavior_data['fixed_value'] = []
    behavior_data['base'] = []
    labels = []
    for i in range(1,18):
        labels.append(f'L{i}')
        it_data['norm_dist'].append(result_norm[f'CORnet-S_norm_dist_L{i}'][0])
        it_data['jumbler'].append(result_jumbler[f'CORnet-S_jumbler_L{i}'][0])
        it_data['fixed_value'].append(result_fixed[f'CORnet-S_fixed_value_L{i}'][0])
        it_data['base'].append(result_base[f'CORnet-S'][0])
        behavior_data['norm_dist'].append(result_norm[f'CORnet-S_norm_dist_L{i}'][1])
        behavior_data['jumbler'].append(result_jumbler[f'CORnet-S_jumbler_L{i}'][1])
        behavior_data['fixed_value'].append(result_fixed[f'CORnet-S_fixed_value_L{i}'][1])
        behavior_data['base'].append(result_base[f'CORnet-S'][1])


    plot_data_base(it_data, 'it_benchmark_single_layer',labels, 'Conv Layers', 'Score', [0.0, 0.6])
    plot_data_base(behavior_data, 'behavior_benchmark_single_layer',labels, 'Conv Layers', 'Score', [0.0, 0.6])


if __name__ == '__main__':
    plot_single_layer_perturbation()