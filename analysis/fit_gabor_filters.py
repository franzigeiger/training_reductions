import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy import linalg
from scipy.optimize import basinhopping
from skimage.filters import gabor_kernel
from skimage.transform import resize
from sklearn import feature_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import ParameterGrid
from torch import nn

from base_models import get_model
from plot.plot_data import plot_subplots_histograms, plot_3d, plot_heatmap, plot_2d, plot_histogram, plot_matrixImage
from utils.correlation import generate_correlation_map, pca, fit_data, multivariate_gaussian
from utils.distributions import mixture_gaussian
from utils.gabors import gabor_kernel_3, normalize, similarity


def score_kernel(X, theta, frequency, sigma, offset, stds):
    kernel = np.real(gabor_kernel(frequency, theta=theta,
                                  sigma_x=sigma, sigma_y=sigma, n_stds=stds, offset=offset))
    kernel = resize(kernel, (X.shape[0], X.shape[0]),
                    anti_aliasing=True)
    kernel = normalize(kernel)
    kernel = np.nan_to_num(kernel, posinf=1, neginf=-1, nan=0)
    score = explained_variance_score(kernel, X)

    return score


def objective_function_2(beta, X, size=3):
    kernel = gabor_kernel_3(beta[0], theta=beta[1], sigma_x=beta[2], sigma_y=beta[3], offset=beta[4], x_c=beta[5],
                            y_c=beta[6], scale=beta[7], ks=size)
    kernel = normalize(kernel)
    error = mean_squared_error(kernel, X)
    return (error)


def objective_function(beta, X):
    kernel = np.real(gabor_kernel(beta[0], theta=beta[1],
                                  sigma_x=beta[2], sigma_y=beta[3], offset=beta[4], n_stds=beta[5]))
    kernel = resize(kernel, (X.shape[0], X.shape[0]),
                    anti_aliasing=True)
    kernel = normalize(kernel)
    kernel = np.nan_to_num(kernel, posinf=1, neginf=-1, nan=0)
    error = mean_squared_error(kernel, X)
    return (error)


def linear_regression(x, y, lin: LinearRegression):
    y_hat = lin.predict(x)
    return 1 - explained_variance_score(y, y_hat)


def fit_linear_regression():
    input = np.random.random([1, 8])
    output = np.random.random([1, 9])
    lin = LinearRegression()
    lin.fit(input, output)
    model = get_model('CORnet-S_base', True)
    counter = 0
    np.random.seed(1)

    def print_fun(x, f, accepted):
        print("at minima %.4f accepted %d" % (f, int(accepted)))

    fun = lambda x, y: linear_regression(x, y, lin)
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d and counter == 0:
            weights = m.weight.data.cpu().numpy()
            lin_params = np.zeros([weights.shape[0], weights.shape[1], 8])
            for i in range(0, weights.shape[0]):
                for j in range(0, weights.shape[1]):
                    kernel = weights[i, j].flatten()
                    minimizer_kwargs = {"method": "L-BFGS-B", 'args': (kernel),
                                        'options': {'maxiter': 200000, 'gtol': 1e-25}}
                    result = basinhopping(fun, [0, 0, 0, 0, 0, 0, 0, 0], niter=15,
                                          minimizer_kwargs=minimizer_kwargs,
                                          callback=print_fun, T=0.00001)
                    y_hat = result.x
                    lin_params[i, j] = y_hat


def hyperparam_gabor():
    model = get_model('CORnet-S_base', True)
    counter = 0
    gabor_params = np.zeros([64, 3, 5])
    np.random.seed(1)
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d and counter == 0:
            weights = m.weight.data.cpu().numpy()
            for i in range(0, 10):
                for j in range(0, 3):
                    kernel = weights[i, j]
                    kernel = normalize(kernel)
                    tuned_params = {"theta": np.arange(0, 1, 0.05),
                                    "frequency": np.arange(0, np.pi, np.pi / 12),
                                    "sigma": np.arange(0, 4, 0.25),
                                    "offset": np.arange(-2, 2, 0.5),
                                    "stds": np.arange(1, 4, 0.5),
                                    }
                    best_score = np.NINF
                    best_params = {}
                    for g in ParameterGrid(tuned_params):
                        score = score_kernel(kernel, **g)
                        if score > best_score:
                            best_score = score
                            print(f'Update best score: {score}')
                            best_params = g
                    print(f'Best grid:{best_params} for kernel {i}, filter {j}')
                    gabor_params[i, j] = np.fromiter(best_params.values(), dtype=float)
            np.save('gabor_params_grid_search_long.npy', gabor_params)
            return


def fit_gabors(version='V1', file='gabor_params_basinhopping'):
    model = get_model('CORnet-S_full_epoch_43', True)
    counter = 0
    length = 7 if version is 'V1' else 9

    np.random.seed(1)
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            if name == 'V2.conv2':
                weights = m.weight.data.cpu().numpy()
                gabor_params = np.zeros([weights.shape[0], weights.shape[1], length])
                for i in range(0, weights.shape[0]):
                    for j in range(0, weights.shape[1]):
                        kernel = weights[i, j]
                        kernel = normalize(kernel)
                        bnds = ((0, 0.5), (-np.pi, 2 * np.pi), (-4, 4), (-4, 4), (-3, 3))
                        params = [0.5, np.pi / 2, 2, 2, 0]

                        def print_fun(x, f, accepted):
                            print("at minima %.4f accepted %d" % (f, int(accepted)))

                        if version is 'V1':
                            print('Use sklearn version')
                            bnds = ((-0.5, 1.5), (-np.pi, 2 * np.pi), (-4, 4), (-4, 4), (-3, 3), (-5, 5))
                            params = [0.5, np.pi / 2, 2, 2, 0, 3]
                            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bnds, 'args': (kernel),
                                                'options': {'maxiter': 200000, 'gtol': 1e-25}}
                            result = basinhopping(objective_function, params, niter=15,
                                                  minimizer_kwargs=minimizer_kwargs,
                                                  callback=print_fun, T=0.00001)
                        else:
                            print('Use Tiagos version')
                            bnds = (
                                (1 / 14, 0.5), (-2 * np.pi, 2 * np.pi), (2, 14), (2, 14), (-2 * np.pi, 2 * np.pi),
                                (-2, 2),
                                (-2, 2), (1e-5, 2))
                            params = [0.2, 0, 4, 4, 0, 0, 0, 1]
                            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bnds, 'args': (kernel),
                                                'options': {'maxiter': 200000, 'gtol': 1e-25}}
                            result = basinhopping(objective_function_2, params, niter=15,
                                                  minimizer_kwargs=minimizer_kwargs,
                                                  callback=print_fun, T=0.00001)
                        beta_hat = result.x
                        gabor_params[i, j] = np.append(beta_hat, result.fun)
                        print(f'Kernel {i}, filter {j}:')
                        print(beta_hat)
                np.save(f'{file}.npy', gabor_params)
                return
            counter += 1


def get_fist_layer_weights():
    # model = get_model('CORnet-S_full_epoch_43', True)
    counter = 0
    plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(10, 3, width_ratios=[1] * 3,
                           wspace=0.5, hspace=0.5, top=0.95, bottom=0.05, left=0.1, right=0.95)
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d and counter == 0:
            weights = m.weight.data.squeeze().numpy()
            idx = 1
            for i in range(0, 10):
                kernel1 = weights[i, 0]
                kernel2 = weights[i, 1]
                kernel3 = weights[i, 2]
                ax = plt.subplot(gs[i, 0])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'Samples parameter K1', pad=2, fontsize=5)
                idx += 1
                plt.imshow(kernel2, cmap='gray')
                ax = plt.subplot(gs[i, 1])
                ax.set_title(f'Samples parameter K2', pad=2, fontsize=5)
                idx += 1
                plt.imshow(kernel1, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax = plt.subplot(gs[i, 2])
                ax.set_title(f'Samples parameter K3', pad=2, fontsize=5)
                plt.imshow(kernel3, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])

                idx += 1
        plt.tight_layout()
        plt.savefig('full_kernels.png')
        plt.show()
        return


def compare_gabors(version='V1', file='gabor_params_basinhopping_bound_6_params'):
    gabor_params = np.load(f'{file}.npy')
    model = get_model('CORnet-S_full_epoch_43', True)
    counter = 0
    plt.figure(figsize=(4, 25))
    gs = gridspec.GridSpec(20, 3, width_ratios=[1] * 3,
                           wspace=0.5, hspace=0.5, top=0.95, bottom=0.05, left=0.1, right=0.95)
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d and counter == 0:
            weights = m.weight.data.squeeze().numpy()
            index = 1
            for i in range(0, 10):
                for j in range(0, 3):
                    beta = gabor_params[i, j]
                    kernel2 = weights[i, j]
                    kernel2 = normalize(kernel2)
                    if version is "V1":
                        kernel1 = np.real(gabor_kernel(beta[0], theta=beta[1],
                                                       sigma_x=beta[2], sigma_y=beta[3], offset=beta[4],
                                                       n_stds=beta[5]))
                        kernel1 = np.nan_to_num(kernel1).astype(np.float32)
                        kernel1 = resize(kernel1, (kernel2.shape[0], kernel2.shape[0]),
                                         anti_aliasing=True, preserve_range=True)
                        kernel1 = normalize(kernel1)
                    else:
                        kernel1 = gabor_kernel_3(beta[0], theta=beta[1],
                                                 sigma_x=beta[2], sigma_y=beta[3], offset=beta[4], x_c=beta[5],
                                                 y_c=beta[6], scale=beta[7], ks=7)
                        kernel1 = normalize(kernel1)
                    ax = plt.subplot(gs[i * 2, j])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'K {i}, F {j}', pad=3)
                    plt.imshow(kernel2, cmap='gray')
                    ax = plt.subplot(gs[(i * 2) + 1, j])
                    ax.set_title(f'Error {beta[-1]:.3}', pad=10)
                    plt.imshow(kernel1, cmap='gray')
                    index += 1
                index += 3

            plt.savefig(f'{file}.png')
            plt.show()
            return


def rank_errors(name1, name2):
    n1 = np.load(f'{name1}.npy')
    n2 = np.load(f'{name2}.npy')
    value = np.mean(n1[:, -1])
    value2 = np.mean(n2[:, -1])
    print(f'Mean error of fit {name1} is {value}, mean error of fit {name2} is {value2}')


def plot_parameter_distribution(name):
    params = np.load(f'{name}.npy')
    data = {}
    names = ['Frequency', 'Theta', 'Sigma X', 'Sigma Y', 'Offset', 'Center X', 'Center Y']
    variables = np.zeros((7, 192))
    for i in range(params.shape[2] - 1):
        param = params[:, :, i]
        data[names[i]] = param.flatten()
        variables[i] = param.flatten()
    bnds = ((1 / 14, 0.5), (-2 * np.pi, 2 * np.pi), (2, 14), (2, 14), (-2 * np.pi, 2 * np.pi), (-2, 2), (-2, 2))
    plot_subplots_histograms(data, 'Gabor parameter distributions', bins=10, bounds=bnds)


def mutual_information():
    model = get_model('CORnet-S_base', True)
    plt.figure(figsize=(4, 25))
    gs = gridspec.GridSpec(20, 3, width_ratios=[1] * 3,
                           wspace=0.5, hspace=0.5, top=0.95, bottom=0.05, left=0.1, right=0.95)
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d and 'V1' not in name:
            weights = m.weight.data.squeeze().numpy()
            scores = np.zeros([weights.shape[0], weights.shape[0]])
            kernels = weights.shape[0]
            for i in range(kernels):
                for j in range(kernels):
                    scores[i, j] = feature_selection.mutual_info_regression(weights[i].flatten().reshape(-1, 1),
                                                                            weights[j].flatten())
            print(f'Kernel mean mutual information {np.mean(scores)}')
            plot_heatmap(scores, title=f'Kernel mutual information {name}', col_labels='Kernels', row_labels='Kernels')
            if len(weights.shape) > 2:
                channels = weights.shape[1]
                scores = np.zeros([kernels, channels, channels])
                for i in range(kernels):
                    for j in range(channels):
                        for k in range(channels):
                            scores[i, j, k] = feature_selection.mutual_info_regression(
                                weights[i, j].flatten().reshape(-1, 1), weights[i, k].flatten())
                scores = np.mean(scores, axis=0)
                plot_heatmap(scores, title=f'Channel mean mutual information {name}', col_labels='Channel',
                             row_labels='Channel')



def kernel_similarity():
    model = get_model('CORnet-S_train_second_kernel_conv_epoch_00', True)
    counter = 0
    plt.figure(figsize=(4, 25))
    gs = gridspec.GridSpec(20, 3, width_ratios=[1] * 3,
                           wspace=0.5, hspace=0.5, top=0.95, bottom=0.05, left=0.1, right=0.95)
    kernel_avgs = []
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            if counter == 1:
                weights = m.weight.data.squeeze().numpy().transpose(1, 0, 2, 3)
                for i in range(0, 64):
                    avgs = []
                    for j in range(0, 64):
                        for k in range(j, 64):
                            if k != j:
                                c1 = weights[i, j]
                                c2 = weights[i, k]
                                avgs.append(similarity(c2, c1))
                    kernel_avgs.append(np.mean(avgs))
                plot_histogram(kernel_avgs, 'Channel similarity within a kernel', bins=20)
                print('Similarity avg:' + str(np.mean(kernel_avgs)))
                return
            counter += 1


def mixture_analysis(pi, mu, cov):
    # k components: pi = (k) , mu = (k), cov = (k,k,k)
    print(pi.shape, mu.shape, cov.shape)


def gaussian_mixture_channels(name):
    params = np.load(f'{name}.npy')
    param = params[:, :, :-1]
    param = param.reshape(-1, 8)
    mixture_base(param, [(0, 10)], [4096, 1])


def gaussian_mixture(name, name2=None):
    params = np.load(f'{name}.npy')
    names = ['Frequency', 'Theta', 'Sigma X', 'Sigma Y', 'Offset', 'Center X', 'Center Y']
    shape = [64, 64, 3, 3]
    param = params[:, :, :-1]
    param = param.reshape(64, -1)
    tuples = []
    for i in range(param.shape[1]):
        tuples.append((i * 10, (i + 1) * 10))
    if name2 is not None:
        params2 = np.load(f'{name}.npy')
        p2 = params[:, :, :-1]
        p2 = p2.reshape(64, -1)
        param = np.concatenate((param, p2), axis=0)
    mixture_base(param, tuples, shape)


def mixture_base(param, tuples, shape, size=3):
    new_param = np.zeros((shape[0], 10 * shape[1]))
    for i in range(shape[0]):
        for s, e in tuples:
            p = param[i, int(s * 8 / 10):int(e * 8 / 10)]
            new_param[i, s:e] = (
                p[0], np.sin(2 * p[1]), np.cos(2 * p[1]), p[2], p[3], np.sin(p[4]), np.cos(p[4]), p[5], p[6], p[7])

    # plot_bic(best_gmm, param)
    best_gmm = mixture_gaussian(new_param)
    samples = best_gmm.sample(new_param.shape[0])[0]
    idx = 1

    plt.figure(figsize=(10, 20))
    gs = gridspec.GridSpec(22, 3, width_ratios=[1] * 3,
                           wspace=0.5, hspace=0.5, top=0.95, bottom=0.05, left=0.1, right=0.95)
    for i in range(shape[0]):
        for s, e in tuples:
            beta = samples[i, s:e]
            kernel2 = gabor_kernel_3(beta[0], theta=np.arctan2(beta[1], beta[2]),
                                     sigma_x=beta[3], sigma_y=beta[4], offset=np.arctan2(beta[5], beta[6]), x_c=beta[7],
                                     y_c=beta[8],
                                     scale=beta[9], ks=size)
            ax = plt.subplot(gs[i, int(s / 10)])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(kernel2, cmap='gray')
        idx += 1
    plt.figure(figsize=(15, 15))

    length = 64 * 3
    matrix = np.empty([length, 0])
    weights = samples.reshape(64, 64, 10)
    for i in range(0, 64):
        row = np.empty([0, 3])
        for j in range(0, 64):
            beta = weights[i, j]
            channel = gabor_kernel_3(beta[0], theta=np.arctan2(beta[1], beta[2]),
                                     sigma_x=beta[3], sigma_y=beta[4], offset=np.arctan2(beta[5], beta[6]), x_c=beta[7],
                                     y_c=beta[8],
                                     scale=beta[9], ks=size)
            row = np.concatenate((row, channel), axis=0)
        f_min, f_max = np.min(row), np.max(row)
        row = (row - f_min) / (f_max - f_min)
        matrix = np.concatenate((matrix, row), axis=1)
    plot_matrixImage(matrix, 'Gabor_conv2')
    plt.tight_layout()
    plt.show()


def plot_bic(clf, X):
    plt.figure(figsize=(20, 20))
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange', 'red', 'yellow'])
    size = X.shape[1]
    Y_ = clf.predict(X)
    idx = 1
    for l in range(size):
        for j in range(X.shape[1]):
            splot = plt.subplot(size, size, idx)
            idx += 1
            for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                                       color_iter)):
                v, w = linalg.eigh(cov)
                if not np.any(Y_ == i):
                    continue
                plt.scatter(X[Y_ == i, l], X[Y_ == i, j], 1, color=color)

                # Plot an ellipse to show the Gaussian component
                angle = np.arctan2(w[0][j], w[0][l])
                angle = 180. * angle / np.pi  # convert to degrees
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                ell = mpl.patches.Ellipse(mean, v[l], v[j], 180. + angle, color=color)
                ell.set_clip_box(splot.bbox)
                ell.set_alpha(.5)
                splot.add_artist(ell)
        plt.xticks(())
        plt.yticks(())
        plt.title(f'Dimension {l} and {j}')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.savefig('Mixture gaussians')
    plt.show()


def analyze_param_dist(name, plot=False):
    params = np.load(f'{name}.npy')
    names = ['Frequency', 'Theta', 'Sigma X', 'Sigma Y', 'Offset', 'Center X', 'Center Y']
    variables = np.zeros((7, 192))
    param = params[:, :, :-1]
    param = param.reshape(64, -1)

    pca_res = pca(param, n_components=21)
    principal_components = pca_res.transform(param)
    if plot:
        plot_2d(principal_components[:, 0], principal_components[:, 1], 'PC 1 & 2')
        plot_2d(principal_components[:, 1], principal_components[:, 2], 'PC 2 & 3')
        plot_2d(principal_components[:, 0], principal_components[:, 2], 'PC 1 & 3')
        plot_2d(principal_components[:, 0], principal_components[:, 3], 'PC 1 & 4')
        plot_2d(principal_components[:, 0], principal_components[:, 4], 'PC 1 & 5')
        plot_3d(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2],
                'Principal components of gabor filter params')
    reg = fit_data(principal_components, variables.T)
    small_samples = multivariate_gaussian(principal_components.T, 10)
    full_params = reg.predict(small_samples.T)  # shape (10, 7)
    small_samples_hat = pca_res.transform(full_params)  # shape (10,3)
    full_params_hat = reg.predict(small_samples_hat)
    print(mean_squared_error(small_samples.T, small_samples_hat))
    idx = 0
    plt.figure(figsize=(5, 10))
    gs = gridspec.GridSpec(20, 3, width_ratios=[1] * 3,
                           wspace=0.5, hspace=0.5, top=0.95, bottom=0.05, left=0.1, right=0.95)
    for i in range(10):
        alpha = full_params[i]
        beta = full_params_hat[i]
        kernel2 = gabor_kernel_3(beta[0], theta=beta[1],
                                 sigma_x=beta[2], sigma_y=beta[3], offset=beta[4], x_c=beta[5], y_c=beta[6], ks=7)
        kernel1 = gabor_kernel_3(alpha[0], theta=alpha[1],
                                 sigma_x=alpha[2], sigma_y=alpha[3], offset=alpha[4], x_c=alpha[5], y_c=alpha[6], ks=7)
        ax = plt.subplot(gs[i, 0])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Samples parameter set', pad=3, fontsize=5)
        idx += 1
        plt.imshow(kernel2, cmap='gray')
        ax = plt.subplot(gs[i, 1])
        ax.set_title(f'Reconstruction', pad=10, fontsize=5)
        plt.imshow(kernel1, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        idx += 1
    plt.savefig(f'reconstructions.png')
    plt.show()

    if plot:
        principal_components = principal_components.T
        corr_pca = generate_correlation_map(principal_components, principal_components)
        corr_map = generate_correlation_map(variables, variables)
        mask = np.zeros_like(corr_map)
        mask[np.triu_indices_from(mask)] = True
        plot_heatmap(corr_map, names, names, title='Gabor parameter correlation', mask=mask)
        mask = np.zeros_like(corr_pca)
        mask[np.triu_indices_from(mask)] = True
        plot_heatmap(corr_pca, names, names, title='PCA correlations', mask=mask)


def compare_two_values():
    # ((-0.5, 1.5), (-np.pi, 2 * np.pi), (-4, 4), (-4, 4), (-3, 3), (-5, 5))
    params = [-0.5, -2 * np.pi / 8, 0, 0, 0, 0]
    params = [0.5, np.pi + ((-2 * np.pi / 8) % np.pi), 0, 0, 0, 0]
    gabor_kernel_3(*params)


if __name__ == '__main__':
    fit_linear_regression()
