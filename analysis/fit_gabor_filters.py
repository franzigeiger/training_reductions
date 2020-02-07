import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy.optimize import minimize, show_options
from skimage.filters import gabor_kernel
from skimage.transform import resize
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import ParameterGrid
from torch import nn

from nets import get_model


def score_kernel(X, theta, frequency, sigma, offset, stds):
    kernel = np.real(gabor_kernel(frequency, theta=theta,
                                  sigma_x=sigma, sigma_y=sigma, n_stds=stds, offset=offset))
    kernel = resize(kernel, (X.shape[0], X.shape[0]),
                    anti_aliasing=True)
    f_min, f_max = kernel.min(), kernel.max()
    kernel = (kernel - f_min) / (f_max - f_min)
    kernel = np.nan_to_num(kernel, posinf=1, neginf=-1, nan=0)
    score = explained_variance_score(kernel, X)

    return score


def objective_function(beta, X):
    kernel = np.real(gabor_kernel(beta[0], theta=beta[1],
                                  sigma_x=beta[2], sigma_y=beta[2], offset=beta[3], n_stds=beta[4]))
    kernel = resize(kernel, (X.shape[0], X.shape[0]),
                    anti_aliasing=True)
    f_min, f_max = kernel.min(), kernel.max()
    kernel = (kernel - f_min) / (f_max - f_min)
    kernel = np.nan_to_num(kernel, posinf=1, neginf=-1, nan=0)
    error = mean_squared_error(kernel, X)
    # print(error)
    return (error)


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
                    f_min, f_max = kernel.min(), kernel.max()
                    kernel = (kernel - f_min) / (f_max - f_min)
                    tuned_params = {"theta": np.arange(0, 1, 0.01),
                                    "frequency": np.arange(0, np.pi, np.pi / 20),
                                    "sigma": np.arange(0, 4, 0.2),
                                    "offset": np.arrange(-1, 1, 0, 2),
                                    "stds": [0, 1, 2, 3, 4]
                                    }
                    best_score = np.NINF
                    best_params = {}
                    for g in ParameterGrid(tuned_params):
                        score = score_kernel(kernel, **g)
                        if score > best_score:
                            best_score = score
                            best_params = g
                    print(f'Best grid:{best_params}')
                    gabor_params[i, j] = np.fromiter(best_params.values(), dtype=float)
            np.save('gabor_params_grid_search.npy', gabor_params)
            return


def fit_gabors():
    model = get_model('CORnet-S_base', True)
    counter = 0
    gabor_params = np.zeros([64, 3, 5])
    np.random.seed(1)
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d and counter == 0:
            weights = m.weight.data.cpu().numpy()
            for i in range(0, 3):
                for j in range(0, 3):
                    kernel = weights[i, j]
                    f_min, f_max = kernel.min(), kernel.max()
                    kernel = (kernel - f_min) / (f_max - f_min)
                    # bnds = ((0, 0.7), (0, np.pi), (1, 3), (-1, 1))
                    # params = np.array([0.5, np.pi/2, 1.5,2,0] )
                    # params = np.random.random(4)
                    params = np.random.random(5)
                    # params[3] = 0
                    # result = minimize(objective_function, params, args=(kernel),
                    #                   method='trust-constr', tol=1e-15, bounds=bnds,
                    #                   options={'maxiter': 40000, 'gtol': 1e-15})
                    # result = minimize(objective_function, params, args=(kernel),
                    #                   method='L-BFGS-B', bounds=bnds, options={'maxiter': 20000, 'dist':True})
                    result = minimize(objective_function, params, args=(kernel),
                                      method='BFGS', options={'maxiter': 200000, 'gtol': 1e-25, 'dist': True})

                    beta_hat = result.x
                    gabor_params[i, j] = beta_hat
                    print(f'Kernel {i}, filter {j}:')
                    print(beta_hat)
            np.save('gabor_params_bfgs.npy', gabor_params)
            return


def compare_gabors():
    file = 'gabor_params_bfgs'
    gabor_params = np.load(f'{file}.npy')
    model = get_model('CORnet-S_base', True)
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
                    f_min, f_max = kernel2.min(), kernel2.max()
                    kernel2 = (kernel2 - f_min) / (f_max - f_min)
                    kernel1 = np.real(gabor_kernel(beta[0], theta=beta[1],
                                                   sigma_x=beta[2], sigma_y=beta[2], offset=beta[3]))
                    # sigma_x=beta[2], sigma_y=beta[2], n_stds=beta[3], offset=beta[4]))
                    kernel1 = np.nan_to_num(kernel1).astype(np.float32)
                    k_resized = resize(kernel1, (kernel2.shape[0], kernel2.shape[0]),
                                       anti_aliasing=True, preserve_range=True)
                    f_min, f_max = k_resized.min(), k_resized.max()
                    k_resized = (k_resized - f_min) / (f_max - f_min)
                    f_min, f_max = kernel1.min(), kernel1.max()
                    kernel1 = (kernel1 - f_min) / (f_max - f_min)
                    kernel1 = np.clip(kernel1, 0, 1)
                    ax = plt.subplot(gs[i * 2, j])
                    ax.set_title(f'K {i}, F {j}', pad=3)
                    plt.imshow(kernel2, cmap='gray')
                    ax = plt.subplot(gs[(i * 2) + 1, j])
                    ax.set_title(f'K {i}, F {j}, fit', pad=3)
                    plt.imshow(k_resized, cmap='gray')
                    index += 1
                index += 3

            plt.tight_layout()
            plt.savefig(f'{file}.png')
            plt.show()
            return


if __name__ == '__main__':
    # hyperparam_gabor()
    # fit_gabors()
    show_options()
    # compare_gabors()
