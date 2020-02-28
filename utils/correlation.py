import numpy as np
from scipy.stats import pearsonr
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def generate_correlation_map(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                   mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    # result /= result[result.argmax()]
    return result[result.size // 2:]


def autocorr2(x):
    result = np.convolve(x, x, mode='full')
    result = result[result.size // 2:]
    dr = 2 * ((result + .25) / (0.5)) - 1.0
    dr = np.clip(dr, -1.0, 1.0)
    return dr


def auto_correlation(x):
    y = np.zeros(x.shape)
    for n in range(x.shape[0]):
        y[n] = autocorr2(x[n])
    return y


def test_generate_correlation_map():
    x = np.random.randint(51, 100, 49).reshape(7, 7)
    y = np.random.randint(51, 100, 49).reshape(7, 7)
    desired = np.empty((7, 7))
    for n in range(x.shape[0]):
        for m in range(y.shape[0]):
            desired[n, m] = pearsonr(x[n, :], y[m, :])[0]
    actual = generate_correlation_map(x, y)
    print(actual)
    print(desired)
    np.testing.assert_array_almost_equal(actual, desired)


def multivariate_gaussian(values, samples):
    means = np.zeros(values.shape[0])
    for i in range(values.shape[0]):
        means[i] = np.mean(values[i])
    covariance = np.cov(values)
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    return np.random.multivariate_normal(means, covariance, samples).T
    # shape: (3,samples)


def gaussian_mixture(values):
    return


def pca(variables, n_components=3):
    pca = PCA(n_components=n_components)
    pca.fit(variables)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    print(pca.components_)
    return pca


def fit_data(X, y):
    return LinearRegression().fit(X, y)


def kernel_convolution(x, y, stride=2, size=3):
    middle = int(size / 2)
    filter = np.zeros((3, 3))
    for i in range(size):
        for j in range(size):
            x_crop = (middle - i) * stride
            y_crop = (middle - j) * stride
            if x_crop <= 0 and y_crop <= 0:
                y_crop = x.shape[0] + y_crop
                x_crop = x.shape[0] + x_crop
                crop_y = y[:x_crop, :y_crop]
                crop_x = x[(x_crop) * -1:, (y_crop) * -1:]
            elif y_crop <= 0:
                y_crop = x.shape[0] + y_crop
                crop_y = y[x_crop:, :y_crop]
                crop_x = x[:x_crop * -1, y_crop * -1:]
            elif x_crop <= 0:
                x_crop = x.shape[0] + x_crop
                crop_y = y[:x_crop, y_crop:]
                crop_x = x[x_crop * -1:, :y_crop * -1]
            else:
                crop_y = y[x_crop:, y_crop:]
                crop_x = x[:x_crop * -1, :y_crop * -1]
            result = pearsonr(crop_x.flatten(), crop_y.flatten())
            filter[i, j] = result[0]
    return filter


def mixture_gaussian(param, n_samples):
    bic = []
    lowest_bic = np.infty
    max_components = param.shape[1] if param.shape[1] < 15 else 15
    for n_components in range(1, max_components):
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type='full', max_iter=20000, tol=1e-15, n_init=20)
        gmm.fit(param)
        bic.append(gmm.bic(param))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            print(f'Lowest bic with number of components {n_components}: {lowest_bic}')
            best_gmm = gmm
    # plot_bic(best_gmm, param)
    return best_gmm.sample(n_samples)[0]


if __name__ == '__main__':
    a = np.random.randint(51, 100, 49).reshape(7, 7)
    b = np.random.randint(51, 100, 49).reshape(7, 7)
    kernel_convolution(a, b)
