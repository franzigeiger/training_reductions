import numpy as np


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


from scipy.stats import pearsonr


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


# def autocorr5(x,lags):
#     '''numpy.correlate, non partial'''
#     mean=x.mean()
#     var=x.var()
#     xp=x-mean
#     corr=np.correlate(xp,xp,'full')[len(x)-1:]/var/len(x)
#
#     return corr[:len(lags)]

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


if __name__ == '__main__':
    # a = np.random.randint(51,100,49).reshape(7,7)
    # b = np.random.randint(51,100,49).reshape(7,7)
    test_generate_correlation_map()
