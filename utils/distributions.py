import numpy as np
from sklearn import mixture


def mixture_gaussian(param):
    bic = []
    lowest_bic = np.infty
    components = param.shape[1] if param.shape[1] < 20 else 20
    for n_components in range(1, components):
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type='full', max_iter=20000, tol=1e-15, n_init=20)
        gmm.fit(param)
        bic.append(gmm.bic(param))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            print(f'Lowest bic with number of components {n_components}: {lowest_bic}')
            best_gmm = gmm

    return best_gmm
