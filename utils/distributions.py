import numpy as np
from sklearn import mixture

from utils.gabors import gabor_kernel_3, show_kernels, plot_conv_weights


def mixture_gaussian(param, n_samples, components=0):
    bic = []
    lowest_bic = np.infty
    max_components = param.shape[1] if param.shape[1] < 15 else 15
    if components != 0:
        gmm = mixture.GaussianMixture(n_components=components,
                                      covariance_type='full', max_iter=20000, tol=1e-15, n_init=20)
        gmm.fit(param)
        print(f'Lowest bic with number of components {components}: {gmm.bic(param)}')
        best_gmm = gmm
    else:
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
    mixture_analysis(best_gmm.weights_, best_gmm.means_, best_gmm.covariances_)
    return best_gmm.sample(n_samples)[0]


def mixture_analysis(pi, mu, cov):
    # k components: pi = (k) , mu = (k), cov = (k,k,k)
    print(pi.shape, mu.shape, cov.shape)
    print(mu)
    print(pi)
    if pi.shape[0] == 8:
        filters = mu.reshape((8, 3, 3))
        plot_conv_weights(filters, 'distribution_init_kernel')
    else:
        filters = np.zeros((4, 3, 7, 7))
        for i in range(4):
            for s, e in ((0, 10), (10, 20), (20, 30)):
                beta = mu[i, s:e]
                filter = gabor_kernel_3(beta[0], theta=np.arctan2(beta[1], beta[2]) / 2,
                                        sigma_x=beta[3], sigma_y=beta[4], offset=np.arctan2(beta[5], beta[6]),
                                        x_c=beta[7],
                                        y_c=beta[8],
                                        scale=beta[9], ks=7)
                filters[i, int(s / 10)] = filter
        show_kernels(filters, 'distribution_init_channel_color')
