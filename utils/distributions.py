import pickle
import warnings
from os import path

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn import mixture

from utils.gabors import gabor_kernel_3, show_kernels, plot_weights

dir = '/braintree/home/fgeiger/weight_initialization/'


def mixture_gaussian(param, n_samples, components=0, name=None, analyze=False):
    if path.exists(f'{dir}/gm_{name}_samples.pkl'):
        if not analyze:
            print(f'Load samples from file {name}')
            pickle_in = open(f'{dir}/gm_{name}_samples.pkl', "rb")
            dict = pickle.load(pickle_in)
            return dict['samples']
        else:
            best_gmm = load_mixture_gaussian(name)
    else:
        bic = []
        lowest_bic = np.infty
        max_components = param.shape[1] if param.shape[1] < 15 else 15
        if components != 0:
            gmm = mixture.GaussianMixture(n_components=components,
                                          covariance_type='full', max_iter=5000, tol=1e-15, n_init=20)
            gmm.fit(param)
            print(f'Lowest bic with number of components {components}: {gmm.bic(param)}')
            best_gmm = gmm
        else:
            for n_components in range(1, max_components):
                # Fit a Gaussian mixture with EM
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type='full', max_iter=5000, tol=1e-15, n_init=20)
                gmm.fit(param)
                bic.append(gmm.bic(param))
                if bic[-1] < lowest_bic:
                    components = n_components
                    lowest_bic = bic[-1]
                    print(f'Lowest bic with number of components {n_components}: {lowest_bic}')
                    best_gmm = gmm
    samples = best_gmm.sample(n_samples)[0]
    if name is not None and not analyze:
        print(f'Save samples and mixture gaussian in file {name}')
        dict = {'samples': samples}
        pickle_out = open(f'{dir}/gm_{name}_samples.pkl', "wb")
        pickle.dump(dict, pickle_out)
        dict = {'comp': components, 'weights': best_gmm.weights_, 'means': best_gmm.means_,
                'cov': best_gmm.covariances_, 'precision': best_gmm.precisions_cholesky_}
        pickle_out = open(f'{dir}/gm_{name}_dist.pkl', "wb")
        pickle.dump(dict, pickle_out)
    mixture_analysis(best_gmm.weights_, best_gmm.means_, best_gmm.covariances_, name)
    return samples


def mixture_analysis(pi, mu, cov, name):
    # k components: pi = (k) , mu = (k), cov = (k,k,k)
    print(pi.shape, mu.shape, cov.shape)
    print(mu)
    print(pi)
    if pi.shape[0] == 8:
        filters = mu.reshape((8, 3, 3))
        filters = filters.reshape(4, 2, 3, 3)
        plot_weights(filters, f'gm_{name}_weights')
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
        show_kernels(filters, f'gm_{name}_gabors')


def load_mixture_gaussian(name):
    pickle_in = open(f'{dir}/gm_{name}_dist.pkl', "rb")
    GM = pickle.load(pickle_in)
    # dict = {'comp': components,'weights':best_gmm.weights_,'means': best_gmm.means_, 'cov':best_gmm.covariances_}
    gmm = mixture.GaussianMixture(n_components=GM['comp'],
                                  covariance_type='full', max_iter=20000, tol=1e-15, n_init=20)
    gmm.means_ = GM['means']
    gmm.weights_ = GM['weights']
    gmm.covariances_ = GM['cov']
    gmm.precisions_cholesky_ = GM['precision']
    return gmm


def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        st.alpha
        , st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2, st.cosine,
        st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f, st.fatiguelife,
        st.fisk,
        st.foldcauchy, st.foldnorm, st.frechet_r, st.frechet_l, st.genlogistic, st.genpareto, st.gennorm, st.genexpon,
        st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz, st.gumbel_r,
        st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma,
        st.invgauss,
        st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign, st.laplace, st.levy, st.levy_l,
        st.levy_stable,
        st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami, st.ncx2,
        st.ncf,
        st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm, st.rdist, st.reciprocal,
        st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang, st.truncexpon, st.truncnorm,
        st.tukeylambda,
        st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy
    ]
    #    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf
