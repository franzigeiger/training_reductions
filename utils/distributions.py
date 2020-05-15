import pickle
import warnings
from os import path

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from sklearn import mixture

from base_models import global_data
from base_models.global_data import base_dir
from utils.gabors import gabor_kernel_3, show_kernels, plot_weights


def mixture_gaussian(param, n_samples, components=0, name=None, analyze=False):
    if path.exists(f'{base_dir}/gm_{name}_samples.pkl'):
        best_gmm = load_mixture_gaussian(name)
        if not analyze and global_data.seed == 0:
            print(f'Load samples from file {name}')
            pickle_in = open(f'{base_dir}/gm_{name}_samples.pkl', "rb")
            dict = pickle.load(pickle_in)
            samples = dict['samples']
            if samples.shape[0] == n_samples:
                return samples
            else:
                name = f'{name}_{n_samples}'
                if path.exists(f'{base_dir}/gm_{name}_samples.pkl'):
                    print(f'Load samples from file {name}')
                    pickle_in = open(f'{base_dir}/gm_{name}_samples.pkl', "rb")
                    dict = pickle.load(pickle_in)
                    return dict['samples']
        else:
            print('Load distribution')
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
    if name is not None and not analyze and global_data.seed == 0:
        print(f'Save samples and mixture gaussian in file {name}')
        dict = {'samples': samples}
        pickle_out = open(f'{base_dir}/gm_{name}_samples.pkl', "wb")
        pickle.dump(dict, pickle_out)
        dict = {'comp': components, 'weights': best_gmm.weights_, 'means': best_gmm.means_,
                'cov': best_gmm.covariances_, 'precision': best_gmm.precisions_cholesky_}
        pickle_out = open(f'{base_dir}/gm_{name}_dist.pkl', "wb")
        pickle.dump(dict, pickle_out)
    if analyze:
        centers = best_gmm.means_
        if centers.shape[-1] == 9:
            centers = centers.reshape(centers.shape[0], 3, 3)
            centers = centers.reshape(1, centers.shape[0], 3, 3)
            print(best_gmm.weights_)
            # plot_weights(centers, name)
        # mixture_analysis(best_gmm.weights_, best_gmm.means_, best_gmm.covariances_, name)
        return best_gmm
        # return samples
    return samples


def mixture_analysis(pi, mu, cov, name, gs=None):
    # k components: pi = (k) , mu = (k), cov = (k,k,k)
    print(pi.shape, mu.shape, cov.shape)
    print(mu)
    print(pi)
    if pi.shape[0] == 8:
        filters = mu.reshape((8, 3, 3))
        filters = filters.reshape(4, 2, 3, 3)
        plot_weights(filters, name)
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
        show_kernels(filters, name, gs)


def load_mixture_gaussian(name):
    pickle_in = open(f'{base_dir}/gm_{name}_dist.pkl', "rb")
    GM = pickle.load(pickle_in)
    # dict = {'comp': components,'weights':best_gmm.weights_,'means': best_gmm.means_, 'cov':best_gmm.covariances_}
    gmm = mixture.GaussianMixture(n_components=GM['comp'],
                                  covariance_type='full', max_iter=20000, tol=1e-15, n_init=20)
    gmm.means_ = GM['means']
    gmm.weights_ = GM['weights']
    gmm.covariances_ = GM['cov']
    gmm.precisions_cholesky_ = GM['precision']
    print(f'Layer {name} has {len(gmm.weights_)} components')
    return gmm


def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    # print('Find best fitting distribution')
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        st.alpha
        , st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford,  # st.burr,
        st.cosine, st.dgamma, st.chi, st.chi2, st.cauchy,  # st.dweibull, st.erlang, st.frechet_r, st.frechet_l,
        st.expon, st.exponnorm, st.exponpow,  # st.f, st.fatiguelife,
        st.foldcauchy, st.foldnorm,  # st.genlogistic, st.genpareto, st.gennorm, st.genexpon,
        st.genextreme, st.gamma,  # st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz, st.gumbel_r,
        st.gumbel_l, st.recipinvgauss, st.hypsecant,
        # st.invgamma, st.invgauss,  st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign,
        st.laplace,  # st.levy, st.levy_l, st.gausshyper,st.fisk, st.powerlognorm, st.exponweib,
        # st.levy_stable, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm,
        st.logistic, st.loggamma, st.loglaplace, st.lognorm,  # st.lomax, st.maxwell, st.mielke, st.nakagami, st.ncx2,
        # st.ncf,
        st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powernorm, st.rdist, st.reciprocal,
        st.rayleigh, st.rice, st.t, st.triang,  # st.truncexpon, st.truncnorm,
        # st.tukeylambda,st.semicircular,
        st.uniform,  # st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy
    ]
    #    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:
        # print(f'Fit to distribution {distribution}')
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

                # print(f'Distribution {distribution} params {params}, sse: {sse}')
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


def poisson_sample(data, shape):
    import numpy as np
    from scipy.optimize import minimize
    from scipy import stats
    # def poisson(k, lamb):
    #     """poisson pdf, parameter lamb is the fit parameter"""
    # return (lamb**k/factorial(k)) * np.exp(-lamb)
    #
    #
    # def negative_log_likelihood(params, data):
    # """
    # The negative log-Likelohood-Function
    # """
    #
    # lnl = - np.sum(np.log(poisson(data, params[0])))
    # return lnl

    def negative_log_likelihood(params, data):
        ''' better alternative using scipy '''
        return -stats.poisson.logpmf(data, params[0]).sum()

    result = minimize(negative_log_likelihood,  # function to minimize
                      x0=np.ones(1),  # start value
                      args=(data,),  # additional arguments for function
                      method='Powell',  # minimization method, see docs
                      )
    mu = result.x
    print(f'Fit poisson with mean {mu}')
    return stats.poisson.rvs(size=shape, mu=mu)


def set_running_averages(checkpoint):
    state_dict = checkpoint['state_dict']
    for k, v in state_dict.items():
        if 'running' in k:
            if path.exists(f'{dir}rersources/{k}_weights.pkl'):
                print(f'Load {k} from file {k}')
                pickle_in = open(f'{dir}resources/{k}_weights.pkl', "rb")
                values = pickle.load(pickle_in)
                state_dict[k] = torch.Tensor(values['weights'])
            else:
                weights = v.numpy()
                mu, std = st.norm.fit(weights)
                print(f'{k} Mean {mu} std {std} min {weights.min()} max {weights.max()}, shape {weights.shape}')
                # weights = np.random.normal(mu, std, size=v.shape[0])
                # print(f'Save samples and mixture gaussian in file {k}')
                dict = {'weights': weights}
                pickle_out = open(f'{dir}resources/{k}_weights.pkl', "wb")
                pickle.dump(dict, pickle_out)
                state_dict[k] = torch.Tensor(weights)
        if 'batches_tracked' in k:
            state_dict[k] = torch.tensor(5005, dtype=torch.long)


def delete_running_averages(checkpoint):
    state_dict = checkpoint['state_dict']
    remove = [k for k in state_dict if 'running' in k or 'batches_tracked' in k]
    for k in remove: del state_dict[k]


def set_half_running_averages(checkpoint, config):
    state_dict = checkpoint['state_dict']
    for k, v in state_dict.items():
        if 'running' in k:
            if path.exists(f'{base_dir}resources/{k}_weights.pkl'):
                print(f'Load {k} from file {k}')
                pickle_in = open(f'{base_dir}resources/{k}_weights.pkl', "rb")
                values = pickle.load(pickle_in)
                state_dict[k] = torch.Tensor(values['weights'])
            else:
                weights = v.numpy()
                # mu, std = st.norm.fit(val)
                # weights = np.random.normal(mu, std, size=v.shape[0])
                print(f'Save samples and mixture gaussian in file {k}')
                dict = {'weights': weights}
                pickle_out = open(f'{base_dir}resources/{k}_weights.pkl', "wb")
                pickle.dump(dict, pickle_out)
                state_dict[k] = torch.Tensor(weights)
        # if 'batches_tracked' in k:
        #     state_dict[k] = torch.tensor(5005, dtype=torch.long)
