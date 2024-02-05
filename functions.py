## Import Packages
import pandas as pd
import numpy as np
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import multivariate_normal
import scipy.integrate as integr

########################  Shifted Gamma Distribution Function ##########################

#  Shifted gamma CDF
def shift_gamma_cdf(x, a, t):
    y = 1 - gamma.cdf(np.sqrt(a) * t - x, a * t, scale=1 / np.sqrt(a))
    return y

#  Shifted gamma PDF
def shift_gamma_pdf(x, a, cor):
    return gamma.pdf(np.sqrt(a) * cor - x, a * cor, scale=1 / np.sqrt(a))


#  Shifted gamma inverse CDF
def shift_gamma_ppf(y, a, t):
    x = np.sqrt(a) * t - gamma.ppf(1 - y, a * t, scale=1 / np.sqrt(a))
    return x


########################  Shifted Gamma LHP  ##########################
#  Shifted gamma Large Homogenous Portfolio Loss CDF
def shift_gamma_lhp_loss_cdf(x, a, cor, pd):
    diff = shift_gamma_ppf(pd, a, 1) - shift_gamma_ppf(x, a, 1 - cor)
    y = 1 - shift_gamma_cdf(diff, a, cor)
    #  Jump at x == 1
    y = y + (x == 1) * shift_gamma_cdf(
        shift_gamma_ppf(pd, a, 1) - np.sqrt(a) * (1 - cor), a, cor
    )
    return y


#  Shifted gamma Large Homogenous Portfolio Loss PDF
def shift_gamma_lhp_loss_pdf(x, a, cor, p):
    p1 = shift_gamma_pdf(
        shift_gamma_ppf(p, a, 1) - shift_gamma_ppf(x, a, 1 - cor), a, cor
    )
    p2 = shift_gamma_pdf(shift_gamma_ppf(x, a, 1 - cor), a, 1 - cor)
    return p1 / p2


#  Shifted gamma Large Homogenous Portfolio Loss inverse CDF
def shift_gamma_lhp_loss_ppf(y, a, cor, p):
    return 1 - shift_gamma_lhp_loss_cdf(1 - y, a, 1 - cor, p)


#  Shifted gamma Tranched Large Homogenous Portfolio Loss CDF
def shift_gamma_lhp_tloss_cdf(x, a, cor, pd, k1, k2):
    p1 = shift_gamma_lhp_loss_cdf(x * k2 + (1 - x) * k1, a, cor, pd)
    p2 = shift_gamma_lhp_loss_cdf(k2, a, cor, pd)
    y = p1 + (x == 1) * (1 - p2)
    return y

#  Shifted gamma Tranched Large Homogenous Portfolio Loss PDF
def shift_gamma_lhp_tloss_pdf(x, a, cor, p, k1, k2):
    p1 = (x == 0) * shift_gamma_lhp_loss_cdf(k1, a, cor, p)
    p2 = (
        (k2 - k1)
        * shift_gamma_lhp_loss_cdf(x * k2 + (1 - x) * k1, a, cor, p)
        * (
            shift_gamma_lhp_loss_cdf(k2, a, cor, p)
            - shift_gamma_lhp_loss_cdf(k1, a, cor, p)
        )
    )
    p3 = (x == 1) * (1 - shift_gamma_lhp_loss_cdf(k2, a, cor, p))
    return p1 + p2 + p3


#  Shifted gamma Tranched Large Homogenous Portfolio Loss Inverse CDF
def shift_gamma_lhp_tloss_ppf(y, a, cor, p, k1, k2):
    quantile = shift_gamma_lhp_loss_ppf(y, a, cor, p)
    return np.minimum(np.maximum(quantile - k1, 0), k2 - k1) / (k2 - k1)


#  Shifted gamma conditional PD
def cond_pd_sg(syst_factor, a, pd, cor):
    threshhold = shift_gamma_ppf(pd, a, 1)
    y = shift_gamma_cdf(threshhold - syst_factor, a, 1 - cor)
    return y


#  Shifted gamma conditional EL for inhomogenous portfolio
def cond_pd_sg_inhom(x, a, pds, exp_weights, cor):
    pds = np.squeeze(pds)
    eval = np.inner(cond_pd_sg(x, a, pds, cor), exp_weights)
    return eval


#  Shifted gamma inverse of conditional PD
def cond_pd_sg_inv(p, a, pd, cor):
    threshhold = shift_gamma_ppf(pd, a, 1)
    y = (
        threshhold
        - (p == 1) * np.sqrt(a) * (1 - cor)
        - (p != 1) * shift_gamma_ppf(p, a, 1 - cor)
    )
    return y


#  Shifted gamma tranched LHP expectation
def tranched_lhp_expect_sg_hom(a, pd, cor, k1, k2):
    tranche_loss_mean = integr.quad(
        lambda x: (1 - shift_gamma_lhp_tloss_cdf(x, a, cor, pd, k1, k2)), 0, 1
    , epsabs = 0.00001)
    return tranche_loss_mean[0]



#  Shifted gamma tranched inhomogenous LHP expectation
def tranched_lhp_expect_sg_inhom(a, pds, exp_weights, cor, xc, xd, c, d):
    pds = np.squeeze(pds)
    threshholds = shift_gamma_ppf(pds, a, 1)
    integrals = np.zeros(len(pds))
    i=0
    for K in threshholds:
        xd_int = max(xd, K-np.sqrt(a)*(1-cor))
        intgr = integr.quad(
        lambda x: shift_gamma_pdf(K-x,a,1-cor)*(1-shift_gamma_cdf(x,a,cor)), xd_int, xc 
        , epsabs = 0.00001)
        integrals[i] = exp_weights[i]*intgr[0]
        i+=1
    return 1  - integrals.sum()/(d-c)


########################  Gaussian LHP  ##########################
#  Gaussian conditional PD for homogenous portfolio
def cond_pd_norm(syst_factor, pd, cor):
    threshhold = norm.ppf(pd)
    y = norm.cdf((threshhold - syst_factor)/np.sqrt(1-cor), 0, 1)
    return y


#  Gaussian conditional EL for inhomogenous portfolio
def cond_el_norm_inhom(x, pds, exp_weights, cor):
    pds = np.squeeze(pds)
    threshholds = norm.ppf(pds)
    eval = np.inner(norm.cdf((threshholds - x)/np.sqrt(1-cor)), exp_weights)
    return eval


#  Gaussian tranched LHP expectation for homogenous portfolios
def tranched_lhp_expect_norm_hom(pd, cor, c):
    mean = np.array([0, 0])  # Mean vector
    covariance = np.array([[1, -np.sqrt(1-cor)], [-np.sqrt(1-cor), 1]])  # Covariance matrix
    bivariate_normal = multivariate_normal(mean=mean, cov=covariance)
    eval_point = np.array([-norm.ppf(c), norm.ppf(pd)])     
    return bivariate_normal.cdf(eval_point)/(1 - c)


#  Gaussian tranched LHP expectation for inhomogenous portfolios
def tranched_lhp_expect_norm_inhom(pds, exp_weights, cor, xc, xd, c, d):
    pds = np.squeeze(pds)
    threshholds = norm.ppf(pds)
    integrals = np.zeros(len(pds))
    i=0
    for K in threshholds:
        integral = integr.quad(
        lambda x: norm.pdf((K - x)/np.sqrt(1-cor))*(1-norm.cdf(x/np.sqrt(cor)))/np.sqrt(1-cor), xd, xc 
        )
        integrals[i] = exp_weights[i]*integral[0]
        i+=1
    return 1  - integrals.sum()/(d-c)


########################  Simulate Losses  ##########################
#  Generate Gaussian factors and asset values
def sim_av_gauss(corr, port_size, nr_of_sims):
    if corr == 0:
        syst_var = np.zeros((1, nr_of_sims))
        idio_var = norm.rvs(0, 1, (port_size, nr_of_sims))
    elif corr == 1:
        syst_var = norm.rvs(0, 1, (1, nr_of_sims))
        idio_var = np.zeros((port_size, nr_of_sims))
    else:
        syst_var = np.sqrt(corr) * norm.rvs(0, 1, (1, nr_of_sims))
        idio_var = np.sqrt(1 - corr) * norm.rvs(0, 1, (port_size, nr_of_sims))
    av = syst_var + idio_var
    return syst_var, idio_var, av


#  Generate shifted gamma factors and asset values
def sim_av_shifted_gamma(a, corr, port_size, nr_of_sims):
    shape = a
    scale = 1 / np.sqrt(a)
    if corr == 0:
        syst_var = np.zeros((1, nr_of_sims))
        idio_var = np.sqrt(a) - gamma.rvs(shape, scale=scale, size=(port_size, nr_of_sims))
    elif corr == 1:
        syst_var = np.sqrt(a) - gamma.rvs(shape, scale=scale, size=(1, nr_of_sims))
        idio_var = np.zeros((port_size, nr_of_sims))
    else:
        syst_var = np.sqrt(a) * corr - gamma.rvs(shape * corr, scale=scale, size=(1, nr_of_sims))
        idio_var = np.sqrt(a) * (1 - corr) - gamma.rvs(
            shape * (1 - corr), scale=scale, size=(port_size, nr_of_sims)
        )
    av = syst_var + idio_var
    return syst_var, idio_var, av


def calc_losses_norm(av, def_prob, exp_weights):
    threshholds = np.array(norm.ppf(def_prob))
    defaults = (av.T < threshholds).T
    losses = np.array(np.sum((defaults.T * exp_weights).T, axis=0))
    return losses

def calc_losses_sg(av, a, def_prob, exp_weights):
    threshholds = np.array(shift_gamma_ppf(def_prob, a, 1))
    defaults = (av.T < threshholds).T
    losses = np.array(np.sum((defaults.T * exp_weights).T, axis=0))
    return losses


#  Tranching transformation
def calc_tranche_losses(losses, k1, k2):
    return (np.fmin(losses, k2) - np.fmin(losses, k1)) / (k2 - k1)


########################  Monte Carlo Simulation  ##########################
#  Produces estimates for the expected tranche loss for Gaussian factors and a homogenous portfolio
def produce_estimates_gauss_hom(nr_of_sims, port_size, pds, exp_weights, cor, c, lhp_tranche_mean_exact):
    s, i, av = sim_av_gauss(cor, port_size, nr_of_sims)
    losses = calc_losses_norm(av, pds, exp_weights)
    tranche_losses = calc_tranche_losses(losses, c, 1)
    mc_est = tranche_losses.mean()
    lhp_losses = np.sum(cond_pd_norm(s, pds.T, cor), axis=0) / port_size
    lhp_tranche_losses = calc_tranche_losses(lhp_losses, c, 1)
    cov = np.cov(tranche_losses, lhp_tranche_losses)
    if cov[0,0]!=0 and cov[1,1]!=0:    
        est_corr = cov[0,1] / np.sqrt(cov[0,0]*cov[1,1])
    else:
        est_corr = 0
    if cov[1,1] != 0:
        opt_beta = cov[0,1] / cov[1,1]
    else:
        opt_beta = 0
    mc_cv_est = mc_est - opt_beta*(lhp_tranche_losses.mean() - lhp_tranche_mean_exact)
    return mc_est, mc_cv_est, est_corr, tranche_losses.std()/np.sqrt(nr_of_sims)


#  Produces estimates for the expected tranche loss for Gaussian factors and an inhomogenous portfolio
def produce_estimates_gauss_inhom(nr_of_sims, port_size, pds, exp_weights, cor, c, d, xc, xd):
    pd_mean = np.inner(pds, exp_weights)[0]
    
    # Simulate (lhp) tranche losses
    s, i, av = sim_av_gauss(cor, port_size, nr_of_sims)
    losses = calc_losses_norm(av, pds, exp_weights)
    tranche_losses = calc_tranche_losses(losses, c, d)
    lhp_losses_inhom = (cond_pd_norm(s, pds.T, cor) * np.reshape(exp_weights, (port_size, 1))).sum(axis=0)
    lhp_losses_hom = np.squeeze(cond_pd_norm(s, pd_mean, cor))  
    lhp_tranche_losses_inhom = calc_tranche_losses(lhp_losses_inhom, c, d)  # lhp tranche random variable
    lhp_tranche_losses_hom = calc_tranche_losses(lhp_losses_hom, c, d)  # lhp tranche random variable

    # Calculate exact tranche loss means
    lhp_tranche_losses_exact_mean_inhom = tranched_lhp_expect_norm_inhom(pds, exp_weights, cor, xc, xd, c, d)
    lhp_tranche_losses_exact_mean_hom = tranched_lhp_expect_norm_hom(pd_mean, cor, c)


    # Compute optimal control variate coefficient
    cov_inhom = np.cov(tranche_losses, lhp_tranche_losses_inhom)
    cov_hom = np.cov(tranche_losses, lhp_tranche_losses_hom)
    
    if cov_inhom[0,0]!=0 and cov_inhom[1,1]!=0:    
        est_corr_inhom = cov_inhom[0,1] / np.sqrt(cov_inhom[0,0]*cov_inhom[1,1])
    else:
        est_corr_inhom = 0
    if cov_inhom[1,1] != 0:
        opt_beta_inhom = cov_inhom[0,1] / cov_inhom[1,1]
    else:
        opt_beta_inhom = 0
    if cov_hom[0,0]!=0 and cov_hom[1,1]!=0:    
        est_corr_hom = cov_hom[0,1] / np.sqrt(cov_hom[0,0]*cov_hom[1,1])
    else:
        est_corr_hom = 0
    if cov_hom[1,1] != 0:
        opt_beta_hom = cov_hom[0,1] / cov_hom[1,1]
    else:
        opt_beta_hom = 0

    cv_adjusted_tranche_losses_inhom = tranche_losses - opt_beta_inhom*(lhp_tranche_losses_inhom - lhp_tranche_losses_exact_mean_inhom)
    cv_adjusted_tranche_losses_hom = tranche_losses - opt_beta_hom*(lhp_tranche_losses_hom - lhp_tranche_losses_exact_mean_hom)

    return tranche_losses, cv_adjusted_tranche_losses_inhom, cv_adjusted_tranche_losses_hom, est_corr_inhom, est_corr_hom, lhp_tranche_losses_exact_mean_inhom, lhp_tranche_losses_exact_mean_hom


#  Produces estimates for the expected tranche loss for shifted gamma factors and a homogenous portfolio
def produce_estimates_sg_hom(nr_of_sims, port_size, a, pds, exp_weights, cor, c, lhp_tranche_mean_exact):
    s, i, av = sim_av_shifted_gamma(a, cor, port_size, nr_of_sims)
    losses = calc_losses_sg(av, a, pds, exp_weights)
    tranche_losses = calc_tranche_losses(losses, c, 1)
    mc_est = tranche_losses.mean()
    lhp_losses = cond_pd_sg(s, a, pds.T, cor)
    lhp_tranche_losses = calc_tranche_losses(lhp_losses, c, 1)
    cov = np.cov(tranche_losses, lhp_tranche_losses)
    if cov[0,0]!=0 and cov[1,1]!=0:    
        est_corr = cov[0,1] / np.sqrt(cov[0,0]*cov[1,1])
    else:
        est_corr = 0
    if cov[1,1] != 0:
        opt_beta = cov[0,1] / cov[1,1]
    else:
        opt_beta = 0
    mc_cv_est = mc_est - opt_beta*(lhp_tranche_losses.mean() - lhp_tranche_mean_exact)
    return mc_est, mc_cv_est, est_corr, tranche_losses.std()/np.sqrt(nr_of_sims)


#  Produces estimates for the expected tranche loss for shifted gamma factors and an inhomogenous portfolio
def produce_estimates_sg_inhom(nr_of_sims, port_size, a, pds, exp_weights, cor, c, d, xc, xd):
    pd_mean = np.inner(pds, exp_weights)[0]
    c  = cond_pd_sg_inhom(xc, a, pds, exp_weights, cor)
    d=1
    
    # Simulate (lhp) tranche losses
    s, _, av = sim_av_shifted_gamma(a, cor, port_size, nr_of_sims)
    losses = calc_losses_sg(av, a, pds, exp_weights)
    tranche_losses = calc_tranche_losses(losses, c, d)
    lhp_losses_inhom = (cond_pd_sg(s, a, pds.T, cor) * np.reshape(exp_weights, (port_size, 1))).sum(axis=0)
    lhp_losses_hom = np.squeeze(cond_pd_sg(s, a, pd_mean, cor))  
    lhp_tranche_losses_inhom = calc_tranche_losses(lhp_losses_inhom, c, d)  # lhp tranche random variable
    lhp_tranche_losses_hom = calc_tranche_losses(lhp_losses_hom, c, d)  # lhp tranche random variable

    # Calculate exact tranche loss means
    lhp_tranche_losses_exact_mean_inhom = tranched_lhp_expect_sg_inhom(a, pds, exp_weights, cor, xc, xd, c, d)
    lhp_tranche_losses_exact_mean_hom = tranched_lhp_expect_sg_hom(a, pd_mean, cor, c, d)


    # Compute optimal control variate coefficient
    cov_inhom = np.cov(tranche_losses, lhp_tranche_losses_inhom)
    cov_hom = np.cov(tranche_losses, lhp_tranche_losses_hom)
    
    if cov_inhom[0,0]!=0 and cov_inhom[1,1]!=0:    
        est_corr_inhom = cov_inhom[0,1] / np.sqrt(cov_inhom[0,0]*cov_inhom[1,1])
    else:
        est_corr_inhom = 0
    if cov_inhom[1,1] != 0:
        opt_beta_inhom = cov_inhom[0,1] / cov_inhom[1,1]
    else:
        opt_beta_inhom = 0
    if cov_hom[0,0]!=0 and cov_hom[1,1]!=0:    
        est_corr_hom = cov_hom[0,1] / np.sqrt(cov_hom[0,0]*cov_hom[1,1])
    else:
        est_corr_hom = 0
    if cov_hom[1,1] != 0:
        opt_beta_hom = cov_hom[0,1] / cov_hom[1,1]
    else:
        opt_beta_hom = 0

    cv_adjusted_tranche_losses_inhom = tranche_losses - opt_beta_inhom*(lhp_tranche_losses_inhom - lhp_tranche_losses_exact_mean_inhom)
    cv_adjusted_tranche_losses_hom = tranche_losses - opt_beta_hom*(lhp_tranche_losses_hom - lhp_tranche_losses_exact_mean_hom)

    return tranche_losses, cv_adjusted_tranche_losses_inhom, cv_adjusted_tranche_losses_hom, est_corr_inhom, est_corr_hom, lhp_tranche_losses_exact_mean_inhom, lhp_tranche_losses_exact_mean_hom

