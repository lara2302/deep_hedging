import numpy as np

def imp_sampling(impsamp_type=None,Z=None,mu=None,N_jumps=None,lam_new=None,
                 sig=None,risk_free=None,lam=None,muJ=None,sigJ=None,
                 time_step=None,maturity=None,Ktrain=None):
    """
    impsamp_type = Importance sampling type ("normal","poisson","both)
    Z = standard normal samples from simulation
    mu = "new" drift used for "normal" importance sampling 
    N_jumps = poisson samples form simulation
    lam_new = "new" jump intensity used for "poisson" importance sampling
    sig = constant diffusion coefficient
    risk_free = risk free constant rate
    lam = jump activity
    muJ = jump mean
    sigJ = jump standard deviation
    time_step = number of hedging steps
    maturity = maturity of the shorted option
    Ktrain = number of training samples
    """ 
    dt = maturity/time_step
    likelihood = np.zeros(Ktrain)

    if impsamp_type == "normal":
        mu_new = mu*dt*np.ones(time_step)
        mu_true = (risk_free - lam * (np.exp(muJ + (sigJ**2)/2)-1) - \
                   (sig**2)/2)*dt*np.ones(time_step)
        Y_rv = mu*dt + sig*np.sqrt(dt)*Z 
        for i in range(Ktrain):
            likelihood[i] = np.exp((np.transpose(1/(sig**2*dt) * \
                                                 mu_true-1/(sig**2*dt) \
                                                 * mu_new) @ Y_rv[i,:] - \
                                    0.5*(1/(sig**2*dt) * \
                                         np.transpose(mu_true)@mu_true - \
                                         1/(sig**2*dt) * \
                                         np.transpose(mu_new)@mu_new)))  
    if impsamp_type == "poisson":
        for i in range(Ktrain):
            N_i = N_jumps[i]
            likelihood[i] = np.exp((lam_new-lam)*maturity)*(lam/lam_new)**N_i
    if impsamp_type == "both":
        mu_new = mu*dt*np.ones(time_step)
        mu_true = (risk_free - lam * (np.exp(muJ + (sigJ**2)/2)-1) - \
                   (sig**2)/2)*dt*np.ones(time_step)
        Y_rv = mu*dt + sig*np.sqrt(dt)*Z 
        for i in range(Ktrain):
            N_i = N_jumps[i]
            likelihood[i] = np.exp((np.transpose(1/(sig**2*dt) * \
                                                 mu_true-1/(sig**2*dt) \
                                                 * mu_new) @ Y_rv[i,:] - \
                                    0.5*(1/(sig**2*dt) * \
                                         np.transpose(mu_true)@mu_true - \
                                         1/(sig**2*dt) * \
                                         np.transpose(mu_new)@mu_new))) * \
                            np.exp((lam_new-lam)*maturity)*(lam/lam_new)**N_i
    return(likelihood)
