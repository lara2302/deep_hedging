import scipy.stats as ss
import numpy as np

class Merton_process():
    """
    Class for the Merton process:
    r = risk free constant rate
    sig = constant diffusion coefficient
    lam = jump activity
    muJ = jump mean
    sigJ = jump standard deviation
    """
    def __init__(self, r, sig, lam , muJ, sigJ):
        self.r = r
        self.lam = lam
        self.muJ = muJ
        if (sig<0 or sigJ<0):
            raise ValueError("sig and sigJ must be positive")
        else:
            self.sig = sig
            self.sigJ = sigJ
        
        # moments
        self.var = self.sig**2 + self.lam * self.sigJ**2 + self.lam * self.muJ**2
        self.skew = self.lam * (3* self.sigJ**2 * self.muJ + self.muJ**3) / self.var**(1.5)
        self.kurt = self.lam * (3* self.sigJ**3 + 6 * self.sigJ**2 * self.muJ**2 + self.muJ**4) / self.var**2
     
    def exp_RV(self, S0, T, N):
        m = self.lam * (np.exp(self.muJ + (self.sigJ**2)/2) -1)    # coefficient m
        W = ss.norm.rvs(0, 1, N)              # The normal RV vector  
        P = ss.poisson.rvs(self.lam*T, size=N)    # Poisson random vector (number of jumps)
        Jumps = np.asarray([ss.norm.rvs(self.muJ, self.sigJ, ind).sum() for ind in P ]) # Jumps vector
        S_T = S0 * np.exp( (self.r - 0.5*self.sig**2 -m )*T + np.sqrt(T)*self.sig*W + Jumps )     # Martingale exponential Merton
        return S_T.reshape((N,1))