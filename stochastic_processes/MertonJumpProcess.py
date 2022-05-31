import numpy as np
import scipy.stats as ss
from tqdm import trange

class MertonJumpProcess():
    """
    Class for the Merton process:
    mu = constant drift coefficient (sometimes referred to as "b" in notes)
    r = risk free constant rate
    sig = constant diffusion coefficient
    lam = jump activity
    muJ = jump mean
    sigJ = jump standard deviation
    risk_neutral = Boolean: True = risk-neutral dynamics
    """ 
    def __init__(self,S0 = 100, mu=0.0,r=0.0,sig=0.2,lam=1.0,muJ=0.0,sigJ=0.2,risk_neutral=True):
        self.S0 = S0
        self.mu = mu
        self.r = r
        self.lam = lam
        self.muJ = muJ
        self.risk_neutral = risk_neutral
        if (sig<0 or sigJ<0):
            raise ValueError("sig and sigJ must be positive")
        else:
            self.sig = sig
            self.sigJ = sigJ
    
    def gen_path(self, length=None, time_step=None, num_paths=None):
        """
        length = T (end of the interval)
        time_step = number of discretization time steps
        num_paths = number of simulated paths
        """
        if self.risk_neutral == True:
            self.mu = self.r - self.lam * (np.exp(self.muJ + (self.sigJ**2)/2)-1) - (self.sig**2)/2
      
        size = (time_step,num_paths)
        dt = length/time_step 
        X = np.zeros((num_paths,time_step+1))
        X[:,0] = np.log(self.S0)
        X_leftlimit = np.zeros((num_paths,time_step+1))
        X_leftlimit[:,0] = np.log(self.S0)
        X_cont = np.zeros((num_paths,time_step+1))
        X_cont[:,0] = np.log(self.S0)
        N_total = ss.poisson.rvs(self.lam*length,size=num_paths)
        Z_total = ss.norm.rvs(0,1,size=(num_paths,time_step))

        X_cont[:, 1:] = np.sqrt(dt)*self.sig*Z_total + self.mu*dt
        X_cont = np.cumsum(X_cont,axis=1) # cumulative sum over rows

        for i in trange(num_paths):
            N = N_total[i]
            U = ss.uniform.rvs(0,length,size=N)
            jumptimes = np.sort(U)
            Y = ss.norm.rvs(self.muJ,self.sigJ,size=N)

            for j in range(time_step):
                t_old = j*dt
                t = (j+1)*dt

                X[i,j+1] = X_cont[i,j+1] + sum((jumptimes <= t)*Y)

                if sum((jumptimes <= t)) > sum((jumptimes <= t_old)):
                    X_leftlimit[i,j+1] = X[i,j]
                else: 
                    X_leftlimit[i,j+1] = X[i,j+1]
      
        S = np.exp(X)
        S_leftlimit = np.exp(X_leftlimit)
        return S, S_leftlimit, N_total, Z_total                  
