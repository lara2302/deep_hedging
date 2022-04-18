import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar   

class MertonJumpCall():
    def __init__(self, S=None, sigma=None, T=None, risk_free=None, strike=None,
               jump_intensity=None, jump_mean=None,
               jump_volatility=None):
        self.S = S
        self.sigma = sigma
        self.T = T
        self.risk_free = risk_free
        self.jump_intensity = jump_intensity
        self.jump_volatility = jump_volatility
        self.jump_mean = jump_mean
        self.strike = strike

    def BlackScholesCall(self):
        C = np.zeros(shape=np.shape(self.S))
        num_paths, timesteps = np.shape(self.S)
        payoff_func = lambda x: np.maximum(x - self.strike, 0.0)
        C[:,-1] = payoff_func(self.S[:,-1])
        tau = self.T - np.linspace(0,self.T,timesteps-1,endpoint=False)

        for i in range(num_paths):
            S_path = self.S[i,:-1]
            d1 = (np.log(S_path/self.strike) + (self.risk_free + self.sigma**2/2)*tau) / (self.sigma*np.sqrt(tau))
            d2 = d1 - self.sigma * np.sqrt(tau)
            C[i,:-1] = S_path * norm.cdf(d1) - self.strike * np.exp(-self.risk_free*tau)* norm.cdf(d2)
        return C

    def MertonCall(self):
        C = np.zeros(shape=np.shape(self.S))
        num_paths, timesteps = np.shape(self.S)
        payoff_func = lambda x: np.maximum(x - self.strike, 0.0)
        C[:,-1] = payoff_func(self.S[:,-1])
        tau = self.T - np.linspace(0,self.T,timesteps-1,endpoint=False)
        self.jump_mean = np.exp(self.jump_mean+self.jump_volatility**2*0.5)

        for i in range(num_paths):
            p=0
            S_path = self.S[i:(i+1),:]
            for k in range(18):
                r_k = self.risk_free - self.jump_intensity*(self.jump_mean-1) + (k*np.log(self.jump_mean)) / tau
                sigma_k = np.sqrt(self.sigma**2 + (k*self.jump_volatility** 2) / tau)
                k_fact = np.array(np.math.factorial(k),dtype=float)
                BS_price = BS_CALL(S_path, self.strike, self.T, r_k, sigma_k)
                p += (np.exp(-self.jump_mean*self.jump_intensity*tau)*(self.jump_mean*self.jump_intensity*tau)**k \
                      / (k_fact))*BS_price[0,:-1]
            C[i,:-1] = p  
        return C