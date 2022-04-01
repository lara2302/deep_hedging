import numpy as np

# References:
# https://www.codearmo.com/python-tutorial/merton-jump-diffusion-model-python

# Assigned seed for testing. Set to 0 for random seeds.

# Merton Jump Diffusion Process.
class MertonJumpProcess:
    def __init__(self, s0=None, sigma=None, risk_free=None,
                 dividend=None, day_count=None, jump_intensity=None, jump_mean=None,
                 jump_volatility=None, seed=0):
        self.s0 = s0
        self.sigma = sigma
        self.risk_free = risk_free
        self.dividend = dividend
        self.day_count = day_count
        self.seed = seed
        self.jump_intensity = jump_intensity
        self.jump_volatility = jump_volatility
        self.jump_mean = jump_mean

    def gen_paths(self, length=None, time_step=None, num_paths=None):
        size = (time_step, num_paths)
        dt = length/time_step
        
        S = np.zeros((time_step+1, num_paths))
        S[0,:] = self.s0
        poi_rv = np.multiply(np.random.poisson(self.jump_intensity*dt, size=size),
                             np.random.normal(self.jump_mean, self.jump_volatility, size=size)).cumsum(axis=0)
        geo = np.cumsum(((self.risk_free - self.sigma**2/2 - \
                        self.jump_intensity*(self.jump_mean + self.jump_volatility**2*0.5))*dt + \
                        self.sigma*np.sqrt(dt) * np.random.normal(size=size)), axis=0)
        S[1:time_step+1,:] = np.exp(geo+poi_rv)*self.s0
        
        return np.tranpose(S)                               
