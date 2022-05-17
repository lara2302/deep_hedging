from scipy import sparse
from scipy.sparse.linalg import splu
import scipy as scp
import scipy.stats as ss
from scipy import signal
from scipy.integrate import quad
import scipy.special as scps

from time import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

from math import factorial

class Merton_pricer():
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference PIDE: Explicit-implicit scheme         
    
        0 = dV/dt + (r -(1/2)sig^2 -m) dV/dx + (1/2)sig^2 d^V/dx^2
                 + \int[ V(x+y) nu(dy) ] -(r+lam)V  
    """
    def __init__(self, Option_info, Process_info):
        """
        Process_info:  of type Merton_process. It contains (r, sig, lam, muJ, sigJ) i.e.  
        interest rate, diffusion coefficient, jump activity and jump distribution parameters
    
        Option_info:  of type Option_param. It contains (S0,K,T) i.e. current price, strike, maturity in years
        """
        self.r = Process_info.r           # interest rate
        self.sig = Process_info.sig       # diffusion coefficient
        self.lam = Process_info.lam       # jump activity
        self.muJ = Process_info.muJ       # jump mean
        self.sigJ = Process_info.sigJ     # jump std
        self.exp_RV = Process_info.exp_RV # function to generate exponential Merton Random Variables
        
        self.S0 = Option_info.S0          # current price
        self.K = Option_info.K            # strike
        self.T = Option_info.T            # maturity in years
        
        self.price = 0
        self.S_vec = None
        self.price_vec = None
        self.mesh = None
        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff
             
        
    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum( S - self.K, 0 )
        elif self.payoff == "put":    
            Payoff = np.maximum( self.K - S, 0 )  
        return Payoff     
            
    
    def PIDE_price(self, steps, Time=False):
        """
        steps = tuple with number of space steps and time steps (Nspace,Ntime)
        payoff = "call" or "put"
        exercise = "European" or "American"
        Time = Boolean. Execution time.
        """
        t_init = time()
        
        Nspace = steps[0]   
        Ntime = steps[1]
        self.Nspace = Nspace
        self.Ntime = Ntime
        
        S_max = 6*float(self.K)                
        S_min = float(self.K)/6
        x_max = np.log(S_max)
        x_min = np.log(S_min) 
        
        dev_X = np.sqrt(self.lam * self.sigJ**2 + self.lam * self.muJ**2)    
        # standard deviation of jump (compare variance of compound poisson process)
        
        dx = (x_max - x_min)/(Nspace-1)            # (N-1) since here: x_min = x_1,...,x_N = x_max
        self.dx = dx
        extraP = int(np.floor(5*dev_X/dx))         # extra points beyond the B.C.
        #  Idea: cover at least 5 standard deviations of the domain of the Lévy measure in each direction
        x = np.linspace(x_min-extraP*dx, x_max+extraP*dx, Nspace + 2*extraP)   # space discretization
        t, dt = np.linspace(0, self.T, Ntime, retstep=True)       # time discretization
        self.dt = dt
        
        Payoff = self.payoff_f(np.exp(x)) # Payoff = terminal payoff for all possible x values
        offset = np.zeros(Nspace-2) # offset = middle points
        V = np.zeros((Nspace + 2*extraP, Ntime)) # grid initialization: different to paper, here rows=space and columns=time
        
        if self.payoff == "call":
            V[:,-1] = Payoff # terminal conditions in last time columnm
            V[-extraP-1:,:] = np.exp(x[-extraP-1:]).reshape(extraP+1,1) * np.ones((extraP+1,Ntime)) - \
                 self.K * np.exp(-self.r* t[::-1] ) * np.ones((extraP+1,Ntime))  # boundary condition
            # the last extraP+1 (including A2) rows: (equals x -> infinity) exp(x) - K*exp(-r(T-t)) (t[::-1] = reversed t)
            V[:extraP+1,:] = 0         # the first extraP+1 (including A1) rows: (equals x -> -infinity) 0
        else:    
            V[:,-1] = Payoff
            V[-extraP-1:,:] = 0
            V[:extraP+1,:] = self.K * np.exp(-self.r* t[::-1] ) * np.ones((extraP+1,Ntime))
            # for put: x -> - infinity, V = K*exp(-r*(T-t))
            #          x -> + infinity, V = 0
        
        # nu(dz) = lambda*norm.pdf(mean=muJ,stddev=sigJ)*dz
        # Approximation: nu_k = int_{(k-1/2)dx}^{(k+1/2)dx}nu(z) dz for -K1<=k<=K2
        # -B1 = (-K1-1/2)*dx and -B1 = -(extraP)dx -> extraP = K1 + 1/2
        # [0] because cdf = array([[result]])
        # [-B1,B2] in [(-K1-1/2)*dx,(K2+1/2)*dx] (strict inclusion), therefore: -B1 >= (-K1-1/2)*dx, extraP <= K1+1/2 = extraP + 1.5
        # -B1 = - extraP*dx and B2 = extraP*dx in order to ensure inclusion: +1.5
        cdf = ss.norm.cdf([np.linspace(-(extraP+1+0.5)*dx, (extraP+1+0.5)*dx, 2*(extraP+2))], loc=self.muJ, scale=self.sigJ)[0]
        nu = self.lam * (cdf[1:] - cdf[:-1])      # compute integral via fundamental theorem of calculus (Anti-derivative of pdf is cdf)  
        self.nu = nu
        self.extraP = extraP
        self.dx = dx

        lam_appr = sum(nu)
        # m_approx = sum((exp(y_j)-1)*v_j), @ = matrix multiplication
        m_appr = np.array([ np.exp(i*dx)-1 for i in range(-(extraP+1), extraP+2)]) @ nu
        
        sig2 = self.sig**2 
        dxx = dx**2
        a = ( (dt/2) * ( (self.r -m_appr -0.5*sig2)/dx - sig2/dxx ) )
        b = ( 1 + dt * ( sig2/dxx + self.r + lam_appr) )
        c = (-(dt/2) * ( (self.r -m_appr -0.5*sig2)/dx + sig2/dxx ) )
        
        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace-2, Nspace-2)).tocsc() # only inner points (without boundary)
        # csc = Compressed Sparse Column matrix
        DD = splu(D)         # splu = LU decomposition
        if self.exercise=="European":        
            for i in range(Ntime-2,-1,-1):
                offset[0] = a * V[extraP,i]       # boundary terms
                offset[-1] = c * V[-1-extraP,i]   # boundary terms
                V_jump = V[extraP+1 : -extraP-1, i+1] + dt * signal.convolve(V[:,i+1],nu[::-1],mode="valid",method="fft")
                V[extraP+1 : -extraP-1, i] = DD.solve( V_jump - offset )
        elif self.exercise=="American":
            for i in range(Ntime-2,-1,-1):
                offset[0] = a * V[extraP,i]
                offset[-1] = c * V[-1-extraP,i]
                V_jump = V[extraP+1 : -extraP-1, i+1] + dt * signal.convolve(V[:,i+1],nu[::-1],mode="valid",method="fft")
                V[extraP+1 : -extraP-1, i] = np.maximum( DD.solve( V_jump - offset ), Payoff[extraP+1 : -extraP-1] )
                
        X0 = np.log(self.S0) # current log-price
        self.S_vec = np.exp(x[extraP+1 : -extraP-1]) # vector of S
        self.price = np.interp(X0, x, V[:,0])
        self.price_vec = V[extraP+1 : -extraP-1,0]
        self.price_matrix = V
        self.logS = x
        self.mesh = V[extraP+1 : -extraP-1, :]
        
        if (Time == True):
            elapsed = time()-t_init
            return self.price, elapsed
        else:
            return(self.price,self.price_matrix,self.logS)


    def plot(self, axis=None):
        if (type(self.S_vec) != np.ndarray or type(self.price_vec) != np.ndarray):
            self.PIDE_price((5000,4000))
            
        plt.plot(self.S_vec, self.payoff_f(self.S_vec) , color='blue',label="Payoff")
        plt.plot(self.S_vec, self.price_vec, color='red',label="Merton curve")
        if (type(axis) == list):
            plt.axis(axis)
        plt.xlabel("S"); plt.ylabel("price"); plt.title("Merton price")
        plt.legend(loc='upper left')
        plt.show()
        
    def mesh_plt(self):
        if (type(self.S_vec) != np.ndarray or type(self.mesh) != np.ndarray):
            self.PDE_price((7000,5000))
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid( np.linspace(0, self.T, self.mesh.shape[1]) , self.S_vec)
        ax.plot_surface(Y, X, self.mesh, cmap=cm.ocean)
        ax.set_title("Merton price surface")
        ax.set_xlabel("S"); ax.set_ylabel("t"); ax.set_zlabel("V")
        ax.view_init(30, -100) # this function rotates the 3d plot
        plt.show() 
    
    def get_pos(self,x,S):
        N = S.shape[1] # number of hedging steps
        pos = np.zeros(N)
        S_log = np.log(S)
        for i in range(N):
            pos[i] = np.where(x>=S_log[0,i])[0][0]
        return(pos.astype(int))
    
    def delta_hedge(self,Ssim):
        # Ssim = left limit stock values
        self.Ssim = Ssim # simulated S values
        Nhedge = Ssim.shape[1] # number of hedging steps
        positions = self.get_pos(x=self.logS,S=Ssim)
        phi = np.zeros(Nhedge-1)
        dt_hedge = self.T/(Nhedge-1)
        S_total = np.exp(self.logS)

        for i in range(Nhedge-1):
            get_timecol = int(np.round(i*dt_hedge/self.dt))
            V_t = self.price_matrix[:,get_timecol]
            pos = positions[i]
            phi[i] = (V_t[pos+1] - V_t[pos])/(np.exp(self.dx))
        return(phi)

    def deltagamma_hedge(self,Ssim,x_hedge,Hedgeoption):
        # Ssim = left limit stock values
        self.Ssim = Ssim # simulated S values
        Nhedge = Ssim.shape[1] # number of hedging steps
        positions = self.get_pos(x=self.logS,S=Ssim)
        positions_hedge = self.get_pos(x=x_hedge,S=Ssim)

        phi = np.zeros(Nhedge-1)
        gamma = np.zeros(Nhedge-1)
        phi_hedgeoption = np.zeros(Nhedge-1)
        delta = np.zeros(Nhedge-1)

        dt_hedge = self.T/(Nhedge-1)
        S_total = np.exp(self.logS)
        S_total_hedge = np.exp(x_hedge)

        for i in range(Nhedge-1):
            get_timecol = int(np.round(i*dt_hedge/self.dt))
            V_t = self.price_matrix[:,get_timecol]
            VH_t = Hedgeoption[:,get_timecol]
            pos = positions[i]
            pos_h = positions_hedge[i]
            phi[i] = (V_t[pos+1] - V_t[pos])/(np.exp(self.dx))
            gamma[i] = (V_t[pos+1] - 2*V_t[pos] + V_t[pos-1])/(VH_t[pos_h+1] - 2*VH_t[pos_h] + VH_t[pos_h-1])
            if np.abs(gamma[i])>=1.5:
                gamma[i] = 1
            phi_hedgeoption[i] = (VH_t[pos_h+1] - VH_t[pos_h])/(np.exp(self.dx))
        
        delta = phi - gamma*phi_hedgeoption
        return(delta,gamma)

    def quadratic_multi(self,Ssim,x_hedge,Hedgeoption):
        # Ssim = left limit stock values
        self.Ssim = Ssim # simulated S values
        Nhedge = Ssim.shape[1] # number of hedging steps
        dt_hedge = self.T/(Nhedge-1)
        positions = self.get_pos(x=self.logS,S=Ssim)
        positions_hedge = self.get_pos(x=x_hedge,S=Ssim)

        phi = np.zeros(Nhedge-1) # dC/dS
        phi_hedgeoption = np.zeros(Nhedge-1) # dF/dS
        quad = np.zeros((Nhedge-1,2)) # col 1 = stock units, col 2 = option units

        for i in range(Nhedge-1):
            get_timecol = int(np.round(i*dt_hedge/self.dt))
            V_t = self.price_matrix[:,get_timecol]
            VH_t = Hedgeoption[:,get_timecol]
            pos = positions[i]
            pos_h = positions_hedge[i]
            phi[i] = (V_t[pos+1] - V_t[pos])/(np.exp(self.dx))
            phi_hedgeoption[i] = (VH_t[pos_h+1] - VH_t[pos_h])/(np.exp(self.dx))
            S_current = Ssim[0,i]

            sigma_t = np.array([S_current*self.sig, phi_hedgeoption[i]*S_current*self.sig])
            Msig = np.outer(sigma_t,sigma_t)

            a = (S_current*(np.exp(np.array(range(-(self.extraP+1), self.extraP+2))*self.dx)-1))**2 @ self.nu
            VH_help = VH_t[pos_h + np.arange(-(self.extraP+1), self.extraP+2)] - VH_t[pos_h]
            b_help = S_current*(np.exp(np.array(range(-(self.extraP+1), self.extraP+2))*self.dx)-1)
            b = (VH_help*b_help) @ self.nu
            c = (VH_help**2) @ self.nu

            Mgam = np.matrix([[a,b],[b,c]])
            Mmat = Msig + Mgam
            Minv = np.linalg.inv(Mmat)

            sigma0 = phi[i]*S_current*self.sig
            V_help = V_t[pos + np.arange(-(self.extraP+1), self.extraP+2)] - V_t[pos]
            d = (V_help*b_help) @ self.nu
            e = (V_help*VH_help) @ self.nu
            gamma0 = np.array([d,e])

            quad[i,:] = Minv @ (sigma0*sigma_t + gamma0)
        return quad

    def quadratic_simple(self,Ssim):
        # Ssim = left limit stock values
        self.Ssim = Ssim # simulated S values
        Nhedge = Ssim.shape[1] # number of hedging steps
        dt_hedge = self.T/(Nhedge-1)
        positions = self.get_pos(x=self.logS,S=Ssim)

        phi = np.zeros(Nhedge-1) # dC/dS
        quad = np.zeros(Nhedge-1) #  stock units

        for i in range(Nhedge-1):
            get_timecol = int(np.round(i*dt_hedge/self.dt))
            V_t = self.price_matrix[:,get_timecol]
            pos = positions[i]
            phi[i] = (V_t[pos+1] - V_t[pos])/(np.exp(self.dx))
            S_current = Ssim[0,i]

            denom = self.sig**2 + (np.exp(np.array(range(-(self.extraP+1), self.extraP+2))*self.dx)-1)**2 @ self.nu
            V_help = V_t[pos + np.arange(-(self.extraP+1), self.extraP+2)] - V_t[pos]
            b_help = np.exp(np.array(range(-(self.extraP+1), self.extraP+2))*self.dx)-1
            num = self.sig**2 * phi[i] + 1/S_current * ((V_help * b_help) @ self.nu)

            quad[i] = num/denom
        return quad