import numpy as np
import jax
import jax.numpy as jnp

class UKF():
    def __init__(self, f:callable, g:callable, n:int, m:int, Q:np.ndarray, R:np.ndarray, dt:float, λ:int = 2, sqrt_type = 'chol'):
        ''' Build a UKF filter with dynamics function f and measurement function g. '''
        self.f = self.discretize(f,dt)
        self.g = g
        self.n = n # size of state dimension
        self.m = m # size of measurement dimension
        self.λ = λ
        self.x = np.zeros((n,1))
        self.cov = np.zeros((n,n))
        self.Q = Q # noise covariance matrix for state vector
        self.R = R # noise covariance matrix for measurement vector
        self.w = np.zeros((2*n+1,1)) # weight vector
        self.w[0] = λ/(λ+n)
        self.w[1:] = 0.5/(λ+n)
        self.sqrt_type = sqrt_type
        
    def discretize(self,f,dt):
        """ Make discretized dynamics function from discrete function, `f(s,u)` with time
        step `dt` using Runge-Kutta 4th order method."""
        def integrator(s, u, f=f, dt=dt):
            k1 = dt * f(s, u)
            k2 = dt * f(s + k1 / 2, u)
            k3 = dt * f(s + k2 / 2, u)
            k4 = dt * f(s + k3, u)
            return np.squeeze(s) + np.squeeze((k1 + 2 * k2 + 2 * k3 + k4))/ 6
        return integrator
        
    def initialize(self,x0:np.ndarray, cov0:np.ndarray):
        ''' Set the state and covariance for the UKF (initiliazation or re-setting). '''
        self.x = x0
        self.cov = cov0
        
    def step(self,ulast,u,y): 
        ''' Taking in the applied control at the currently estimated step, `ulast`, the control
        to be applied at the next step, `u` , and the measurement of the new step to be 
        estimated, `y`, propagate the mean and covariance forward. '''
        # Predict
        xbar_last = self.sig_point(self.x,self.cov)
        xbar = np.zeros((2*self.n+1,self.n))
        for i in range(2*self.n+1):
            xbar[i] = self.f(xbar_last[i],ulast) # calculate expected transition for each point
            if np.any(np.isnan(xbar[i])):
                print(xbar_last[i])
                print(ulast)
        xp, covp = self.sig_inv(xbar)
        # Update
        xbar_new = self.sig_point(xp, covp)
        ybar = np.zeros((2*self.n+1,self.m))
        for i in range(2*self.n+1):
            ybar[i] = self.g(xbar_new[i],u) # calculate expected measurement for each point
        yhat = np.squeeze(np.sum((self.w@np.ones((1,self.m)))*ybar,axis=0)) # take weighted sum of measurements to get estimate of mean measurement
        covy = self.R
        covxy = np.zeros((self.n,self.m))
        for i in range(2*self.n+1):
            covy = covy + self.w[i]*(ybar[i] - yhat).reshape((-1,1))@(ybar[i]-yhat).reshape((1,-1))
            covxy = covxy + self.w[i]*(xbar_new[i] - xp).reshape((-1,1))@(ybar[i]-yhat).reshape((1,-1))
        invcovy = np.linalg.inv(covy)
        self.x = xp + covxy@invcovy@(y-yhat)
        self.cov = covp - covxy@invcovy@(covxy.T)
        return self.x, self.cov
            
    def sig_point(self,x,cov):
        x= np.squeeze(x)
        xbar = np.zeros((2*self.n+1,self.n))
        xbar[0,:]  = x
        if self.sqrt_type.casefold() == 'axis'.casefold():
            U,S,V = np.linalg.svd(cov)
            M = U@np.diag(np.sqrt(S))@(U.T) # axis-aligned
        elif self.sqrt_type.casefold() == 'ellipse'.casefold():
            U,S,V = np.linalg.svd(cov)
            M = U@np.diag(np.sqrt(S)) # axis-aligned
        else:
            M = np.linalg.cholesky(cov).T
        for i in range(1,2*self.n+1):
            xbar[i] = x + np.squeeze((1-2*((i+1)%2))*np.sqrt(self.n + self.λ)*M[:,-1+(i+1)//2])
        return xbar
            
    def sig_inv(self,xbar):
        x = np.squeeze(np.sum((self.w@np.ones((1,self.n)))*xbar,axis=0)) # take weighted sum of xbar to get new x
        cov = self.Q
        for i in range(2*self.n+1):
            cov = cov + self.w[i]*(xbar[i] - x).reshape((-1,1))@(xbar[i]-x).reshape((1,-1))
        return x,cov
    