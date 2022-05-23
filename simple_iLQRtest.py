import iLQR
import importlib
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

importlib.reload(iLQR)

rng = np.random.default_rng(12345)

# Simple test
n, m = 2, 1
A = np.array([[-0.9, 0.6], [0, -0.9]])
B = np.array([[0.], [1.]])
Q = np.eye(n)
R = np.eye(m)
umax  = 1
u_range = umax*np.array([[-1,1]])
dt = 0.1 # discrete time resolution
T = 10
t = np.arange(0., T + dt, dt)
N = t.size - 1
s0 = np.array([0, 50])

f = lambda s,u: A@s + B@u
c = lambda s,u,t: 0.5*s.T@Q@s + 0.5*u@R@u # cost function

# generate reference trajectory from random control within limits
u_ref = 0.1*((u_range[0,1]-u_range[0,0])*rng.random((N,m)) - u_range[0,0])
# u_ref = 0.5*umax*np.ones((N,m))

# test iLQR
F,u,s = iLQR.iLQR(f, c, u_ref, N, s0, u_range, dt = dt, max_iters=100, tol = 1E-8)

plt.plot(s[:,0],s[:,1])
plt.show()
plt.plot(t[1:],u[:])
plt.show()