import iLQR
import importlib
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

importlib.reload(iLQR)

rng = np.random.default_rng(12345)

def springdamp(s, u):
    """Compute the spring-damp state derivative."""
    m = 1.     # mass
    k = 10   # spring coeff
    b = 0 # damping coeff
    g = 9.81    # gravitational acceleration
    xeq = 0 # spring equilibrium height
    x, v = s
    a = u/m - (b/m)*v - (k/m)*(x-xeq) - g # spring damper in gravity
    ds = jnp.array([v, a[0]])
    return ds

# initialize
n = 2                               # state dimension
m = 1                                # control dimension
s0 = np.array([2, 0])  # initial 
period = 2
sgoal = lambda t: jnp.array([2*(jnp.cos(2*np.pi*t/period)**2),0])
dt = 0.05                             # discrete time resolution
u_range = np.array([[-100, 100]])
T = 10.                              # total simulation time
Q = 10*np.eye(n)                   # state cost matrix
R = np.eye(m)                   # control cost matrix
c = lambda s,u,t: (s-sgoal(t)).T@Q@(s-sgoal(t))+ u@R@u   # cost function
h = lambda s: (s-sgoal(T)).T@Q@(s-sgoal(T))
t = np.arange(0., T + dt, dt)
N = t.size - 1

# generate reference trajectory from random control within limits
u_ref = 0.01*(u_range[0,1]-u_range[0,0])*rng.random((N,m)) - u_range[0,0]
u_ref = 5*np.array([np.cos(2*np.pi*t[:-1]/period)]).T

F,u,s = iLQR.iLQR(springdamp, c, h, u_ref, N, s0, u_range, dt)

plt.plot(t,s[:,0])
plt.show()
plt.plot(t,s[:,1])
plt.show()
plt.plot(t[1:],u[:])
plt.show()