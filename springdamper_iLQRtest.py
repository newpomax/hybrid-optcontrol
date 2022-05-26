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
    k = 5   # spring coeff
    B = 0.1  # non-linear spring coeff
    b = 2 # damping coeff
    g = 9.81    # gravitational acceleration 
    xeq = 0.2 # spring equilibrium height
    cube_side = 1 # m, size of cube
    CdAρ = 0.8*(cube_side**2)*1.275 # coefficient for drag equation, ~
    x, v = s
    a = u/m - (b/m)*v - (k/m)*(x-xeq) - (B/m)*((x-xeq)**3) - g - 0.5*CdAρ*(v**2) # spring damper in gravity with drag
    ds = jnp.array([v, a[0]])
    return ds

# initialize
n = 2                               # state dimension
m = 1                                # control dimension
period = 2
sgoal = lambda t: jnp.array([jnp.sin(2*np.pi*t/period) + 0.25*jnp.sin(8*np.pi*t/period) ,2*np.pi/period*jnp.cos(2*np.pi*t/period) + 2*np.pi/period*jnp.cos(8*np.pi*t/period) ])
s0 = sgoal(0)  # initial state much match goal function
dt = 0.1                       # discrete time resolution
u_range = np.array([[-20, 20]])
T = 10                          # total simulation time
Q = 10*np.eye(n)                   # state cost matrix
R = 100*np.eye(m)                   # control cost matrix
# c = lambda s,u,t: (s-sgoal(t)).T@Q@(s-sgoal(t))+ u@R@u   # cost function
c = lambda s,u,t: (s-sgoal(t)).T@Q@(s-sgoal(t)) # cost function (no control penalty)
h = lambda s: (s-sgoal(T)).T@Q@(s-sgoal(T))
t = np.arange(0., T + dt, dt)
N = t.size - 1

# generate reference control from random control within limits
u_ref = 0.1*((u_range[0,1]-u_range[0,0])*rng.random((N,m)) - u_range[0,0])

s,u,L,l = iLQR.iLQR(springdamp, c, h, u_ref, N, s0, u_range, dt = dt, tol = 1E-1)

plt.plot(t,s[:,0])
plt.plot(t,np.array([sgoal(x)[0] for x in t]))
plt.ylim([-10,10])
plt.show()
plt.plot(t,s[:,1])
plt.show()
plt.plot(t[1:],u[:])
plt.show()