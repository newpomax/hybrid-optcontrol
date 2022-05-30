import numpy as np
from scipy.linalg import sqrtm
import UKF
import matplotlib.pyplot as plt

rng = np.random.default_rng(130)

dt = 0.01
tend = 20
N = int(tend/dt)
tspan = np.linspace(0,tend,N+1)

n = 3
m = 1
u = lambda t: np.array([1, np.sin(t)])
f = lambda x,u: np.array([np.cos(x[2])*u[0], np.sin(x[2])*u[0], u[1]])
g = lambda x,u: np.sqrt(x[0]**2 + x[1]**2)
Q = 0.001*dt*np.eye(n)
R = 0.001*np.eye(m)

mu0 = np.zeros((n,1))
cov0 = 0.01*np.eye(n)

ukf = UKF.UKF(f,g,n,m,Q,R,dt,sqrt_type='axis')
ukf.initialize(mu0,cov0)

# Simulate
Qsqrt = sqrtm(Q)
Rsqrt = sqrtm(R)
y = np.zeros((N,m))
us = np.zeros((N,2))
x = np.zeros((N+1,n))
mu = np.zeros((N+1,n))
cov = np.zeros((N+1,n,n))

x[0] = np.squeeze(mu0 + Qsqrt@rng.normal(size=(n,1)))
mu[0] = np.squeeze(mu0)
cov[0] = cov0

for i in range(tspan.size-1):
    us[i] = u(tspan[i]) # control at this time step
    x[i+1] = ukf.f(x[i],us[i]) + np.squeeze(Qsqrt@rng.normal(size=(n,1))) # integrate + noise
    unext = u(tspan[i+1])
    y[i] = g(x[i+1],unext) + np.squeeze(Rsqrt@rng.normal(size=(m,1))) # measurement + noise
    mu[i+1], cov[i+1] = ukf.step(us[i],unext,y[i]) # compute predicted mean
    
plt.figure()
plt.plot(mu[:,0],mu[:,1],label='$\mu$')
plt.plot(x[:,0],x[:,1],label='Truth')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectory')
plt.legend()

plt.figure()
plt.subplot(311)
plt.plot(tspan,mu[:,0],label='$\mu$')
cov1 = 2*np.sqrt(cov[:,0,0])
# plt.plot(tspan,mu[:,0]-cov1,mu[:,0]+cov1,label='95% err')
plt.plot(tspan,x[:,0],label='Truth')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()

plt.subplot(312)
plt.plot(tspan,mu[:,1],label='$\mu$')
cov1 = 2*np.sqrt(cov[:,1,1])
# plt.plot(tspan,mu[:,1]-cov1,mu[:,1]+cov1,label='95% err')
plt.plot(tspan,x[:,1],label='Truth')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()

plt.subplot(313)
plt.plot(tspan,mu[:,2],label='$\mu$')
cov1 = 2*np.sqrt(cov[:,2,2])
# plt.plot(tspan,mu[:,2]-cov1,mu[:,2]+cov1,label='95% err')
plt.plot(tspan,x[:,2],label='Truth')
plt.xlabel('t')
plt.ylabel('$\theta$')
plt.legend()

plt.show()



    
    
    
