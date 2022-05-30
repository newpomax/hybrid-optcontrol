import iLQR
import importlib
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy import optimize
import matplotlib.pyplot as plt
from jaxinterp2d import CartesianGrid
from functools import partial
from scipy.optimize import root_scalar
from scipy.interpolate import interp2d
from itertools import product

importlib.reload(iLQR)

rng = np.random.default_rng(12345)

## Constants - paraffin and lox hybrid ##
#---------------------------------------#
n = 0.62 # power law for radius growth as function of G
m = 0.015 # power law for mdot as function of inverse distance from nozzle
a = 9.27E-5 # m^(2n+m+1) kg^(-n) s^(n-1), multiplier in radius/mdot equations
rhof = 920 # kg/m^3, density of fuel grain
Pamb = 101325 # Pa, ambient pressure
Ru = 8.31446261815324 # J/K/mol, universal gas constant

r0 = 0.0508 #m ,initial radius
L = 1.143 # m, length of grain
tburn = 30 # end of simulation

Athroat = np.pi*(0.05**2) # 2 cm radius nozzle? guessing here
exp_ratio = 3.2 # again, guessing
mox_max = 1.7*2.55*5 # kg/s, guessing from value used in example with multiplicative factor

period = 2
thrust_goal = lambda t: 35E3 + 1E3*(((t//period)%2)*2 - 1) # N, target thrust in time

k = 10 # state dimension
x_start = L/1000 # starting point
logbase = 10000 # base of logspace - higher value means tighter spacing at port start
x = np.logspace(np.log(x_start)/np.log(logbase), np.log(L)/np.log(logbase), num = k, base = logbase) # create logarithmically-spaced points to track curvature near port start

def f(M,g): # mach area function
    return (((g+1)*0.5)**((g+1)*0.5/(g-1))) * M/((1 + 0.5*(g-1)*(M**2))**((g+1)*0.5/(g-1)))

def invf(Aratio,g):
    return root_scalar(lambda M: (f(M,g)-Aratio),method='brentq', bracket=[1, 10]).root

combdata = np.loadtxt("CEA/ParaffinLOXCEA.csv",skiprows=1,delimiter=',',usecols=range(5))
OF = combdata[:,0] # OF ratio
Pt = combdata[:,1] # Chamber pressure, Pa
OFmin = OF[0]
OFmax = OF[-1]
Pmin = Pt[0]
Pmax = Pt[-1]
num_cols = np.argwhere(OF != OF[0])[1][0] - 1
num_rows = np.argwhere(Pt == Pt[0])[1][0] - 1

Tt = combdata[:,2] # Chamber temperature, K
gamma = combdata[:,3]
Me = np.array([invf(1/exp_ratio, g) for g in gamma]) # calculate mach number as function of OF,P
mw = combdata[:,4]/1000 # molecular weight, converted from g/mol to kg/mol
mdot = gamma/(((gamma+1)*0.5)**((gamma+1)*0.5/(gamma-1))) * Pt * Athroat / np.sqrt(gamma * Ru/mw * Tt) # mdot grid in OF, P
fmdot = interp2d(OF.reshape((-1,num_cols)),Pt.reshape((-1,num_cols)),mdot.reshape((-1,num_cols))) # mdot as function of OF, P

def getmdot(OF,P):
    return fmdot(OF,P)[0]

def getP(OF,mdot_goal):
    return root_scalar(lambda P: getmdot(OF,P)-mdot_goal,x0=Pmin,x1=Pmax,rtol=1E-6).root

mdotmin = np.amax(mdot.reshape((-1,num_cols))[:,0])
mdotmax = np.amin(mdot.reshape((-1,num_cols))[:,-1])
mdot_range = np.linspace(mdotmin,mdotmax,10) # create regular grid of mdot
P = np.array([getP(of, m) for (of,m) in product(OF.reshape((-1,num_cols))[:,0],mdot_range)]) # calculate pressure as function of mdot, OF
Pe = Pt * ( 1 + 0.5*(gamma-1)*(Me**2) )**(-gamma/(gamma-1))
Ue = Me*np.sqrt(gamma * Ru/mw * Tt) # exhaust velocity

# Create interpolative functions, using Adam Coogan's jaxinterp2d
fPcc = CartesianGrid(((OFmin,OFmax),(mdotmin,mdotmax)), P.reshape((num_rows,-1))) # use this to locate current chamber pressure from OF, mdot
fPe = CartesianGrid(((OFmin,OFmax),(Pmin,Pmax)), Pe.reshape((-1,num_cols))) # use this to locate current chamber pressure from OF, mdot
fUe = CartesianGrid(((OFmin,OFmax),(Pmin,Pmax)), Ue.reshape((-1,num_cols))) # from known OF, P - calculate exit vel

##         Hybrid functions            ##
#---------------------------------------#
@partial(jax.jit,static_argnums=(2,3,4,5,6,7))
def thrust(r:np.ndarray,mox:np.ndarray,fPcc:callable=fPcc,fPe:callable=fPe,fUe:callable=fUe,exp_ratio=exp_ratio,Pamb=Pamb,Athroat=Athroat):   
    mfuel = mf(r,mox)[-1] # calculate total fuel flow rate
    # Given a known OF and total mass flow rate, find the resulting thrust
    OF = mox/mfuel 
    mdot = mox+mfuel
    Pcc = fPcc(OF, mdot) # find chamber pressure that satisfies mass flow
    Thrust = mdot*fUe(OF,Pcc) + exp_ratio*Athroat*(fPe(OF,Pcc)-Pamb)
    return Thrust

@partial(jax.jit,static_argnums=(2,3,4,5,6))
def mf(r:np.ndarray,mox:np.ndarray,x=x,a:float=a,n:float=n,m:float=m,rhof:float=rhof):
    mf = jnp.zeros(r.shape)
    for i in range(mf.size): # make this jit-friendly!
        if i == 0:
            mflast = 0 # fuel flow at start of port is zero
            dx = x[0]
        else:
            mflast = mf[i-1] # use previous point to do integration
            dx = x[i]-x[i-1]
        mf = mf.at[i].set(jnp.squeeze(mflast + dx*2*np.pi*r[i]*rhof*a/(x[i]**m)*((mox+mflast)/(np.pi*(r[i]**2)))**n)) # Euler integration
    return mf

@partial(jax.jit,static_argnums=(2,3,4))
def rdot_full(r:np.ndarray,mox:np.ndarray,a:float=a,n:float=n,m:float=m):
    mfuel = mf(r,mox)
    drdt = a/(x**m)*((mfuel+mox[0])/(np.pi*(r**2)))**n
    return drdt


##     iLQR variable setup and run     ##
#---------------------------------------#
f = lambda s,u: rdot_full(s,u)
s0 = r0*jnp.ones((k,))
sgoal = thrust_goal
dt = 0.1                      # discrete time resolution
u_range = jnp.array([[0.75*mdotmin, 0.75*mdotmax]]) # control range
T = tburn
Q = 1*np.eye(1)                 # state cost matrix
R = 1e3*np.eye(1)                # control cost matrix
c = lambda s,u,t: (thrust(s,u)-thrust_goal(t)).T@Q@(thrust(s,u)-thrust_goal(t)) + u@R@u   # cost function
# c = lambda s,u,t: (thrust(s,u)-thrust_goal(t)).T@Q@(thrust(s,u)-thrust_goal(t)) # cost function (no control penalty)
h = lambda s: 1
t = np.arange(0., T + dt, dt)
N = t.size - 1

u_ref = 0.1*(u_range[0,1]-u_range[0,0])*rng.random((N,1)) + u_range[0,0] # generate reference control from random control within limits

Thrustgoal = np.array([thrust_goal(t[i]) for i in range(N)])
print('Min thrust = %0.1f N at mox = %0.2f kg/s'%(thrust(s0,u_range[0,0]),u_range[0,0]))
print('Max thrust = %0.1f N at mox = %0.2f kg/s'%(thrust(s0,u_range[0,1]),u_range[0,1]))
print('Min target thrust = %0.1f N, max target thrust = %0.1f N'%(np.amin(Thrustgoal),np.amax(Thrustgoal)))

print('Starting iLQR')
R,MOX,L,l = iLQR.iLQR(f, c, h, u_ref, N, s0, u_range, dt = dt, tol = 1E-1)

# Compute output
MOX = np.squeeze(MOX)
MF = np.array([mf(R[i,:],MOX[i])[-1] for i in range(N)])
print(MF.shape)
OF = MF/MOX
MTOT = MF + MOX
Thrust = np.array([thrust(R[i,:],MOX[i]) for i in range(N)])


# plot
plt.figure()
num_curves = min(10,N)
for i in range(num_curves):
    thisi = i*(N//num_curves)
    plt.plot(x,R[thisi,:],label='t = %.2f s'%t[thisi])
    
plt.xlabel('$x$, m')
plt.ylabel('$r$, m')
plt.title('Port radius in time')
plt.legend()

plt.figure()
plt.plot(t[:-1],Thrust,label='Thrust')
plt.plot(t[:-1],Thrustgoal,'--',label='Goal')
plt.xlabel('$t$, s')
plt.ylabel('$F_{Thrust}$, N')
plt.title('Thrust')
plt.legend()

plt.figure()
plt.plot(t[:-1],OF)
plt.xlabel('$t$, s')
plt.ylabel('OF')
plt.title('OF')

plt.figure()
plt.plot(t[:-1],MOX,label='$\dot{m}_{ox}$')
plt.plot(t[:-1],MF,label='$\dot{m}_{fuel}$')
plt.plot(t[:-1],MTOT,label='$\dot{m}_{total}$')
plt.xlabel('$t$, s')
plt.ylabel('$\dot{m}$, kg/s')
plt.title('Mass flow at port end')
plt.legend()

plt.show()


