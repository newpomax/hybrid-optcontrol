import iLQR
import UKF
import importlib
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy import optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
from jaxinterp2d import CartesianGrid
from functools import partial
from scipy.optimize import root_scalar
from scipy.interpolate import interp2d
from itertools import product

class hybrid():
    def __init__(self):
        self.rng = np.random.default_rng(12345) # consistent seed rng
        
        self.n = 0.62 # power law for radius growth as function of G
        self.m = 0.015 # power law for mdot as function of inverse distance from nozzle
        self.a = 9.27E-5 # m^(2n+m+1) kg^(-n) s^(n-1), multiplier in radius/mdot equations
        self.rhof = 920 # kg/m^3, density of fuel grain
        self.Pamb = 101325 # Pa, ambient pressure
        self.Ru = 8.31446261815324 # J/K/mol, universal gas constant
        
        self.L = 1.143 # m, length of grain
        self.tburn = 15 # end of simulation
        
        self.Athroat = np.pi*(0.05**2) # 2 cm radius nozzle? guessing here
        self.exp_ratio = 3.2 # again, guessing
        self.mox_max = 1.7*2.55*5 # kg/s, guessing from value used in example with multiplicative factor

        self.k = 10 # state dimension
        x_start = self.L/1000 # starting point
        logbase = 10000 # base of logspace - higher value means tighter spacing at port start
        self.x = np.logspace(np.log(x_start)/np.log(logbase), np.log(self.L)/np.log(logbase), num = self.k, base = logbase) # create logarithmically-spaced points to track curvature near port start
        
        self.N = 100 # number of time steps
        self.Rmax = 0.10 # maximum grain radius
        
        self.loadcombdata("CEA/ParaffinLOXCEA.csv")
        
        self.t_iLQR = None
        self.t_sim = None
    
    def target_thrust(self,t):
        # Target thrust
        period = 2
        # return 38E3 + 2E3*(((t//period)%2)*2 - 1) # square wave
        return 25E3 + 1E3*jnp.sin(2*np.pi*t/period) # sinusoid
        # return 35E3 # constant
    
    def r0(self,x):
        return 0.0508*np.ones_like(x)
    
    def loadcombdata(self,combfile):
        combdata = np.loadtxt(combfile,skiprows=1,delimiter=',',usecols=range(5))
        OF = combdata[:,0] # OF ratio
        Pt = combdata[:,1] # Chamber pressure, Pa
        OFmin = OF[0]
        OFmax = OF[-1]
        Pmin = Pt[0]
        Pmax = Pt[-1]
        self.Pmin = Pmin
        self.Pmax = Pmax
        num_cols = np.argwhere(OF != OF[0])[1][0] - 1
        num_rows = np.argwhere(Pt == Pt[0])[1][0] - 1

        Tt = combdata[:,2] # Chamber temperature, K
        gamma = combdata[:,3]
        Me = np.array([self.invf(1/self.exp_ratio, g) for g in gamma]) # calculate mach number as function of OF,P
        mw = combdata[:,4]/1000 # molecular weight, converted from g/mol to kg/mol
        mdot = gamma/(((gamma+1)*0.5)**((gamma+1)*0.5/(gamma-1))) * Pt * self.Athroat / np.sqrt(gamma * self.Ru/mw * Tt) # mdot grid in OF, P
        self.fmdot = interp2d(OF.reshape((-1,num_cols)),Pt.reshape((-1,num_cols)),mdot.reshape((-1,num_cols))) # mdot as function of OF, P
        
        mdotmin = np.amax(mdot.reshape((-1,num_cols))[:,0])
        mdotmax = np.amin(mdot.reshape((-1,num_cols))[:,-1])
        self.mdotmin = mdotmin
        self.mdotmax = mdotmax
        mdot_range = np.linspace(mdotmin,mdotmax,10) # create regular grid of mdot
        P = np.array([self.getP(of, m) for (of,m) in product(OF.reshape((-1,num_cols))[:,0],mdot_range)]) # calculate pressure as function of mdot, OF
        Pe = Pt * ( 1 + 0.5*(gamma-1)*(Me**2) )**(-gamma/(gamma-1))
        Ue = Me*np.sqrt(gamma * self.Ru/mw * Tt) # exhaust velocity

        # Create interpolative functions, using Adam Coogan's jaxinterp2d
        self.fPcc = CartesianGrid(((OFmin,OFmax),(mdotmin,mdotmax)), P.reshape((num_rows,-1))) # use this to locate current chamber pressure from OF, mdot
        self.fPe = CartesianGrid(((OFmin,OFmax),(Pmin,Pmax)), Pe.reshape((-1,num_cols))) # use this to locate current chamber pressure from OF, mdot
        self.fUe = CartesianGrid(((OFmin,OFmax),(Pmin,Pmax)), Ue.reshape((-1,num_cols))) # from known OF, P - calculate exit vel
    
    @staticmethod
    def f(M,g): # mach area function
        return (((g+1)*0.5)**((g+1)*0.5/(g-1))) * M/((1 + 0.5*(g-1)*(M**2))**((g+1)*0.5/(g-1)))
    
    @staticmethod
    def invf(Aratio,g):
        return root_scalar(lambda M: (hybrid.f(M,g)-Aratio),method='brentq', bracket=[1, 10]).root
    
    def getmdot(self,OF,P):
        return self.fmdot(OF,P)[0]

    def getP(self,OF,mdot_goal):
        return root_scalar(lambda P: self.getmdot(OF,P)-mdot_goal,x0=self.Pmin,x1=self.Pmax,rtol=1E-6).root
    
    @partial(jax.jit,static_argnums=(0,))
    def thrust(self,r:np.ndarray,mox:np.ndarray,a,n,m):   
        mfuel = self.mf(r,mox,a,n,m)[-1] # calculate total fuel flow rate
        # Given a known OF and total mass flow rate, find the resulting thrust
        OF = mox/mfuel 
        mdot = mox+mfuel
        Pcc = self.fPcc(OF, mdot) # find chamber pressure that satisfies mass flow
        Thrust = mdot*self.fUe(OF,Pcc) + self.exp_ratio*self.Athroat*(self.fPe(OF,Pcc)-self.Pamb)
        return Thrust
    
    @partial(jax.jit,static_argnums=(0,))
    def thrustandpress(self,r:np.ndarray,mox:np.ndarray,a,n,m):   
        mfuel = self.mf(r,mox,a,n,m)[-1] # calculate total fuel flow rate
        # Given a known OF and total mass flow rate, find the resulting thrust
        OF = mox/mfuel 
        mdot = mox+mfuel
        Pcc = self.fPcc(OF, mdot) # find chamber pressure that satisfies mass flow
        Thrust = mdot*self.fUe(OF,Pcc) + self.exp_ratio*self.Athroat*(self.fPe(OF,Pcc)-self.Pamb)
        return Thrust, Pcc

    @partial(jax.jit,static_argnums=(0,))
    def mf(self,r:np.ndarray,mox:np.ndarray,a,n,m):
        mf = jnp.zeros(r.shape)
        for i in range(mf.size): # make this jit-friendly with lax!
            if i == 0:
                mflast = 0 # fuel flow at start of port is zero
                dx = self.x[0]
            else:
                mflast = mf[i-1] # use previous point to do integration
                dx = self.x[i]-self.x[i-1]
            mf = mf.at[i].set(jnp.squeeze(mflast + dx*2*np.pi*r[i]*self.rhof*a/(self.x[i]**m)*((mox+mflast)/(np.pi*(r[i]**2)))**n)) # Euler integration
        return mf

    @partial(jax.jit,static_argnums=(0,))
    def rdot(self,r:np.ndarray,mox:np.ndarray,a,n,m):
        mfuel = self.mf(r,mox,a,n,m)
        drdt = a/(self.x**m)*((mfuel+mox[0])/(np.pi*(r**2)))**n
        return drdt
    
    def runiLQR(self,trange = np.nan,a = np.nan,n = np.nan,m = np.nan, r0 = np.nan, tol=1E-1):
        # importlib.reload(iLQR) # force reload of iLQR in case of changes
        if np.any(np.isnan(trange)):
            trange = np.linspace(0,self.tburn,self.N+1)
        if np.any(np.isnan(r0)):
            r0 = self.r0(self.x)
        params = [a, n, m]
        selfparams = [self.a, self.n, self.m]
        a,n,m = np.where(np.isnan(params), selfparams, params) # replace nan from input with self values
        f = lambda s,u: self.rdot(s,u,a,n,m)
        s0 = jnp.array(r0)
        u_range = jnp.array([[0.75*self.mdotmin, 0.75*self.mdotmax]]) # control range
        N = trange.size - 1
        Q = 1*np.eye(1)                 # state cost matrix
        R = 1e3*np.eye(1)                # control cost matrix
        thrust = lambda s,u : self.thrust(s,u,a,n,m)
        c = lambda s,u,t: (thrust(s,u)-self.target_thrust(t)).T@Q@(thrust(s,u)-self.target_thrust(t)) + jnp.sum(jnp.exp((s-self.Rmax)*2500)) 
        h = lambda s: 1 #terminal penalty
        self.t_iLQR = trange
        
        self.Thrustgoal = np.array([self.target_thrust(self.t_iLQR[i]) for i in range(self.N)])
        print('Min thrust = %0.1f N at mox = %0.2f kg/s'%(thrust(s0,u_range[0,0]),u_range[0,0]))
        print('Max thrust = %0.1f N at mox = %0.2f kg/s'%(thrust(s0,u_range[0,1]),u_range[0,1]))
        print('Min target thrust = %0.1f N, max target thrust = %0.1f N'%(np.amin(self.Thrustgoal),np.amax(self.Thrustgoal)))
        
        u_ref = 0.1*(u_range[0,1]-u_range[0,0])*self.rng.random((self.N,1)) + u_range[0,0] # generate reference control from random control within limits

        print('Starting iLQR')
        self.R_bar,self.MOX_bar,self.L,self.l = iLQR.iLQR(f, c, h, u_ref, N, s0, u_range, trange, tol = tol)
        self.a_bar, self.n_bar, self.m_bar = (a, n, m)
        
    def simulate(self, trange = np.nan, a0 = np.nan, a_σ = 0, m0 = np.nan, m_σ = 0, n0 = np.nan, n_σ = 0, r0 = np.nan, r_σ = 0):
        if self.t_iLQR is None:
            print('Run iLQR before trying to simulate.')
            return
        if np.any(np.isnan(trange)):
            trange = np.linspace(0,self.tburn,self.N+1)
        if np.any(np.isnan(r0)):
            r0 = self.r0(self.x)
        params = [a0, n0, m0]
        selfparams = [self.a, self.n, self.m]
        a,n,m = np.where(np.isnan(params), selfparams, params) # replace nans with self values
        
        def rk4(f,s,u,dt):
            k1 = dt * f(s, u)
            k2 = dt * f(s + k1 / 2, u)
            k3 = dt * f(s + k2 / 2, u)
            k4 = dt * f(s + k3, u)
            return (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        # Initialize storage arrays
        N = trange.size - 1
        a_true = self.a + self.rng.normal(scale=a_σ)*np.ones((N+1,1)) # record truth parameter values for simulation
        n_true = self.n + self.rng.normal(scale=n_σ)*np.ones((N+1,1))
        m_true = self.m + self.rng.normal(scale=m_σ)*np.ones((N+1,1))
        a_sim = a*np.ones((N+1,1)) # record truth parameter values for simulation
        n_sim = n*np.ones((N+1,1)) 
        m_sim = m*np.ones((N+1,1))
        MOX = np.zeros((N,1))
        R = np.zeros((N+1,self.k)) # our estimate of port radius in time
        R_true = np.zeros((N+1,self.k)) # true port radius in time
        R_true[0] = self.r0(self.x) + self.rng.normal(scale = r_σ, size=r0.size) # add noise to true initial port radius
        R[0] = r0 + self.rng.normal(scale=r_σ, size = r0.size)
        dt = self.tburn/N        
        
        for i in range(N):
            # Noisy measurement of state
            # compute new control based on best estimate of current state
            ds = R[i] - self.R_bar[i] # compute delta between estimate and reference
            MOX[i] = self.MOX_bar[i] + self.L[i]@ds + self.l[i] # control output based on known info
            # integrate true dynamics (no noise on port growth, just parameters)
            R_true[i+1] = R_true[i] + rk4(lambda s,u: self.rdot(s,u,a_true[i],n_true[i],m_true[i]),R_true[i],MOX[i],dt)
            R[i+1] = R_true[i+1] + self.rng.normal(scale=r_σ, size = r0.size)
            # add noise to truth parameter values, clipping to reasonable range
            a_true[i+1] = self.a + self.rng.normal(scale = a_σ)
            n_true[i+1] = self.n + self.rng.normal(scale = n_σ)
            m_true[i+1] = self.m + self.rng.normal(scale = m_σ)
            # compute noisy measurements of current state: thrust and chamber pressure
            a_sim[i+1] = a
            n_sim[i+1] = n
            m_sim[i+1] = m
            
        self.a_sim, self.n_sim, self.m_sim = (a_sim,n_sim,m_sim)
        self.R_sim, self.R_true, self.MOX_sim, self.a_true, self.n_true, self.m_true = (R,R_true,MOX,a_true,n_true,m_true)
        self.t_sim = trange
        self.plot_output(True)
        
    def get_measurement(self,R,MOX,a,n,m):   
        thrust, press = self.thrustandpress(R,MOX,a,n,m)
        return np.squeeze(np.array([thrust,press]))
        
    def plot_output(self,plot_sim=False):
        if self.t_iLQR is None:
            print('Run iLQR before trying to plot.')
            return
        
        if plot_sim and self.t_sim is not None:
            # Compute output for simulated values
            N_sim = self.t_sim.size - 1
            MOX = np.squeeze(self.MOX_sim)
            R = self.R_sim
            R_true = self.R_true
            MF = np.array([self.mf(R[i,:],MOX[i],self.a_sim[i],self.n_sim[i],self.m_sim[i])[-1] for i in range(N_sim)])
            MF_true = np.array([self.mf(R_true[i,:],MOX[i],self.a_true[i],self.n_true[i],self.m_true[i])[-1] for i in range(N_sim)])
            OF = MF/MOX
            OF_true = MF_true/MOX
            MTOT = MF + MOX
            MTOT_true = MF_true + MOX
            Thrust = np.array([self.thrust(R[i,:],MOX[i],self.a_sim[i],self.n_sim[i],self.m_sim[i]) for i in range(N_sim)])
            Thrust_true = np.array([self.thrust(R_true[i,:],MOX[i],self.a_true[i],self.n_true[i],self.m_true[i]) for i in range(N_sim)])
            
        # Compute output for simulated values
        N_bar = len(self.t_iLQR) - 1
        MOX_bar = np.squeeze(self.MOX_bar)
        R_bar = self.R_bar
        MF_bar = np.array([self.mf(R_bar[i,:],MOX_bar[i],self.a_bar,self.n_bar,self.m_bar)[-1] for i in range(N_bar)])
        OF_bar = MF_bar/MOX_bar
        MTOT_bar = MF_bar + MOX_bar
        Thrust_bar = np.array([self.thrust(R_bar[i,:],MOX_bar[i],self.a_bar,self.n_bar,self.m_bar) for i in range(N_bar)])
        
        plt.figure(figsize = (16,24))
        plt.subplot(311)
        num_curves = min(5,N_bar)
        c = np.arange(1, num_curves + 1)
        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
        cmap.set_array([])
        ts = ['']*num_curves
        plt.axhline(self.Rmax,color='k',linestyle='--',label='$R_{max}$')
        for i in range(num_curves):
            thisi = i*(N_bar//(num_curves-1))
            ts[i] = str(self.t_iLQR[thisi])
            if i == 0:
                plt.plot(self.x,R_bar[thisi],c=cmap.to_rgba(i + 1),label = 'Ref.')
            else:
                plt.plot(self.x,R_bar[thisi],c=cmap.to_rgba(i + 1),label = '_nolegend_')
        if plot_sim:
            num_curves = min(5,N_sim)
            for i in range(num_curves):
                thisi = i*(N_sim//(num_curves-1))
                ts[i] = str(self.t_sim[thisi])
                if i == 0:
                    plt.plot(self.x,R[thisi],':^',c=cmap.to_rgba(i + 1),label = 'Sim. est.')
                    plt.plot(self.x,R_true[thisi],'--',c=cmap.to_rgba(i + 1),label = 'Sim. truth')
                else:
                    plt.plot(self.x,R[thisi],':^',c=cmap.to_rgba(i + 1),label = '_nolegend_')
                    plt.plot(self.x,R_true[thisi],'--',c=cmap.to_rgba(i + 1),label = '_nolegend_')
        plt.legend()
        cbar = plt.colorbar(cmap,ticks=c,label='$t$, s')
        cbar.set_ticklabels(ts)
        plt.xlabel('$x$, m')
        plt.ylabel('$r$, m')
        plt.title('Port radius in time')

        plt.subplot(323)
        plt.plot(self.t_iLQR[:-1],self.Thrustgoal,'k:^',label='Target')
        plt.plot(self.t_iLQR[:-1],Thrust_bar,'k',label='Ref.')
        if plot_sim:
            # plt.plot(self.t_sim[:-1],Thrust,':^',label='Thrust, sim est.')
            plt.plot(self.t_sim[:-1],Thrust_true,'k--',label='Sim.')
        plt.xlabel('$t$, s')
        plt.ylabel('$F_{Thrust}$, N')
        plt.title('Thrust')
        plt.legend()

        plt.subplot(324)
        plt.plot(self.t_iLQR[:-1],OF_bar,'k',label='Ref.')
        if plot_sim:
            # plt.plot(self.t_sim[:-1],OF,':^',label='sim est.')
            plt.plot(self.t_sim[:-1],OF_true,'k--',label='Sim.')
        plt.xlabel('$t$, s')
        plt.ylabel('OF')
        plt.title('OF')
        plt.legend()
        
        plt.subplot(325)
        plt.plot(self.t_iLQR[:-1],MF_bar,'k',label='Ref.')
        if plot_sim:
            # plt.plot(self.t_sim[:-1],MF,':^',label='Sim est.')
            plt.plot(self.t_sim[:-1],MF_true,'k--',label='Sim.')
        plt.xlabel('$t$, s')
        plt.ylabel('$\dot{m}_{fuel}$, kg/s')
        plt.title('Fuel mass flow at port end')
        plt.legend()

        plt.subplot(326)
        plt.plot(self.t_iLQR[:-1],MOX_bar,'k',label='Ref.')
        if plot_sim:
            plt.plot(self.t_sim[:-1],MOX,'k--',label='Sim.')
        plt.xlabel('$t$, s')
        plt.ylabel('$\dot{m}_{ox}$, kg/s')
        plt.title('Oxidizer mass flow at port end')
        plt.legend()
        
        plt.savefig('images/HybridOutput.png',bbox_inches='tight')
        
        if plot_sim:
            plt.figure(figsize=(18, 6))
            plt.subplot(131)
            plt.plot(self.t_sim, self.a_sim, 'k', label='$a$, sim est.')
            plt.plot(self.t_sim, self.a_true, 'k--', label='$a$, sim truth')
            plt.xlabel('$t$, s')
            plt.ylabel('$a$')
            plt.title('$a$ parameter estimation')
            plt.legend()
            plt.subplot(132)
            plt.plot(self.t_sim, self.n_sim, 'k', label='$n$, sim est.')
            plt.plot(self.t_sim, self.n_true, 'k--', label='$n$, sim truth')
            plt.xlabel('$t$, s')
            plt.ylabel('$n$')
            plt.title('$n$ parameter estimation')
            plt.legend()
            plt.subplot(133)
            plt.plot(self.t_sim, self.m_sim, 'k', label='$m$, sim est.')
            plt.plot(self.t_sim, self.m_true, 'k--', label='$m$, sim truth')
            plt.xlabel('$t$, s')
            plt.ylabel('$m$')
            plt.title('$m$ parameter estimation')
            plt.legend()

        plt.show()        

