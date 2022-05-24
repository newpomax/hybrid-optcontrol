import numpy as np
import cvxpy as cvx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from functools import partial


def regularize(A: jnp.ndarray, λ:float):
    w,v = jnp.linalg.eig(A)
    w = jnp.real(w)
    w = jnp.maximum(jnp.zeros_like(w), w) + λ
    winv = 1/w
    w = jnp.diag(w)
    winv = jnp.diag(winv)
    v = jnp.real(v)
    return winv, v # return inverse eig matrix, eigenvec matrix

@partial(jax.jit, static_argnums=(0,))
# @partial(jax.vmap, in_axes=(None, 0, 0))
def linearize_dynamics(fd: callable,
                        s: jnp.ndarray,
                        u: jnp.ndarray):
    """ Linearize the discretized dynamics function `fd(s,u)` around `(s,u)`.
        Returns the linear dynamics matrices A and B for the augmented delta dynamics,
        f(s,u) = A@[dx,1].T+B@du 
    """
    fd_s, fd_u = jax.jacfwd(fd, (0,1))(s,u)
    A = jnp.array(fd_s) + jnp.eye(fd_s.shape[0])
    B = jnp.array(fd_u)
    return A, B

@partial(jax.jit, static_argnums=(0,))
# @partial(jax.vmap, in_axes=(None, 0, 0))
def quadratize_cost(c: callable,
                    s: jnp.ndarray,
                    u: jnp.ndarray,
                    t: float):
    """ Quadratize the cost function `c(s,u)` around `(s,u)`.
        Returns the regularized quadratic matrices Q, R, and S for the quadratized augmented dynamics,
        c(s,u) = 0.5*sum([quad_form([x,1].T,Q) + quad_form(u,R) + 2*[x.T, 1]@S@u for i in range(N)])
    """
    dc = jax.jacfwd(c, (0,1))
    cs, cu = dc(s,u,t)
    (css, csu), (cus, cuu) = jax.jacfwd(dc,(0,1))(s,u,t)
    Q = jnp.array(css)
    θ = jnp.array(cs,ndmin=2).T
    q = jnp.array(c(s,u,t),ndmin=2)
    R = jnp.array(cuu)
    r = jnp.array(cu,ndmin=2).T
    P = jnp.array(cus)
    
    return Q,θ,q,R,r,P

def rk4(f,s,u,dt):
    k1 = dt * f(s, u)
    k2 = dt * f(s + k1 / 2, u)
    k3 = dt * f(s + k2 / 2, u)
    k4 = dt * f(s + k3, u)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6

def discretize(f,dt):
    """ Make discretized dynamics function from discrete function, `f(s,u)` with time step `dt` 
        using Runge-Kutta 4th order method.
    """
    def integrator(s, u, f=f, dt=dt):
        return rk4(f,s,u,dt)
    
    return integrator

def LQR_iteration(fd: callable, c: callable, h: callable, N: int, u_bar:np.ndarray, s_bar:np.ndarray, s0: np.ndarray, u_range:np.ndarray, λ:float, dt:float, εmax:float = 1 ):
    """ Solve a single LQR sub-problem with discretized dynamics `fd(s,u)`, cost function
        `c(s,u,t)`, reference linear control law `F` for augmented delta dynamics, reference control `u_bar`,
        reference trajectory `s_bar`, initial state `s0`, and Levenberg-Marquardt constant λ.
        
        Returns the linear feedback law at each time step for the augmented dynamics vector,
        [s, 1].T
    """
    S = np.zeros((N+1, s0.size, s0.size))
    ς = np.zeros((N+1, s0.size, 1))
    s = np.zeros((N+1, 1, 1))
    S[-1], ς[-1], s[-1] = quadratize_cost(lambda s,u,t: h(s), s_bar[N], jnp.zeros_like(u_bar[0]), dt*(N))[:3] # initialize P matrix for Ricatti recursion, augmented for affine dynamics/cost
    L = np.zeros((N, u_bar.shape[1], s0.size)) # initialize output linear control matrix 
    l = np.zeros((N, u_bar.shape[1], 1)) # initialize output linear control affine term
    for i in range(N-1,-1,-1):
        A, B = linearize_dynamics(fd, s_bar[i], u_bar[i])
        Q, θ, q, R, r, P = quadratize_cost(c, s_bar[i], u_bar[i], i*dt)

        H = R + B.T@S[i+1]@B
        G = P + B.T@S[i+1]@A
        g = r + B.T@ς[i+1]
        Dinv, V = regularize(H, λ) # Levenberg-Marquardt-esque regularization
        ℋinv = V@Dinv@V.T # compute quick inverse using eigen decomp values from regularization
        dustar = -ℋinv@g
        ε = min(1,εmax)
        # ε = jnp.amin(jnp.clip((u_range[:,1]- u_bar[i])/dustar,0,1))
        ε = min(ε, jnp.amin(jnp.clip((u_range[:,0]- u_bar[i])/dustar,0,1)))
        l[i] = ε*dustar
        L[i] = -ℋinv@G 
        for j in range(u_bar.shape[1]): # for each dimension of control
            if (l[i,j] + u_bar[i,j]) <= u_range[j,0] or (l[i,j] + u_bar[i,j]) >= u_range[j,1]:
                L[i,j,:] = 0 # control constraints impact choice of optimal L
        S[i] = Q + A.T@S[i+1]@A + L[i].T@H@L[i] + L[i].T@G + G.T@L[i] # quadratic cost matrix for augmented delta-dynamics
        ς[i] = θ + A.T@ς[i+1] + L[i].T@g + G.T@l[i]
        s[i] = q + s[i+1] + 0.5*l[i].T@H@l[i] + l[i].T@g

    new_s_bar, new_u_bar = simulate(fd, L, l, u_bar, s_bar, s0, N, u_range) # forward pass - simulate reference control foward using augmented dynamics
    return L, new_u_bar, new_s_bar, np.squeeze(s[0])
    
def simulate(fd:callable, L: np.ndarray, l:np.ndarray, u_bar: np.ndarray, s_bar: np.ndarray, s0: np.ndarray, N:int, u_range:np.ndarray):
    """ Simulate the results of a linear control matrix `F` for `N` time steps via the discretized 
        delta dynamics function 'fd(s,u)' given an initial state, `s0`. 
    """
    s = np.zeros((N+1,s0.size))
    u = np.zeros((N,u_bar.shape[1]))
    s[0] = s0
    for i in range(N):
        ds = jnp.array(s[i]-s_bar[i],ndmin=2).T # control law is based on distance from reference, s_bar
        u[i] = jnp.squeeze( L[i]@ds + l[i] + jnp.array(u_bar[i],ndmin=2).T ) # output of control is difference from reference, u_bar
        u[i] = jnp.clip(u[i],u_range[:,0],u_range[:,1])  # impose control contraint
        s[i+1] = s[i] + fd(s[i],u[i]) # use discretized function to integrate
    return s, u

def iLQR(f:callable, c:callable, h:callable, u_ref: np.ndarray, N: int, s0: np.ndarray, u_range:np.ndarray,
         dt:float = 1E-2, max_iters:int = 400, tol: float = 1E-4, λmax:float= 1E4 ):
    """ Solve the iLQR problem for a system with discrete dynamics `f(s,u)`, cost function
        `c(s,u)`, terminal cost `h(s)`, initial reference control `uref`, number of time steps `N+1`, initial state `s0`,
        time step `dt`, max number of iterations 'max_iters', and convergence tolerance `tol`.
    """
    fd = jax.jit(discretize(f,dt))
    # Create initial reference trajectory from initial reference control, and compute its cost
    s_bar = np.zeros((N+1,s0.size))
    u_bar = u_ref
    cost = np.inf
    s_bar[0] = s0
    
    # do initial simulation
    for i in range(N):
        s_bar[i+1] = s_bar[i] + fd(s_bar[i],u_bar[i]) 
                
    λbase = 1E-8
    λfactor = 10
    λ = λbase
    u_dif = -1
    new_cost = -1
    print('Reference cost = %0.2e' % cost)
    for k in ( pbar := tqdm(range(max_iters)) ):
        pbar.set_description("λ = %.2e, u_dif = %.2e, last cost = %.5e, best cost-so-far = %0.5e" % (λ,u_dif,new_cost,cost))
        F, new_u_bar, new_s_bar, new_cost = LQR_iteration(fd, c, h, N, u_bar, s_bar, s0, u_range, λ, dt, 1)
        u_dif = np.amax(np.linalg.norm((u_bar - new_u_bar),np.inf,axis=1))
        if new_cost < cost and u_dif < tol: # if cost has converged 
            print("iLQR converged after {0} iterations with cost {1:.2e}.".format(k+1,cost))
            return F, new_u_bar, new_s_bar  # return control law, reference control, and reference trajectory
        elif k == max_iters - 1:
            plt.plot(dt*np.arange(1,N+1),new_u_bar)
            plt.show()
            print("WARNING: Sub-optimal output. iLQR failed to converge. Terminal λ value is {0} with best-cost-so-far {1:.2e}".format(λ,new_cost))
            return F, new_u_bar, new_s_bar
        
        if new_cost < cost:
            # TODO: make this adaptive!
            λ = min(λbase,λ/λfactor) # decrease "line-search" parameter
            u_bar, s_bar, cost = new_u_bar, new_s_bar, new_cost # update reference trajectory and best-cost-so-far         
        else:
            # TODO: make this adaptive!
            if λfactor*λ > λmax:
                print("WARNING: Sub-optimal output. iLQR failed to converge. Terminal λ value is {0} with best-cost-so-far {1:.2e}".format(λ,new_cost))
                return F, new_u_bar, new_s_bar
            λ = λfactor*λ # increase "line-search" parameter and try again with same u_bar, s_bar
        
            