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
    w = jnp.diag(jnp.maximum(jnp.zeros_like(w), w) + λ)
    return w, jnp.real(v) # return eig matrix, eigenvec matrix

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
    ck = jnp.array(c(s,u,t),ndmin=2)
    cs = jnp.array(cs,ndmin=2).T 
    cu = jnp.array(cu,ndmin=2).T
    Q = jnp.vstack((jnp.hstack((css, cs)),jnp.hstack((cs.T, 2*ck))))
    R = cuu
    S = jnp.vstack((csu,cu.T))
    
    return Q,R,S

def discretize(f,dt):
    """ Make discretized dynamics function from discrete function, `f(s,u)` with time step `dt` 
        using Runge-Kutta 4th order method.
    """
    def integrator(s, u, dt=dt):
        k1 = dt * f(s, u)
        k2 = dt * f(s + k1 / 2, u)
        k3 = dt * f(s + k2 / 2, u)
        k4 = dt * f(s + k3, u)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    return integrator
    
def LQR_iteration(fd: callable, c: callable, N: int, u_bar:np.ndarray, s_bar:np.ndarray, s0: np.ndarray, u_range:np.ndarray, λ:float, dt:float):
    """ Solve a single LQR sub-problem with discretized dynamics `fd(s,u)`, cost function
        `c(s,u,t)`, reference linear control law `F` for augmented delta dynamics, reference control `u_bar`,
        reference trajectory `s_bar`, initial state `s0`, and Levenberg-Marquardt constant λ.
        
        Returns the linear feedback law at each time step for the augmented dynamics vector,
        [s, 1].T
    """
    P = np.zeros((N, s0.size + 1, s0.size + 1))
    P[-1] = quadratize_cost(c, s_bar[N-1], u_bar[N-1],dt*(N-1))[0] # initialize P matrix for Ricatti recursion, augmented for affine dynamics/cost
    L = np.zeros((N, u_bar.shape[1], s0.size + 1)) # initialize output linear control matrix for augmented dynamics
    for i in range(N-2,-1,-1):
        A, B = linearize_dynamics(fd, s_bar[i], u_bar[i])
        Q, R, S = quadratize_cost(c, s_bar[i], u_bar[i],i*dt)
        D, V = regularize(R+B.T@P[i+1]@B, λ) # Levenberg-Marquardt-esque regularization
        Hinv = V@(1/D)@V.T # compute quick inverse using eigen decomp values from regularization
        L[i] = -Hinv@(B.T@P[i+1]@A + S.T) # linear control law for augmented delta-dynamics
        for j in range(u_bar.shape[1]):
            if u_bar[i,j] <= u_range[j,0] or u_bar[i,j] >= u_range[j,1]:
                L[i,j,:] = 0 # control constraints impact choice of optimal L
        P[i] = Q + A.T@P[i+1]@A + (A.T@P[i+1]@B + S)@L[i] # quadratic cost matrix for augmented delta-dynamics

    new_s_bar, new_u_bar = simulate(fd, L, u_bar, s_bar, s0, N, u_range) # forward pass - simulate reference control foward using augmented dynamics
    cost = 0
    for i in range(N): # compute cost
        cost += c(new_s_bar[i], new_u_bar[i],dt*i)
    return L, new_u_bar, new_s_bar, cost
    
def simulate(fd:callable, L: np.ndarray, u_bar: np.ndarray, s_bar: np.ndarray, s0: np.ndarray, N:int, u_range:np.ndarray):
    """ Simulate the results of a linear control matrix `F` for `N` time steps via the discretized 
        delta dynamics function 'fd(s,u)' given an initial state, `s0`. 
    """
    s = np.zeros((N+1,s0.size))
    u = np.zeros((N,u_bar.shape[1]))
    s[0] = s0
    for i in range(N):
        ds = np.array(s[i]-s_bar[i],ndmin=2).T # control law is based on distance from reference, s_bar
        u[i] = L[i]@jnp.vstack((ds,[[1]])) + u_bar[i] # output of control is difference from reference, u_bar
        u[i] = jnp.clip(u[i],u_range[:,0],u_range[:,1])  # impose control contraint
        s[i+1] = fd(s[i],u[i])
    return s, u

def iLQR(f:callable, c:callable, u_ref: np.ndarray, N: int, s0: np.ndarray, u_range:np.ndarray,
         dt:float = 1E-2, max_iters:int = 400, tol: float = 1E-4, λmax:float= 1E4 ):
    """ Solve the iLQR problem for a system with discrete dynamics `f(s,u)`, cost function
        `c(s,u)`, initial reference control `uref`, number of time steps `N+1`, initial state `s0`,
        time step `dt`, max number of iterations 'max_iters', and convergence tolerance `tol`.
    """
    fd = jax.jit(discretize(f,dt))
    # fd = f
    # Create initial reference trajectory from initial reference control, and compute its cost
    s_bar = np.zeros((N+1,s0.size))
    u_bar = u_ref
    cost = 0
    s_bar[0] = s0
    for i in range(N):
        s_bar[i+1] = fd(s_bar[i],u_bar[i])
        cost += c(s_bar[i],u_bar[i],i*dt)
                
    λbase = 1E-4
    λfactor = 10
    λ = λbase
    u_dif = np.inf
    print('Reference cost = %0.2e' % cost)
    for k in ( pbar := tqdm(range(max_iters)) ):
        pbar.set_description("λ = %.2e, u_dif = %.2e, best cost-so-far = %0.2e" % (λ,u_dif,cost))
        F, new_u_bar, new_s_bar, new_cost = LQR_iteration(fd, c, N, u_bar, s_bar, s0, u_range, λ, dt)
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
        
            