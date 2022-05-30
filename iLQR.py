import numpy as np
import cvxpy as cvx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from functools import partial
import cvxpy as cvx

@partial(jax.jit)
def regularize(A: jnp.ndarray, λ:float):
    w,v = jnp.linalg.eig(A)
    w = jnp.real(w)
    w = jnp.maximum(jnp.zeros_like(w), w) + λ
    winv = 1/w
    D = jnp.diag(w)
    Dinv = jnp.diag(winv)
    V = jnp.real(v)
    return D, Dinv, V # return inverse eig matrix, eigenvec matrix

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

@partial(jax.jit, static_argnums=(0,))
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

def LQR_iteration(fd: callable, c: callable, h: callable, N: int, u_bar:np.ndarray, s_bar:np.ndarray, s0: np.ndarray, u_range:np.ndarray, λ:float, tspan:np.ndarray, near_sol:bool = False, tstart:float = 0):
    """ Solve a single LQR sub-problem with discretized dynamics `fd(s,u)`, cost function
        `c(s,u,t)`, reference linear control law `F` for augmented delta dynamics, reference control `u_bar`,
        reference trajectory `s_bar`, initial state `s0`, and Levenberg-Marquardt constant λ.
        
        Returns the linear feedback law at each time step for the augmented dynamics vector,
        [s, 1].T
    """
    n = s0.size
    m = u_bar.shape[1]
    # Initialize running cost and control matrices
    S = np.zeros((N+1, n, n))
    ς = np.zeros((N+1, n, 1))
    s = np.zeros((N+1, 1, 1))
    L = np.zeros((N, m, n)) # initialize output linear control matrix 
    l = np.zeros((N, m, 1)) # initialize output linear control affine term
    
    
    # Calculate terminal cost from quadratizing `h(s)`
    S[-1], ς[-1], s[-1] = quadratize_cost(lambda s,u,t: h(s), s_bar[N], jnp.zeros_like(u_bar[0]), tspan[-1])[:3] # get terminal state cost
    
    # Backwards pass, recursing on cost-to-go
    for i in range(N-1,-1,-1):
        # Linearize and quadratize
        A, B = linearize_dynamics(fd, s_bar[i], u_bar[i])
        Q, θ, q, R, r, P = quadratize_cost(c, s_bar[i], u_bar[i], tspan[i])
        
        # Form intermediates
        H = R + B.T@S[i+1]@B
        G = P + B.T@S[i+1]@A
        g = r + B.T@ς[i+1]
        
        # Regularize H matrix to make it positive-definite
        D, Dinv, V = regularize(H, λ) # Levenberg-Marquardt-esque regularization
        ℋ = V@D@V.T # regularized version of H, guaranteed positive semi-definite
        ℋinv = V@Dinv@V.T # compute quick inverse using eigen decomp values from regularization        
        # Clip control to ensure new reference control doesn't go outside control limits
        if not near_sol: # if not near a solution
            # do simple back trace
            dustar = -ℋinv@g
            upperε = jnp.amin(jnp.where(dustar + u_bar[i] > u_range[:,1], (u_range[:,1] - u_bar[i])/dustar , 1))
            lowerε = jnp.amin(jnp.where(dustar + u_bar[i] < u_range[:,0], (u_range[:,0] - u_bar[i])/dustar , 1))
            ε = max(0,min(upperε,lowerε))
            uopt = ε*dustar
        else:
            x = cvx.Variable(m)
            prob = cvx.Problem(cvx.Minimize(0.5*cvx.quad_form(x, ℋ) + g.T@x),
                     [x >= u_range[:,0],
                      x <= u_range[:,1]])
            prob.solve()
            uopt = x.value

        # Check output to ensure its in range!
        l[i] = uopt.reshape((m,1))
        L[i] = -ℋinv@G 
        for j in range(u_bar.shape[1]): # for each dimension of control
            if (l[i,j] + u_bar[i,j]) <= u_range[j,0] or (l[i,j] + u_bar[i,j]) >= u_range[j,1]:
                L[i,j,:] = 0 # control constraints impact choice of optimal L
                
        # Compute cost-to-go components
        S[i] = Q + A.T@S[i+1]@A + L[i].T@H@L[i] + L[i].T@G + G.T@L[i] # quadratic cost matrix for augmented delta-dynamics
        ς[i] = θ + A.T@ς[i+1] + L[i].T@g + G.T@l[i]
        s[i] = q + s[i+1] + 0.5*l[i].T@H@l[i] + l[i].T@g
    
    # Forward pass, simulating new control to get new reference trajectory
    new_s_bar, new_u_bar, new_cost = simulate(fd, c, h, L, l, u_bar, s_bar, s0, N, u_range, tspan) # forward pass - simulate reference control foward using augmented dynamics
    return L, l, new_u_bar, new_s_bar, new_cost

def simulate(fd:callable, c:callable, h:callable, L: np.ndarray, l:np.ndarray, u_bar: np.ndarray, s_bar: np.ndarray, s0: np.ndarray, N:int, u_range:np.ndarray, tspan:np.ndarray):
    """ Simulate the results of a linear control matrix `F` for `N` time steps via the discretized 
        delta dynamics function 'fd(s,u)' given an initial state, `s0`. 
    """
    s = np.zeros((N+1,s0.size))
    u = np.zeros((N,u_bar.shape[1]))
    s[0] = s0
    cost = 0
    for i in range(N):
        ds = jnp.array(s[i]-s_bar[i],ndmin=2).T # control law is based on distance from reference, s_bar
        u[i] = jnp.squeeze( L[i]@ds + l[i] + jnp.array(u_bar[i],ndmin=2).T ) # output of control is difference from reference, u_bar
        u[i] = jnp.clip(u[i],u_range[:,0],u_range[:,1])  # impose control contraint
        cost += c(s[i],u[i],tspan[i])
        s[i+1] = s[i] + fd(s[i],u[i]) # use discretized function to integrate
    cost += h(s[N])
    return s, u, cost

def iLQR(f:callable, c:callable, h:callable, u_ref: np.ndarray, N: int, s0: np.ndarray, u_range:np.ndarray,
         tspan:np.ndarray, max_iters:int = 400, tol: float = 1E-4, λmax:float= 1000):
    """ Solve the iLQR problem for a system with discrete dynamics `f(s,u)`, cost function
        `c(s,u)`, terminal cost `h(s)`, initial reference control `uref`, number of time steps `N+1`, initial state `s0`,
        time step `dt`, max number of iterations 'max_iters', and convergence tolerance `tol`.
    """
    dt = tspan[1]-tspan[0] # assume regular time step size
    fd = discretize(f,dt) # discretized function for true dynamics (used for simulation)
    
    # Create initial reference trajectory from initial reference control, and compute its cost
    s_bar = np.zeros((N+1,s0.size))
    u_bar = u_ref
    cost = np.inf
    s_bar[0] = s0
    
    # Simulate reference control to get reference trajectory
    for i in range(N):
        s_bar[i+1] = s_bar[i] + fd(s_bar[i],u_bar[i]) 
                
    # Iterate on LQR solutions until convergence in cost change and control change
    λbase = 1E-2
    λfactor = 10
    λ = λbase
    u_dif = 0
    cost_dif = 0
    pbar = tqdm(range(max_iters))
    pbar.set_description("λ = %.2e, u_dif = %.2e, cost_dif = %.2e, best cost-so-far = %0.5e" % (λ,u_dif,cost_dif,cost))
    for k in pbar:
        near_sol = cost_dif > 0 and cost_dif < 1.1*tol and u_dif > 0 and u_dif < 1.1*tol and k > 0 and not np.isinf(cost) # if within order of magnitude, use optimal solver instead of back-tracer
        if near_sol:
            print('Near solution! Moving to more optimal solver...')
        L, l, new_u_bar, new_s_bar, new_cost = LQR_iteration(fd, c, h, N, u_bar, s_bar, s0, u_range, λ, tspan, near_sol = near_sol) 
        if k > 0 and not np.isinf(cost):
            u_dif = np.amax(np.linalg.norm((u_bar - new_u_bar),np.inf,axis=1))
            cost_dif = abs((new_cost-cost)/cost)
            pbar.set_description("λ = %.2e, u_dif = %.2e, cost_dif = %.5e, best cost-so-far = %0.5e" % (λ,u_dif,cost_dif,cost))
            if cost_dif < tol and u_dif < tol: # if cost has converged, minimum 2 iterations to ensure convergence
                print("iLQR converged after {0} iterations with cost {1:.2e}.".format(k+1,cost))
                return new_s_bar, new_u_bar, L, l  # return control law, reference control, and reference trajectory
            
        if k == max_iters - 1:
            print("WARNING: Sub-optimal output. iLQR failed to converge after max iterations. Terminal λ value is {0} with best-cost-so-far {1:.2e}".format(λ,new_cost))
            return new_s_bar, new_u_bar, L, l 
        
        if new_cost < cost:
            λ = max(λbase,λ/λfactor)  # decrease "line-search" parameter to favor the optimal descent along Hession
            u_bar, s_bar, cost = new_u_bar, new_s_bar, new_cost # update reference trajectory and best-cost-so-far         
        else:
            if λ >= λmax: # if already tried this, don't bother trying again
                print("WARNING: Sub-optimal output. iLQR failed to converge, cannot further increase λ above λmax, {0}. Best-cost-so-far {1:.2e}.".format(λ,new_cost))
                return new_s_bar, new_u_bar, L, l 
            λ = λmax # increase "line-search" parameter to move along gradient
        
            