"""
1.5D Godunov Finite Volume Fluid–Particle Solver
=================================================
(header unchanged — see original for full docstring)
"""

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

GAMMA = 1.4
ShearBox = namedtuple("ShearBox", ["Omega", "q"])


class Particle_type:
    def __init__(self, xs, vxs, vys, tstop=1., mass=1.):
        self.np    = len(xs)
        self.xs    = np.asarray(xs,   dtype=float)
        self.vxs   = np.asarray(vxs,  dtype=float)
        self.vys   = np.asarray(vys,  dtype=float)
        self.tstop = tstop
        self.mass  = mass

    def tot_M(self):
        return np.sum(self.vxs * self.mass)


def prim_to_cons(rho, ux, uy, p):
    E = p / (GAMMA - 1.0) + 0.5 * rho * (ux**2 + uy**2)
    return np.array([rho, rho*ux, rho*uy, E])


def cons_to_prim(U):
    rho = U[0]; ux = U[1]/rho; uy = U[2]/rho; E = U[3]
    p   = (GAMMA - 1.0) * (E - 0.5*rho*(ux**2 + uy**2))
    p   = np.maximum(p,   1.0e-14)
    rho = np.maximum(rho, 1.0e-14)
    return rho, ux, uy, p


def sound_speed(rho, p):
    return np.sqrt(GAMMA * np.maximum(p, 1.0e-14) / np.maximum(rho, 1.0e-14))


def enforce_isothermal(U, cs_iso):
    rho  = np.maximum(U[0], 1.0e-14)
    p    = rho * cs_iso**2 / GAMMA
    U[3] = p / (GAMMA - 1.0) + 0.5 * (U[1]**2 + U[2]**2) / rho
    return U


def euler_flux(rho, ux, uy, p):
    E = p / (GAMMA - 1.0) + 0.5 * rho * (ux**2 + uy**2)
    return np.array([rho*ux, rho*ux**2+p, rho*ux*uy, ux*(E+p)])


def riemann_hllc(UL, UR):
    rhoL, uxL, uyL, pL = cons_to_prim(UL)
    rhoR, uxR, uyR, pR = cons_to_prim(UR)
    cL = sound_speed(rhoL, pL); cR = sound_speed(rhoR, pR)
    rho_bar = 0.5*(rhoL+rhoR); c_bar = 0.5*(cL+cR)
    p_star  = np.maximum(0.0, 0.5*(pL+pR) - 0.5*(uxR-uxL)*rho_bar*c_bar)
    qL = np.where(p_star>pL, np.sqrt(1.0+(GAMMA+1)/(2*GAMMA)*(p_star/pL-1)), 1.0)
    qR = np.where(p_star>pR, np.sqrt(1.0+(GAMMA+1)/(2*GAMMA)*(p_star/pR-1)), 1.0)
    SL = uxL - cL*qL; SR = uxR + cR*qR
    denom  = rhoL*(SL-uxL) - rhoR*(SR-uxR)
    S_star = (pR-pL + rhoL*uxL*(SL-uxL) - rhoR*uxR*(SR-uxR)) / denom
    FL = euler_flux(rhoL, uxL, uyL, pL)
    FR = euler_flux(rhoR, uxR, uyR, pR)
    def star_state(U, rho, ux, uy, p, S):
        f   = rho*(S-ux)/(S-S_star)
        Esp = U[3]/rho
        return np.array([f, f*S_star, f*uy,
                         f*(Esp+(S_star-ux)*(S_star+p/(rho*(S-ux))))])
    U_starL = star_state(UL, rhoL, uxL, uyL, pL, SL)
    U_starR = star_state(UR, rhoR, uxR, uyR, pR, SR)
    return np.where(SL>=0., FL,
           np.where(S_star>=0., FL+SL*(U_starL-UL),
           np.where(SR>=0.,    FR+SR*(U_starR-UR), FR)))


def _weno_z_weights(vm2, vm1, v0, vp1, vp2):
    eps = 1.0e-36
    q0 = (2.*vm2 - 7.*vm1 + 11.*v0)/6.
    q1 = (-vm1 + 5.*v0 + 2.*vp1)/6.
    q2 = (2.*v0 + 5.*vp1 - vp2)/6.
    b0 = (13./12.)*(vm2-2.*vm1+v0)**2 + (1./4.)*(vm2-4.*vm1+3.*v0)**2
    b1 = (13./12.)*(vm1-2.*v0+vp1)**2 + (1./4.)*(vm1-vp1)**2
    b2 = (13./12.)*(v0-2.*vp1+vp2)**2 + (1./4.)*(3.*v0-4.*vp1+vp2)**2
    tau5 = np.abs(b0-b2); d0,d1,d2 = 1./10., 3./5., 3./10.
    a0=d0*(1.+tau5/(eps+b0)); a1=d1*(1.+tau5/(eps+b1)); a2=d2*(1.+tau5/(eps+b2))
    return (a0*q0+a1*q1+a2*q2)/(a0+a1+a2)


def _weno_zpi_weights(vm2, vm1, v0, vp1, vp2):
    eps = 1.0e-40; lam = 3.3
    q0 = (2.*vm2 - 7.*vm1 + 11.*v0)/6.
    q1 = (-vm1 + 5.*v0 + 2.*vp1)/6.
    q2 = (2.*v0 + 5.*vp1 - vp2)/6.
    b0 = (13./12.)*(vm2-2.*vm1+v0)**2 + (1./4.)*(vm2-4.*vm1+3.*v0)**2
    b1 = (13./12.)*(vm1-2.*v0+vp1)**2 + (1./4.)*(vm1-vp1)**2
    b2 = (13./12.)*(v0-2.*vp1+vp2)**2 + (1./4.)*(3.*v0-4.*vp1+vp2)**2
    tau5  = np.abs(b0-b2)
    b_max = np.maximum(np.maximum(b0,b1),b2)
    b_min = np.minimum(np.minimum(b0,b1),b2)
    aa    = 1. - b_min/(b_max+eps); extra = lam*aa/(b_max+eps)
    d0,d1,d2 = 1./10., 3./5., 3./10.
    a0=d0*(1.+(tau5/(eps+b0))**2+extra*b0)
    a1=d1*(1.+(tau5/(eps+b1))**2+extra*b1)
    a2=d2*(1.+(tau5/(eps+b2))**2+extra*b2)
    return (a0*q0+a1*q1+a2*q2)/(a0+a1+a2)


def reconstruct_weno_z(U, weight_fn):
    n = U.shape[1]; n_faces = n-1
    UL = np.empty((U.shape[0], n_faces)); UR = np.empty((U.shape[0], n_faces))
    for k in range(2, n_faces-2):
        UL[:,k] = weight_fn(U[:,k-2],U[:,k-1],U[:,k],U[:,k+1],U[:,k+2])
        UR[:,k] = weight_fn(U[:,k+3],U[:,k+2],U[:,k+1],U[:,k],U[:,k-1])
    for k in [0,1,n_faces-2,n_faces-1]:
        UL[:,k] = U[:,k]; UR[:,k] = U[:,k+1]
    return UL, UR


def apply_bc_gen(U, bc="transmissive"):
    if bc == "transmissive":
        U[0]=U[2]; U[1]=U[2]; U[-1]=U[-3]; U[-2]=U[-3]
    elif bc == "reflective":
        U[0]=U[2]; U[0]=-U[2]; U[1]=U[2]; U[1]=-U[2]
        U[-1]=U[-3]; U[-1]=-U[-3]; U[-2]=U[-3]; U[-2]=-U[-3]
    elif bc in ("periodic","shear_periodic"):
        U[0]=U[-4]; U[1]=U[-3]; U[-2]=U[2]; U[-1]=U[3]
    else:
        raise ValueError(f"Unknown BC: {bc!r}")
    return U


def apply_bc(U, bc="transmissive"):
    if bc == "transmissive":
        U[:,0]=U[:,2]; U[:,1]=U[:,2]; U[:,-1]=U[:,-3]; U[:,-2]=U[:,-3]
    elif bc == "reflective":
        U[:,0]=U[:,2];  U[1,0]=-U[1,2]
        U[:,1]=U[:,2];  U[1,1]=-U[1,2]
        U[:,-1]=U[:,-3]; U[1,-1]=-U[1,-3]
        U[:,-2]=U[:,-3]; U[1,-2]=-U[1,-3]
    elif bc in ("periodic","shear_periodic"):
        U[:,0]=U[:,-4]; U[:,1]=U[:,-3]; U[:,-2]=U[:,2]; U[:,-1]=U[:,3]
    else:
        raise ValueError(f"Unknown BC: {bc!r}")
    return U


def spatial_residual(U, dx, bc, recons='WENOZpI',
                     gas_sources=[], gas_source_params=[]):
    U = apply_bc(U, bc)
    n_cells = U.shape[1]; F = np.zeros_like(U)
    if recons in ('WENOZ','WENOZpI'):
        wfn = _weno_z_weights if recons=='WENOZ' else _weno_zpi_weights
        UL_all, UR_all = reconstruct_weno_z(U, wfn)
    for i in range(1, n_cells-1):
        match recons:
            case 'godunov':  F[:,i] = riemann_hllc(U[:,i], U[:,i+1])
            case 'WENOZ':    F[:,i] = riemann_hllc(UL_all[:,i], UR_all[:,i])
            case 'WENOZpI':  F[:,i] = riemann_hllc(UL_all[:,i], UR_all[:,i])
            case _: raise ValueError(f"Unknown recons {recons!r}")
    L = np.zeros_like(U)
    L[:,2:-2] = -(1./dx)*(F[:,2:-2]-F[:,1:-3])
    for source, params in zip(gas_sources, gas_source_params):
        L += source(U, params)
    return L


def compute_dt(U, dx, cfl=0.9, particles=None):
    rho, ux, uy, p = cons_to_prim(U)
    c = sound_speed(rho, p); S_max = np.max(np.abs(ux)+c)
    dt = cfl*dx/S_max if S_max > 1.0e-14 else 1.0e-6
    if particles is not None:
        for p_type in particles:
            if p_type.np == 0: continue
            vx_max = np.max(np.abs(p_type.vxs))
            if vx_max > 1.0e-14: dt = min(dt, cfl*dx/vx_max)
    return dt


def _ssprk_tableau(name):
    name = name.lower()
    if name == "ssprk1":
        s=1; alpha=np.zeros((2,1)); beta=np.zeros((2,1))
        alpha[1,0]=1.; beta[1,0]=1.
    elif name == "ssprk2":
        s=2; alpha=np.zeros((3,2)); beta=np.zeros((3,2))
        alpha[1,0]=1.; beta[1,0]=1.
        alpha[2,0]=.5; alpha[2,1]=.5; beta[2,1]=.5
    elif name == "ssprk3":
        s=3; alpha=np.zeros((4,3)); beta=np.zeros((4,3))
        alpha[1,0]=1.; beta[1,0]=1.
        alpha[2,0]=3/4; alpha[2,1]=1/4; beta[2,1]=1/4
        alpha[3,0]=1/3; alpha[3,2]=2/3; beta[3,2]=2/3
    elif name == "ssprk4":
        s=4; alpha=np.zeros((5,4)); beta=np.zeros((5,4))
        alpha[1,0]=1.; beta[1,0]=1/2
        alpha[2,1]=1.; beta[2,1]=1/2
        alpha[3,0]=2/3; alpha[3,2]=1/3; beta[3,2]=1/6
        alpha[4,3]=1.; beta[4,3]=1/2
    elif name == "ssprk5":
        s=5; alpha=np.zeros((6,5)); beta=np.zeros((6,5))
        alpha[1,0]=1.; beta[1,0]=0.39175222700392
        alpha[2,0]=0.44437049406734; alpha[2,1]=0.55562950593266; beta[2,1]=0.36841059262959
        alpha[3,0]=0.62010185138540; alpha[3,2]=0.37989814861460; beta[3,2]=0.25189177424738
        alpha[4,0]=0.17807995410773; alpha[4,3]=0.82192004589227; beta[4,3]=0.54497475021237
        alpha[5,0]=0.00683325884039; alpha[5,2]=0.51723167208978
        alpha[5,4]=0.47591583767062-1e-15
        beta[5,2]=0.12759831133288; beta[5,4]=0.08460416338212
    else:
        raise ValueError(f"Unknown SSPRK {name!r}")
    return alpha, beta


def _tsc_weights(x_p, x0, dx):
    x_p = np.asarray(x_p, dtype=float); scalar = x_p.ndim==0
    x_p = np.atleast_1d(x_p)
    eta = (x_p-x0)/dx; i_c = np.round(eta).astype(int); delta = eta-i_c
    w = np.empty((x_p.size,3))
    w[:,0]=0.5*(0.5-delta)**2; w[:,1]=0.75-delta**2; w[:,2]=0.5*(0.5+delta)**2
    if scalar: return i_c[0], w[0]
    return i_c, w


def grid_to_particles(q_grid, x_grid, x_particles):
    q_grid=np.asarray(q_grid,dtype=float); x_grid=np.asarray(x_grid,dtype=float)
    x_particles=np.asarray(x_particles,dtype=float)
    N=q_grid.size; dx=x_grid[1]-x_grid[0]; x0=x_grid[0]
    i_c, w = _tsc_weights(x_particles, x0, dx)
    idx = np.stack([np.clip(i_c-1,0,N-1), np.clip(i_c,0,N-1), np.clip(i_c+1,0,N-1)], axis=1)
    return np.einsum("pj,pj->p", w, q_grid[idx])


def particles_to_grid(q_particles, weights_p, x_particles, x_grid):
    q_particles=np.asarray(q_particles,dtype=float)
    x_particles=np.asarray(x_particles,dtype=float)
    x_grid=np.asarray(x_grid,dtype=float)
    N_p=q_particles.size; N=x_grid.size; dx=x_grid[1]-x_grid[0]; x0=x_grid[0]
    weights_p = np.ones(N_p,dtype=float) if weights_p is None else np.asarray(weights_p,dtype=float)
    i_c, w = _tsc_weights(x_particles, x0, dx)
    eff_w = w * weights_p[:,np.newaxis]; q_grid = np.zeros(N, dtype=float)
    for s in range(3):
        idx = np.clip(i_c+s-1, 0, N-1)
        np.add.at(q_grid, idx, eff_w[:,s]*q_particles)
    return q_grid


def _fold_ghost_deposits(deposit, Nx, bc):
    if bc in ('periodic','shear_periodic'):
        deposit[Nx]   += deposit[0]; deposit[Nx+1] += deposit[1]
        deposit[0]=0.; deposit[1]=0.
        deposit[2]    += deposit[Nx+2]; deposit[3] += deposit[Nx+3]
        deposit[Nx+2]=0.; deposit[Nx+3]=0.
    return deposit


def shearing_box_force(x_p, vx_p, vy_p, vgx_p, vgy_p, params):
    sb = params
    return 2.*sb.Omega*vy_p, -(2.-sb.q)*sb.Omega*vx_p


def gas_shearing_box_source(U, params):
    sb, eta = params
    rho = U[0,2:-2]; ux = U[1,2:-2]/rho; uy = U[2,2:-2]/rho
    S = np.zeros_like(U)
    S[1,2:-2] = rho*(2.*sb.Omega*uy + 2.*eta*sb.Omega**2)
    S[2,2:-2] = rho*(-(2.-sb.q)*sb.Omega*ux)
    return S


def particles_step(particles, U_n, U_half, x_grid, dt,
                   bc='outflow', integrator="EM",
                   forces=[], force_params=[],
                   shear_box=None, analytic_x=False):
    """
    Advance all particle species by one time step dt.

    Uses _em_velocity_update (Cases 1/2/3) to compute new particle
    velocities and deposits the drag-only momentum change onto the gas.

    Parameters
    ----------
    particles    : list of Particle_type   (modified in place)
    U_n          : ndarray (4, Nx+4)       conserved state at time n (padded)
    U_half       : ndarray (4, Nx+4)       half-time state (padded, BCs applied)
    x_grid       : ndarray (Nx,)           physical cell-centre positions
    dt           : float                   time step
    bc           : str                     boundary condition
    integrator   : str                     'EM' (only EM implemented here)
    forces       : list                    external force functions
    force_params : list                    parameters for each force function
    shear_box    : ShearBox or None        for shear_periodic BC velocity kick
    analytic_x   : bool                    use analytic position update

    Returns
    -------
    particles     : list of Particle_type  (updated)
    mom_x_deposit : ndarray (Nx+4,)        x-momentum change on padded grid
    mom_y_deposit : ndarray (Nx+4,)        y-momentum change on padded grid
    """
    dx     = x_grid[1] - x_grid[0]
    x_grid = np.copy(x_grid)
    Nx     = len(x_grid)

    x_phys_min = x_grid[0] - dx/2
    x_phys_max = x_grid[-1] + dx/2
    L_domain   = x_phys_max - x_phys_min

    x_grid_pad = np.concatenate([
        [x_grid[0] - 2*dx, x_grid[0] - dx],
        x_grid,
        [x_grid[-1] + dx,  x_grid[-1] + 2*dx],
    ])

    mom_x_deposit = np.zeros(len(x_grid_pad))
    mom_y_deposit = np.zeros(len(x_grid_pad))

    for p_index, p_type in enumerate(particles):

        xn  = p_type.xs.copy()
        vxn = p_type.vxs.copy()
        vyn = p_type.vys.copy()

        # Half-step positions (leapfrog, wrapped for periodic BCs)
        x_p_halfs = p_type.xs + p_type.vxs * dt * 0.5
        if bc in ('periodic', 'shear_periodic'):
            x_p_halfs = (x_phys_min
                         + (x_p_halfs - x_phys_min) % L_domain)

        # Gas velocity at t^n, interpolated to particle positions
        ugx_n = U_n[1] / U_n[0]
        ugy_n = U_n[2] / U_n[0]
        ugx_n_at_p = grid_to_particles(ugx_n, x_grid_pad, p_type.xs)
        ugy_n_at_p = grid_to_particles(ugy_n, x_grid_pad, p_type.xs)

        ugx_half = U_half[1] / U_half[0]
        ugy_half = U_half[2] / U_half[0]
        ugx_half_at_p = grid_to_particles(ugx_half, x_grid_pad, x_p_halfs)
        ugy_half_at_p = grid_to_particles(ugy_half, x_grid_pad, x_p_halfs)

        #p_type.xs = x_p_halfs

        # Dust-to-gas ratio at particle positions
        eps_at_p = _compute_eps_at_p(U_n, particles, p_index, x_grid_pad, bc=bc)

        # EM velocity update (Cases 1 / 2 / 3)
        vxf, vyf, h1G, f_mid_x, f_mid_y = _em_velocity_update(
            vxn, vyn,
            ugx_n_at_p, ugy_n_at_p,
            eps_at_p, dt, p_type.tstop,
            forces, force_params, x_p_halfs,
        )

        # Drag-only momentum change (subtract force contribution h1G*f_mid)
        # The gas does not directly receive external particle forces;
        # those are already included in gas_sources.
        delta_px = p_type.mass * (vxf - vxn - h1G * f_mid_x)
        delta_py = p_type.mass * (vyf - vyn - h1G * f_mid_y)

        p_type.vxs = vxf
        p_type.vys = vyf

        if analytic_x:
            ts  = p_type.tstop
            emf = 1.0 - np.exp(-dt / ts)
            vtx = f_mid_x * ts + ugx_n_at_p
            vty = f_mid_y * ts + ugy_n_at_p
            p_type.xs = xn + vtx*dt + (vxn - vtx)*ts*emf
        else:
            p_type.xs = x_p_halfs + p_type.vxs * dt * 0.5

        if bc == 'periodic':
            p_type.xs = (x_phys_min
                         + (p_type.xs - x_phys_min) % L_domain)
        elif bc == 'shear_periodic':
            crossed_right = p_type.xs >= x_phys_min + L_domain
            crossed_left  = p_type.xs <  x_phys_min
            p_type.xs = (x_phys_min
                         + (p_type.xs - x_phys_min) % L_domain)
            if shear_box is not None:
                delta_vy = shear_box.q * shear_box.Omega * L_domain
                p_type.vys[crossed_right] += delta_vy
                p_type.vys[crossed_left ] -= delta_vy

        mom_x_deposit += particles_to_grid(-delta_px, None, x_p_halfs, x_grid_pad)
        mom_y_deposit += particles_to_grid(-delta_py, None, x_p_halfs, x_grid_pad)

    mom_x_deposit = _fold_ghost_deposits(mom_x_deposit, Nx, bc)
    mom_y_deposit = _fold_ghost_deposits(mom_y_deposit, Nx, bc)

    return particles, mom_x_deposit, mom_y_deposit

def _em_velocity_update(vxn, vyn, ugx_n_at_p, ugy_n_at_p,
                         eps_at_p, dt, tstop,
                         forces, force_params, x_p_halfs):
    """
    Exponential midpoint (EM) velocity update for one particle species.

    Solves the coupled drag + external-force ODE for (vx, vy) over one
    time step dt.  Three cases are handled in order of specialisation:

    Case 1 -- no external forces
        Pure drag, closed-form solution of the (delta_v, U_cm) system.
            v^{n+1} = [(E + eps)*v^n + (1-E)*u^n] / (1+eps)
        where E = exp(-(1+eps)*dt/tau_s).

    Case 2 -- single shearing-box force  (forces=[shearing_box_force])
        Linear Coriolis + epicyclic.  Substituting the implicit midpoint
        v_mid = (v^n + v^{n+1})/2 into the EM update gives a 2x2 linear
        system that is solved exactly.

    Case 3 -- general force f(x, v)
        Fixed-point iteration (up to 5 steps) with a Newton fallback
        (up to 6 steps).

    In all cases the gas velocity used is ugx_n_at_p / ugy_n_at_p.
    The caller supplies the appropriate gas state (U_n for the predictor,
    U_n again for the corrector -- the half-time gas state enters via
    the positions x_p_halfs used to evaluate position-dependent forces,
    not directly in the velocity formula).

    Parameters
    ----------
    vxn, vyn         : ndarray (N_p,)  particle velocity at step start
    ugx_n_at_p,
    ugy_n_at_p       : ndarray (N_p,)  gas velocity interpolated to particles
                                       (from U_n for both predictor and corrector)
    eps_at_p         : ndarray (N_p,)  local dust-to-gas ratio eps at particles
    dt               : float           time step
    tstop            : float           aerodynamic stopping time tau_s
    forces           : list            external force functions
    force_params     : list            parameters for each force function
    x_p_halfs        : ndarray (N_p,)  half-step particle positions
                                       (position argument to force functions)

    Returns
    -------
    vxf      : ndarray (N_p,)  updated x-velocity
    vyf      : ndarray (N_p,)  updated y-velocity
    h1G      : ndarray or float
                               force integration weight h1G.
                               Zero (scalar 0.0) for Case 1.
                               Array for Cases 2/3.
    f_mid_x  : ndarray (N_p,)  midpoint x-force (zeros for Case 1)
    f_mid_y  : ndarray (N_p,)  midpoint y-force (zeros for Case 1)

    Notes
    -----
    The drag-only deposit used in both the predictor and corrector is:

        delta_px = mass * (vxf - vxn - h1G * f_mid_x)
        delta_py = mass * (vyf - vyn - h1G * f_mid_y)

    For Case 1: h1G = 0 and f_mid = 0, so delta_p = mass*(vf - vn).
    For Cases 2/3: subtracting h1G*f_mid isolates the drag impulse and
    prevents Coriolis (or other external) forces from being back-reacted
    onto the gas.  The gas already receives those forces via gas_sources.
    """
    one_p_eps = 1.0 + eps_at_p
    exp_scale = np.exp(-one_p_eps * dt / tstop)

    if not forces:
        # ------------------------------------------------------------------
        # Case 1: Pure drag
        # Closed-form solution with G = 0:
        #   v^{n+1} = [(E + eps)*v^n + (1-E)*u^n] / (1+eps)
        # ------------------------------------------------------------------
        vxf = (exp_scale + eps_at_p) * vxn / one_p_eps \
            + (1.0 - exp_scale) * ugx_n_at_p / one_p_eps
        vyf = (exp_scale + eps_at_p) * vyn / one_p_eps \
            + (1.0 - exp_scale) * ugy_n_at_p / one_p_eps
        h1G     = 0.0
        f_mid_x = np.zeros_like(vxn)
        f_mid_y = np.zeros_like(vyn)

    elif (len(forces) == 1
          and forces[0] is shearing_box_force):
        # ------------------------------------------------------------------
        # Case 2: Shearing box -- analytic 2x2 solve
        #
        # EM update with G = [2*Omega*vy, -(2-q)*Omega*vx]:
        #   v^{n+1} = base + h1G * G(v_mid),   v_mid = (v^n + v^{n+1})/2
        #
        # h1G = tau*(1-E)/(1+eps)^2 + eps*dt/(1+eps)
        # The eps*dt/(1+eps) term comes from the U_cm drift (dU_cm/dt = eps*G)
        # and is required for correct equilibrium structure.
        #
        # Substituting v_mid gives the 2x2 linear system:
        #   [1,     -Om1] [vxf]   [bx]
        #   [Om2,      1] [vyf] = [by]
        # with det = 1 + Om1*Om2.
        # ------------------------------------------------------------------
        sb = force_params[0]

        h1G = (tstop * (1.0 - exp_scale) / one_p_eps**2
               + eps_at_p * dt / one_p_eps)

        bx0 = ((exp_scale + eps_at_p) * vxn / one_p_eps
               + (1.0 - exp_scale) * ugx_n_at_p / one_p_eps)
        by0 = ((exp_scale + eps_at_p) * vyn / one_p_eps
               + (1.0 - exp_scale) * ugy_n_at_p / one_p_eps)

        Om1 = h1G * sb.Omega
        Om2 = h1G * (2.0 - sb.q) * sb.Omega * 0.5

        bx  = bx0 + Om1 * vyn
        by  = by0 - Om2 * vxn

        det = 1.0 + Om1 * Om2
        vxf = (bx + Om1 * by) / det
        vyf = (by - Om2 * bx) / det

        # Midpoint velocity and force (needed to isolate drag in deposit)
        vmx = 0.5 * (vxn + vxf)
        vmy = 0.5 * (vyn + vyf)
        f_mid_x =  2.0 * sb.Omega * vmy
        f_mid_y = -(2.0 - sb.q) * sb.Omega * vmx

    else:
        # ------------------------------------------------------------------
        # Case 3: General f(x, v) -- fixed-point iteration, Newton fallback
        # ------------------------------------------------------------------
        h1G = (tstop * (1.0 - exp_scale) / one_p_eps**2
               + eps_at_p * dt / one_p_eps)

        def _eval_G(vx_eval, vy_eval):
            fx_s = np.zeros_like(vxn)
            fy_s = np.zeros_like(vyn)
            for j, force_fn in enumerate(forces):
                dfx, dfy = force_fn(
                    x_p_halfs, vx_eval, vy_eval,
                    ugx_n_at_p, ugy_n_at_p,
                    force_params[j])
                fx_s += dfx
                fy_s += dfy
            return fx_s, fy_s

        def _kick(fx, fy):
            """Base EM update + force contribution h1G*f."""
            return (
                (exp_scale + eps_at_p) * vxn / one_p_eps
                + (1.0 - exp_scale) * ugx_n_at_p / one_p_eps
                + h1G * fx,
                (exp_scale + eps_at_p) * vyn / one_p_eps
                + (1.0 - exp_scale) * ugy_n_at_p / one_p_eps
                + h1G * fy)

        # Initial guess: G evaluated at v^n
        fx0, fy0 = _eval_G(vxn, vyn)
        vxf, vyf = _kick(fx0, fy0)

        # Fixed-point iterations
        _fp_max      = 5
        _fp_conv     = False
        _fp_prev_err = np.inf
        for _fp_k in range(_fp_max):
            vmx = 0.5 * (vxn + vxf)
            vmy = 0.5 * (vyn + vyf)
            fx_s, fy_s = _eval_G(vmx, vmy)
            vxf_new, vyf_new = _kick(fx_s, fy_s)
            err = (np.max(np.abs(vxf_new - vxf))
                   + np.max(np.abs(vyf_new - vyf)))
            vxf, vyf = vxf_new, vyf_new
            if err < 1e-13 * (1.0 + np.max(np.abs(vxf))
                                   + np.max(np.abs(vyf))):
                _fp_conv = True
                break
            if err > 2.0 * _fp_prev_err and _fp_k >= 1:
                break
            _fp_prev_err = err

        # Newton fallback if fixed-point did not converge
        if not _fp_conv:
            fx0, fy0 = _eval_G(vxn, vyn)
            vxf, vyf = _kick(fx0, fy0)
            _eps_fd = 1e-7
            _nw_max = 6
            for _nw_k in range(_nw_max):
                vmx = 0.5 * (vxn + vxf)
                vmy = 0.5 * (vyn + vyf)
                fx_s, fy_s = _eval_G(vmx, vmy)
                kx, ky = _kick(fx_s, fy_s)
                Fx = vxf - kx;  Fy = vyf - ky
                res = np.max(np.abs(Fx)) + np.max(np.abs(Fy))
                if res < 1e-13 * (1.0 + np.max(np.abs(vxf))
                                       + np.max(np.abs(vyf))):
                    break
                sc_x = _eps_fd * (1.0 + np.abs(vxf))
                fx_p, fy_p = _eval_G(vmx + 0.5*sc_x, vmy)
                kxp, kyp = _kick(fx_p, fy_p)
                dkx_dvx = (kxp - kx) / sc_x
                dky_dvx = (kyp - ky) / sc_x
                sc_y = _eps_fd * (1.0 + np.abs(vyf))
                fx_p, fy_p = _eval_G(vmx, vmy + 0.5*sc_y)
                kxp, kyp = _kick(fx_p, fy_p)
                dkx_dvy = (kxp - kx) / sc_y
                dky_dvy = (kyp - ky) / sc_y
                J00 = 1.0 - dkx_dvx;  J01 =     - dkx_dvy
                J10 =     - dky_dvx;  J11 = 1.0 - dky_dvy
                det_J = J00*J11 - J01*J10
                det_J = np.where(np.abs(det_J) < 1e-30,
                                 np.sign(det_J)*1e-30 + 1e-30, det_J)
                dvx = -(J11*Fx - J01*Fy) / det_J
                dvy = -(J00*Fy - J10*Fx) / det_J
                vxf += dvx;  vyf += dvy

            vmx = 0.5*(vxn + vxf);  vmy = 0.5*(vyn + vyf)
            fx_s, fy_s = _eval_G(vmx, vmy)

        f_mid_x = fx_s
        f_mid_y = fy_s

    return vxf, vyf, h1G, f_mid_x, f_mid_y

def _compute_eps_at_p(U, particles, p_index, x_grid, bc="periodic"):
    dx = x_grid[1] - x_grid[0]
    if len(x_grid) < len(U[0]):
        Nx = len(x_grid)
        x_grid = np.concatenate([[x_grid[0]-2*dx, x_grid[0]-dx],
                                  x_grid,
                                  [x_grid[-1]+dx, x_grid[-1]+2*dx]])
    else:
        Nx = len(x_grid) - 4
    rho_gas  = np.maximum(U[0], 1.0e-14)
    eps_grid = np.zeros_like(rho_gas)
    for p_type in particles:
        rho_dust = particles_to_grid(p_type.mass*np.ones(p_type.np),
                                     None, p_type.xs, x_grid) / dx
        rho_dust = apply_bc_gen(_fold_ghost_deposits(rho_dust, Nx, bc), bc)
        eps_grid += rho_dust / rho_gas
    p_type        = particles[p_index]
    eps_particles = grid_to_particles(eps_grid, x_grid, p_type.xs)
    return eps_particles


def _particle_drag_predictor(particles, x_grid, U, dt, bc='periodic'):
    """
    Predictor drag source for the fluid's first SSPRK stage.
    Uses the pure-drag exponential (no forces) with full dt, matching
    the Stage 2 drag kernel of particles_step.
    """
    dx = x_grid[1] - x_grid[0]; Nx = len(x_grid)
    x_grid_pad = np.concatenate([[x_grid[0]-2*dx, x_grid[0]-dx],
                                  x_grid,
                                  [x_grid[-1]+dx, x_grid[-1]+2*dx]])
    source  = np.zeros_like(U)
    ugx_pad = U[1] / U[0]
    ugy_pad = U[2] / U[0]

    for p_index, p_type in enumerate(particles):
        eps_at_p  = _compute_eps_at_p(U, particles, p_index, x_grid, bc=bc)
        one_p_eps = 1.0 + eps_at_p
        exp_scale = np.exp(-one_p_eps * dt / p_type.tstop)

        ugx_at_p = grid_to_particles(ugx_pad, x_grid_pad, p_type.xs)
        ugy_at_p = grid_to_particles(ugy_pad, x_grid_pad, p_type.xs)

        ucm_x = ugx_at_p + eps_at_p * p_type.vxs
        ucm_y = ugy_at_p + eps_at_p * p_type.vys

        vx_new = exp_scale*(p_type.vxs - ugx_at_p)/one_p_eps + ucm_x/one_p_eps
        vy_new = exp_scale*(p_type.vys - ugy_at_p)/one_p_eps + ucm_y/one_p_eps

        delta_vx = vx_new - p_type.vxs
        delta_vy = vy_new - p_type.vys

        dep_x = particles_to_grid(-p_type.mass*delta_vx, None, p_type.xs, x_grid_pad)
        dep_y = particles_to_grid(-p_type.mass*delta_vy, None, p_type.xs, x_grid_pad)
        dep_x = _fold_ghost_deposits(dep_x, Nx, bc)
        dep_y = _fold_ghost_deposits(dep_y, Nx, bc)
        source[1] += dep_x / dx
        source[2] += dep_y / dx

    return source


def ssprk_step(U, dx, dt, bc="transmissive", tableau="ssprk3",
               recons='WENOZpI', particles=None, x_grid=None,
               dim=1.0, shear_box=None, forces=[], force_params=[],
               gas_sources=[], gas_source_params=[],
               integrator='EM', analytic_x=False, cs_iso=None):

    if particles is not None:
        tableau = "ssprk2"

    alpha, beta = _ssprk_tableau(tableau)
    s = alpha.shape[0] - 1
    U_n = U.copy()

    if particles is not None:
        if cs_iso is not None:
            enforce_isothermal(U_n, cs_iso)

        L_n  = spatial_residual(U_n, dx, bc, recons, gas_sources, gas_source_params)
        pred = _particle_drag_predictor(particles, x_grid, U_n, dt, bc)

        U_star = U_n + dt * L_n + pred
        U_half = 0.5 * (U_n + U_star)
        U_half = apply_bc(U_half, bc)

        particles, Mx_deposit, My_deposit = particles_step(
            particles, U_n, U_half, x_grid, dt,
            bc=bc, shear_box=shear_box,
            forces=forces, force_params=force_params,
            integrator=integrator, analytic_x=analytic_x,
        )

        L_star = spatial_residual(U_star, dx, bc, recons, gas_sources, gas_source_params)

        U_new = U_n + 0.5*dt*(L_star + L_n)
        U_new[1,2:-2] += Mx_deposit[2:-2] / dx
        U_new[2,2:-2] += My_deposit[2:-2] / dx
        if cs_iso is not None:
            enforce_isothermal(U_new, cs_iso)
        U_new = apply_bc(U_new, bc)

        return U_new, particles

    stages = [None]*(s+1); stages[0] = U_n
    for i in range(1, s+1):
        acc = np.zeros_like(U)
        for k in range(i):
            if alpha[i,k] != 0.: acc += alpha[i,k]*stages[k]
            if beta[i,k]  != 0.: acc += beta[i,k]*dt*spatial_residual(
                stages[k], dx, bc, recons, gas_sources, gas_source_params)
        if cs_iso is not None: enforce_isothermal(acc, cs_iso)
        stages[i] = acc
    return apply_bc(stages[s], bc)


def solve(rho_init, ux_init, p_init, x, t_end,
          uy_init=None, dt=1.e-2, cfl=0.9, bc="transmissive",
          ssprk="ssprk3", verbose=True, recons='WENOZpI',
          particles=None, shear_box=None,
          forces=[], force_params=[],
          gas_sources=[], gas_source_params=[],
          integrator='EM', analytic_x=False, cs_iso='auto'):

    cfl_eff = {"ssprk1":1.,"ssprk2":1.,"ssprk3":1.,"ssprk4":2.,"ssprk5":1.508}.get(ssprk.lower(),1.)
    rho_init=np.asarray(rho_init,dtype=float); ux_init=np.asarray(ux_init,dtype=float)
    p_init  =np.asarray(p_init,  dtype=float); x      =np.asarray(x,      dtype=float)
    uy_init = np.zeros_like(rho_init) if uy_init is None else np.asarray(uy_init,dtype=float)
    dx = x[1]-x[0]

    def pad(a): return np.concatenate([[a[0],a[0]],a,[a[-1],a[-1]]])
    U = prim_to_cons(pad(rho_init), pad(ux_init), pad(uy_init), pad(p_init))

    if cs_iso == 'auto':
        cs_iso = np.sqrt(np.mean(p_init)/np.mean(rho_init)) if particles is not None else None

    t = 0.; step = 0
    while t < t_end:
        dt_cfl = compute_dt(U[:,2:-2], dx, cfl*cfl_eff, particles=particles)
        dt     = min(dt_cfl, t_end-t)
        if particles is None:
            U = ssprk_step(U, dx, dt, bc, ssprk, recons,
                           gas_sources=gas_sources, gas_source_params=gas_source_params,
                           cs_iso=cs_iso)
        else:
            U, particles = ssprk_step(U, dx, dt, bc, ssprk, recons,
                                      particles, x, shear_box=shear_box,
                                      forces=forces, force_params=force_params,
                                      gas_sources=gas_sources, gas_source_params=gas_source_params,
                                      integrator=integrator, analytic_x=analytic_x, cs_iso=cs_iso)
        t += dt; step += 1
        if verbose and step%100==0:
            print(f"  step {step:5d}  t = {t:.4f}  dt = {dt:.2e}")

    rho_f, ux_f, uy_f, p_f = cons_to_prim(U[:,2:-2])
    return (rho_f,ux_f,uy_f,p_f,t) if particles is None else (rho_f,ux_f,uy_f,p_f,t,particles)


# ── Test problems ────────────────────────────────────────

def sod_shock_tube(N=200):
    x=np.linspace(0.,1.,N); rho=np.where(x<.5,1.,.125); ux=np.zeros(N)
    uy=np.zeros(N); p=np.where(x<.5,1.,.1); return x,rho,ux,uy,p,0.2

def lax_problem(N=200):
    x=np.linspace(0.,1.,N); rho=np.where(x<.5,.445,.5); ux=np.where(x<.5,.698,0.)
    uy=np.zeros(N); p=np.where(x<.5,3.528,.571); return x,rho,ux,uy,p,0.14

def shu_osher(N=400):
    x=np.linspace(-5.,5.,N); rho=np.where(x<-4.,3.857143,1.+0.2*np.sin(5.*x))
    ux=np.where(x<-4.,2.629369,0.); uy=np.zeros(N); p=np.where(x<-4.,10.3333,1.)
    return x,rho,ux,uy,p,1.8

def two_blast_waves(N=400):
    x=np.linspace(0.,1.,N); rho=np.ones(N); ux=np.zeros(N); uy=np.zeros(N)
    p=np.where(x<.1,1000.,np.where(x>.9,100.,.01)); return x,rho,ux,uy,p,0.038

def particle_drag_single(N=100, vgx=0., vgy=0., mp=1., vpx=1., vpy=0., tstop=1.):
    x=np.linspace(0.,100.,N); rho=np.ones(N); ux=np.full(N,vgx); uy=np.full(N,vgy)
    p=np.ones(N)*1e-4
    particles=[Particle_type(np.array([10.]),np.array([vpx]),np.array([vpy]),tstop,mp)]
    return x,rho,ux,uy,p,5.,particles

def nsh_equilibrium(N=128, epsilon=1., St=.1, eta=.05, Omega=1., q=1.5,
                    N_particles=None, L=1., p0=1.):
    if N_particles is None: N_particles=N
    Delta_v=eta*Omega; t_stop=St/Omega; D=(1.+epsilon)**2+St**2
    u_gx=+2.*epsilon*St*Delta_v/D; u_gy=-(1.+epsilon+St**2)*Delta_v/D
    v_px=-2.*St*Delta_v/D;         v_py=-(1.+epsilon)*Delta_v/D
    x=np.linspace(0.,L,N,endpoint=False); rho=np.ones(N)
    ux=np.full(N,u_gx); uy=np.full(N,u_gy); p=np.full(N,p0)
    x_p=np.linspace(0.,L,N_particles,endpoint=False)+0.5*L/N_particles
    m_p=epsilon*L/N_particles
    particles=[Particle_type(x_p,np.full(N_particles,v_px),np.full(N_particles,v_py),
                             tstop=t_stop,mass=m_p)]
    return x,rho,ux,uy,p,10.*(2.*np.pi/Omega),particles,ShearBox(Omega=Omega,q=q)

def clumpy_particle_drift(N=128, N_particles=512, clump_fraction=.1,
                           epsilon=1., tstop=.1, vp=1., vg=0.):
    L=1.; x=np.linspace(0.,L,N,endpoint=False)
    rho=np.ones(N); ux=np.full(N,vg); uy=np.zeros(N); p=np.ones(N)*1e-2
    cc=.25*L; ch=.5*clump_fraction*L
    x_p=np.linspace(cc-ch,cc+ch,N_particles,endpoint=False)+.5*(2*ch)/N_particles
    particles=[Particle_type(x_p,np.full(N_particles,vp),np.zeros(N_particles),
                             tstop=tstop,mass=epsilon*L/N_particles)]
    return x,rho,ux,uy,p,5.*tstop,particles


# ── Exact Sod ────────────────────────────────────────────

def sod_exact(x, t, gamma=1.4):
    from scipy.optimize import brentq
    rho_L,u_L,p_L=1.,0.,1.; rho_R,u_R,p_R=.125,0.,.1; x_0=.5
    cL=np.sqrt(gamma*p_L/rho_L); cR=np.sqrt(gamma*p_R/rho_R)
    gm1=gamma-1.; gp1=gamma+1.
    def f(p,pk,rk,ck):
        if p<=pk: return 2.*ck/gm1*((p/pk)**(gm1/(2.*gamma))-1.)
        else:
            A=2./(gp1*rk); B=gm1/gp1*pk
            return (p-pk)*np.sqrt(A/(p+B))
    p_star=brentq(lambda p: f(p,p_L,rho_L,cL)+f(p,p_R,rho_R,cR)+(u_R-u_L),
                  1e-10,10*max(p_L,p_R),xtol=1e-12,rtol=1e-12)
    u_star=u_L-f(p_star,p_L,rho_L,cL)
    if p_star<=p_L:
        rstar_L=rho_L*(p_star/p_L)**(1./gamma)
        cstar_L=cL*(p_star/p_L)**(gm1/(2.*gamma))
    else:
        rstar_L=rho_L*(p_star/p_L+gm1/gp1)/(gm1/gp1*p_star/p_L+1.)
        cstar_L=None
    rstar_R=(rho_R*(p_star/p_R+gm1/gp1)/(gm1/gp1*p_star/p_R+1.) if p_star>p_R
             else rho_R*(p_star/p_R)**(1./gamma))
    S_head=u_L-cL; S_tail=u_star-cstar_L
    S_shock=u_R+cR*np.sqrt(gp1/(2.*gamma)*p_star/p_R+gm1/(2.*gamma))
    xi=(x-x_0)/t; ro=np.empty_like(x); uo=np.empty_like(x); po=np.empty_like(x)
    for i,xi_i in enumerate(xi):
        if   xi_i<=S_head:    ro[i],uo[i],po[i]=rho_L,u_L,p_L
        elif xi_i<=S_tail:
            uf=2./gp1*(cL+gm1/2.*u_L+xi_i); cf=cL-gm1/2.*(uf-u_L)
            rf=rho_L*(cf/cL)**(2./gm1); pf=p_L*(rf/rho_L)**gamma
            ro[i],uo[i],po[i]=rf,uf,pf
        elif xi_i<=u_star:    ro[i],uo[i],po[i]=rstar_L,u_star,p_star
        elif xi_i<=S_shock:   ro[i],uo[i],po[i]=rstar_R,u_star,p_star
        else:                  ro[i],uo[i],po[i]=rho_R,u_R,p_R
    return ro,uo,np.zeros_like(ro),po


# ── Animation ────────────────────────────────────────────

def animate_euler(rho_init, ux_init, p_init, x, t_end,
                  uy_init=None, n_frames=100, variables="primitive",
                  cfl=0.9, bc="transmissive", ssprk="ssprk3",
                  figsize=(10,9), exact_fn=None, interval=40, recons='WENOZpI'):
    cfl_eff={"ssprk1":1.,"ssprk2":1.,"ssprk3":1.,"ssprk4":2.,"ssprk5":1.508}.get(ssprk.lower(),1.)
    x=np.asarray(x,dtype=float); rho_init=np.asarray(rho_init,dtype=float)
    ux_init=np.asarray(ux_init,dtype=float); p_init=np.asarray(p_init,dtype=float)
    uy_init=np.zeros_like(rho_init) if uy_init is None else np.asarray(uy_init,dtype=float)
    dx=x[1]-x[0]
    def pad(a): return np.concatenate([[a[0],a[0]],a,[a[-1],a[-1]]])
    if variables=="primitive":
        labels=[r"$\rho$",r"$u_x$",r"$u_y$",r"$p$"]
        colours=["#2171b5","#238b45","#fd8d3c","#cb181d"]
        def extract(U_): return cons_to_prim(U_[:,2:-2])
    elif variables=="conserved":
        labels=[r"$\rho$",r"$\rho u_x$",r"$\rho u_y$",r"$E$"]
        colours=["#2171b5","#238b45","#fd8d3c","#cb181d"]
        def extract(U_): return (U_[0,2:-2].copy(),U_[1,2:-2].copy(),U_[2,2:-2].copy(),U_[3,2:-2].copy())
    else: raise ValueError("variables must be 'primitive' or 'conserved'")
    q_init=extract(prim_to_cons(pad(rho_init),pad(ux_init),pad(uy_init),pad(p_init)))
    rho_f,ux_f,uy_f,p_f,_=solve(rho_init,ux_init,p_init,x,t_end,
                                  uy_init=uy_init,cfl=cfl,bc=bc,ssprk=ssprk,
                                  verbose=False,recons=recons)
    q_end=(rho_f,ux_f,uy_f,p_f) if variables=="primitive" else \
          extract(prim_to_cons(pad(rho_f),pad(ux_f),pad(uy_f),pad(p_f)))
    def _lims(a,b,m=.08):
        lo=min(a.min(),b.min()); hi=max(a.max(),b.max()); s=hi-lo if hi>lo else 1.
        return lo-m*s, hi+m*s
    ylims=[_lims(qi,qf) for qi,qf in zip(q_init,q_end)]
    frame_times=np.linspace(0.,t_end,n_frames+1)
    U=prim_to_cons(pad(rho_init),pad(ux_init),pad(uy_init),pad(p_init)); t=0.
    fig=plt.figure(figsize=figsize,facecolor="white")
    gs=gridspec.GridSpec(4,1,hspace=.5,left=.1,right=.95,top=.93,bottom=.06)
    axes=[fig.add_subplot(gs[i]) for i in range(4)]
    lines=[]; exact_lines=[]
    for ax,q,label,col,ylim in zip(axes,q_init,labels,colours,ylims):
        ln,=ax.plot(x,q,color=col,lw=1.5); lines.append(ln)
        if exact_fn is not None:
            ex,=ax.plot(x,q,color=".65",lw=1.,ls="--",zorder=0,label="exact")
            exact_lines.append(ex)
        ax.set_ylabel(label,fontsize=9); ax.set_xlim(x[0],x[-1]); ax.set_ylim(*ylim)
        ax.tick_params(labelsize=8); ax.grid(True,lw=.4,alpha=.4)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    if exact_fn is not None: axes[0].legend(fontsize=8,loc="upper right",framealpha=.6,edgecolor="none")
    title=fig.suptitle("t = 0.0000",fontsize=10,x=.52)
    def update(frame_idx):
        nonlocal U,t
        t_target=frame_times[frame_idx]
        while t<t_target:
            dt_s=compute_dt(U[:,2:-2],dx,cfl*cfl_eff); dt_s=min(dt_s,t_target-t)
            U=ssprk_step(U,dx,dt_s,bc,ssprk,recons=recons); t+=dt_s
        q1,q2,q3,q4=extract(U)
        for ln,q in zip(lines,(q1,q2,q3,q4)): ln.set_ydata(q)
        if exact_fn is not None:
            er,eu,euy,ep=exact_fn(x,max(t,1e-12))
            exact_qs=(er,eu,euy,ep) if variables=="primitive" else \
                     (er,er*eu,er*euy,ep/(GAMMA-1.)+.5*er*(eu**2+euy**2))
            for ex_ln,eq in zip(exact_lines,exact_qs): ex_ln.set_ydata(eq)
        title.set_text(f"t = {t:.4f}  (frame {frame_idx} / {n_frames})")
        return lines+exact_lines+[title]
    anim=FuncAnimation(fig,update,frames=n_frames+1,interval=interval,blit=True)
    plt.close(fig); return anim,HTML(anim.to_jshtml())


if __name__ == "__main__":
    import sys
    problem_map={"sod":sod_shock_tube,"lax":lax_problem,"shu":shu_osher,"blast":two_blast_waves}
    problem_name=sys.argv[1] if len(sys.argv)>1 else "sod"
    ssprk_name  =sys.argv[2] if len(sys.argv)>2 else "ssprk3"
    if problem_name not in problem_map:
        print(f"Unknown problem {problem_name!r}. Choose: {list(problem_map)}"); sys.exit(1)
    x,rho0,ux0,uy0,p0,t_end=problem_map[problem_name]()
    print(f"Running {problem_name} with {ssprk_name} …")
    rho,ux,uy,p,t=solve(rho0,ux0,p0,x,t_end,uy_init=uy0,ssprk=ssprk_name,cfl=.9,verbose=True)
    print(f"Done.  t_final = {t:.6f}")
    exact_fn=sod_exact if problem_name=="sod" else None
    fig,axes=plt.subplots(4,1,figsize=(9,10),sharex=True); fig.subplots_adjust(hspace=.4)
    fields=[(rho,r"$\rho$"),(ux,r"$u_x$"),(uy,r"$u_y$"),(p,r"$p$")]
    colours=["#2171b5","#238b45","#fd8d3c","#cb181d"]
    for ax,(q,lbl),col in zip(axes,fields,colours):
        ax.plot(x,q,color=col,lw=1.5,label="numerical"); ax.set_ylabel(lbl,fontsize=10)
        ax.grid(True,lw=.4,alpha=.4)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    if exact_fn is not None:
        erho,eux,euy,ep=exact_fn(x,t)
        for ax,eq in zip(axes,[erho,eux,euy,ep]):
            ax.plot(x,eq,"k--",lw=1.,alpha=.6,label="exact")
        axes[0].legend(fontsize=9)
    axes[-1].set_xlabel("x",fontsize=10)
    fig.suptitle(f"{problem_name}  |  {ssprk_name}  |  t = {t:.4f}",fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{problem_name}_{ssprk_name}.png",dpi=150)
    print(f"Figure saved to {problem_name}_{ssprk_name}.png")
