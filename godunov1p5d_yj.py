"""
1.5D Godunov Finite Volume Fluid–Particle Solver
=================================================
Solves the 1.5-D Euler equations coupled to Lagrangian dust particles
via aerodynamic (Epstein/Stokes) drag.  The fluid is advanced with a
Godunov finite-volume scheme using an HLLC Riemann solver.  Gas-particle
coupling uses the Yang & Johansen (2016) Strang-split drag operator with
per-cell analytical solutions and particle-mesh back reaction (PMBR).

Grid & Equations
----------------
The grid is 1-D (scalar x positions) but the velocity field is a
2-component vector (uₓ, uᵧ).  uᵧ is advected passively by uₓ with
no y-pressure gradient ("1.5-D" or "2-velocity" model).

Governing equations (conservative form):
  ∂U/∂t + ∂F/∂x = S_gas

  U = [ρ, ρuₓ, ρuᵧ, E]ᵀ                      (conserved variables)
  F = [ρuₓ, ρuₓ²+p, ρuₓuᵧ, uₓ(E+p)]ᵀ        (fluxes)
  E = p/(γ-1) + ½ρ(uₓ²+uᵧ²)                  (total energy, ideal gas)

Equation of state:
  Adiabatic:   p = (γ-1)(E - ½ρ(uₓ²+uᵧ²))
  Isothermal:  p = ρ·cs²   (controlled by cs_iso parameter)

Operator Splitting
------------------
The drag and (optionally) shearing-box forces are handled by a
Verlet-ordered Strang splitting.  Each full timestep has the structure:

  P(dt/2)   — x += v^n * dt/2        (drift with pre-drag velocity)
  D(dt/2)   — per-cell analytical drag [+ Coriolis if shear_box set],
              velocities and gas momenta updated, positions unchanged
  L(dt)     — pure SSPRK2 hyperbolic advance, no drag source terms,
              particle positions fixed at x^{n+1/2}
  D(dt/2)   — second half-drag, positions still at x^{n+1/2}
  P(dt/2)   — x += v^{n+1} * dt/2    (drift with post-drag velocity)
              + boundary wrapping at t^{n+1}

D(h) is a pure velocity operator.  The commutator [D, P] is O(h²)
because D changes v by O(h), shifting P·x by O(h²) — exactly the
condition required for Strang splitting to give second-order convergence.

Previously positions were updated inside D(h), making D a compound
operator.  D(h/2)∘D(h/2) ≠ D(h) because PMBR in the first call
modified the gas state before the second call, giving O(dt) error
per step and first-order convergence overall.

Particle-mesh back reaction (PMBR, Yang & Johansen §2.3)
---------------------------------------------------------
After the per-cell solve, total particle velocity changes are scattered
back to the gas grid via TSC.  This corrects for the inter-cell gas
coupling that the local per-cell solve cannot see, and is essential
for correct convergence rates (see Yang & Johansen §4.2.1).

For the shearing box case the gas velocity update is computed directly
from the per-cell center-of-mass and relative-mode decomposition,
not via PMBR alone, because the gas also feels Coriolis and epicyclic
forces during D(h) that PMBR does not capture.

References
----------
  Yang & Johansen (2016) ApJS 224:39  — per-cell analytical drag + PMBR
  Gottlieb, Shu & Tadmor (2001)       — SSPRK methods
  Borges et al. (2008)                — WENO-Z reconstruction
  Luo & Wu (2021)                     — WENO-Z+I reconstruction
  Toro (2009)                         — HLLC Riemann solver
  Fung & Muley (2019) ApJS 244:42     — analytic position update
  Nakagawa, Sekiya & Hayashi (1986)   — NSH equilibrium
"""

from collections import namedtuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
try:
    from IPython.display import HTML
except ImportError:
    HTML = None

# ─────────────────────────────────────────────────────────
#  Global constants
# ─────────────────────────────────────────────────────────

GAMMA = 1.4   # ratio of specific heats (ideal diatomic gas)

ShearBox = namedtuple("ShearBox", ["Omega", "q", "eta"])


# ─────────────────────────────────────────────────────────
#  Particle container
# ─────────────────────────────────────────────────────────

class Particle_type:
    """
    A single species of Lagrangian particles sharing the same stopping time
    and mass.

    Attributes
    ----------
    xs   : ndarray (N_p,)   radial positions
    vxs  : ndarray (N_p,)   radial   (x) velocities
    vys  : ndarray (N_p,)   azimuthal (y) velocities
    tstop: float            aerodynamic stopping time
    mass : float            mass per particle
    """

    def __init__(self, xs, vxs, vys, tstop=1., mass=1.):
        self.np    = len(xs)
        self.xs    = np.asarray(xs,   dtype=float)
        self.vxs   = np.asarray(vxs,  dtype=float)
        self.vys   = np.asarray(vys,  dtype=float)
        self.tstop = tstop
        self.mass  = mass

    def tot_M(self):
        """Total x-momentum of the particle species."""
        return np.sum(self.vxs * self.mass)


# ─────────────────────────────────────────────────────────
#  Equation-of-state helpers  (1.5-D)
# ─────────────────────────────────────────────────────────

def prim_to_cons(rho, ux, uy, p):
    """Convert primitive variables (ρ, uₓ, uᵧ, p) to conserved [ρ, ρuₓ, ρuᵧ, E]."""
    E = p / (GAMMA - 1.0) + 0.5 * rho * (ux**2 + uy**2)
    return np.array([rho, rho*ux, rho*uy, E])


def cons_to_prim(U):
    """Convert conserved vector U = [ρ, ρuₓ, ρuᵧ, E] to primitives (ρ, uₓ, uᵧ, p)."""
    rho = U[0]
    ux  = U[1] / rho
    uy  = U[2] / rho
    E   = U[3]
    p   = (GAMMA - 1.0) * (E - 0.5 * rho * (ux**2 + uy**2))
    p   = np.maximum(p,   1.0e-14)
    rho = np.maximum(rho, 1.0e-14)
    return rho, ux, uy, p


def sound_speed(rho, p):
    """Ideal-gas sound speed c = sqrt(γp/ρ)."""
    return np.sqrt(GAMMA * np.maximum(p, 1.0e-14) / np.maximum(rho, 1.0e-14))


def enforce_isothermal(U, cs_iso):
    """Enforce p = ρ·cs² by recomputing the energy row in place."""
    rho  = np.maximum(U[0], 1.0e-14)
    p    = rho * cs_iso**2 / GAMMA
    U[3] = p / (GAMMA - 1.0) + 0.5 * (U[1]**2 + U[2]**2) / rho
    return U


def euler_flux(rho, ux, uy, p):
    """Physical flux F(U) in x for the 1.5-D Euler equations."""
    E = p / (GAMMA - 1.0) + 0.5 * rho * (ux**2 + uy**2)
    return np.array([
        rho * ux,
        rho * ux**2 + p,
        rho * ux * uy,
        ux  * (E + p),
    ])


# ─────────────────────────────────────────────────────────
#  Riemann solver  (HLLC, 1.5-D)
# ─────────────────────────────────────────────────────────

def riemann_hllc(UL, UR):
    """HLLC approximate Riemann solver for the 1.5-D Euler equations."""
    rhoL, uxL, uyL, pL = cons_to_prim(UL)
    rhoR, uxR, uyR, pR = cons_to_prim(UR)

    cL = sound_speed(rhoL, pL)
    cR = sound_speed(rhoR, pR)

    rho_bar = 0.5 * (rhoL + rhoR)
    c_bar   = 0.5 * (cL   + cR  )
    p_star  = np.maximum(
        0.0, 0.5*(pL + pR) - 0.5*(uxR - uxL)*rho_bar*c_bar)

    qL = np.where(p_star > pL,
                  np.sqrt(1.0 + (GAMMA+1)/(2*GAMMA)*(p_star/pL - 1)), 1.0)
    qR = np.where(p_star > pR,
                  np.sqrt(1.0 + (GAMMA+1)/(2*GAMMA)*(p_star/pR - 1)), 1.0)

    SL = uxL - cL*qL
    SR = uxR + cR*qR

    denom  = rhoL*(SL - uxL) - rhoR*(SR - uxR)
    S_star = (pR - pL
              + rhoL*uxL*(SL - uxL)
              - rhoR*uxR*(SR - uxR)) / denom

    FL = euler_flux(rhoL, uxL, uyL, pL)
    FR = euler_flux(rhoR, uxR, uyR, pR)

    def star_state(U, rho, ux, uy, p, S):
        factor = rho * (S - ux) / (S - S_star)
        E_spec = U[3] / rho
        return np.array([
            factor,
            factor * S_star,
            factor * uy,
            factor * (E_spec + (S_star - ux)
                      * (S_star + p / (rho * (S - ux)))),
        ])

    U_starL = star_state(UL, rhoL, uxL, uyL, pL, SL)
    U_starR = star_state(UR, rhoR, uxR, uyR, pR, SR)

    return np.where(
        SL >= 0.0, FL,
        np.where(S_star >= 0.0, FL + SL*(U_starL - UL),
        np.where(SR     >= 0.0, FR + SR*(U_starR - UR),
                                FR)))


# ─────────────────────────────────────────────────────────
#  WENO reconstruction
# ─────────────────────────────────────────────────────────

def _weno_z_weights(vm2, vm1, v0, vp1, vp2):
    """WENO-Z nonlinear weights (Borges et al. 2008)."""
    eps = 1.0e-36
    q0 = ( 2.0*vm2 -  7.0*vm1 + 11.0*v0 ) / 6.0
    q1 = (         -  1.0*vm1 +  5.0*v0  +  2.0*vp1) / 6.0
    q2 = ( 2.0*v0  +  5.0*vp1 -  1.0*vp2) / 6.0
    b0 = (13./12.)*(vm2 - 2.*vm1 + v0 )**2 + (1./4.)*(vm2 - 4.*vm1 + 3.*v0)**2
    b1 = (13./12.)*(vm1 - 2.*v0  + vp1)**2 + (1./4.)*(vm1 - vp1)**2
    b2 = (13./12.)*(v0  - 2.*vp1 + vp2)**2 + (1./4.)*(3.*v0 - 4.*vp1 + vp2)**2
    tau5  = np.abs(b0 - b2)
    d0, d1, d2 = 1./10., 3./5., 3./10.
    a0 = d0 * (1. + tau5/(eps+b0))
    a1 = d1 * (1. + tau5/(eps+b1))
    a2 = d2 * (1. + tau5/(eps+b2))
    a_sum = a0 + a1 + a2
    return (a0*q0 + a1*q1 + a2*q2) / a_sum


def _weno_zpi_weights(vm2, vm1, v0, vp1, vp2):
    """WENO-Z+I nonlinear weights (Luo & Wu 2021)."""
    eps = 1.0e-40
    lam = 3.3
    q0 = ( 2.0*vm2 -  7.0*vm1 + 11.0*v0 ) / 6.0
    q1 = (         -  1.0*vm1 +  5.0*v0  +  2.0*vp1) / 6.0
    q2 = ( 2.0*v0  +  5.0*vp1 -  1.0*vp2) / 6.0
    b0 = (13./12.)*(vm2 - 2.*vm1 + v0 )**2 + (1./4.)*(vm2 - 4.*vm1 + 3.*v0)**2
    b1 = (13./12.)*(vm1 - 2.*v0  + vp1)**2 + (1./4.)*(vm1 - vp1)**2
    b2 = (13./12.)*(v0  - 2.*vp1 + vp2)**2 + (1./4.)*(3.*v0 - 4.*vp1 + vp2)**2
    tau5    = np.abs(b0 - b2)
    b_max   = np.maximum(np.maximum(b0, b1), b2)
    b_min   = np.minimum(np.minimum(b0, b1), b2)
    alpha_a = 1.0 - b_min / (b_max + eps)
    d0, d1, d2 = 1./10., 3./5., 3./10.
    extra = lam * alpha_a / (b_max + eps)
    a0 = d0 * (1. + (tau5/(eps+b0))**2 + extra*b0)
    a1 = d1 * (1. + (tau5/(eps+b1))**2 + extra*b1)
    a2 = d2 * (1. + (tau5/(eps+b2))**2 + extra*b2)
    a_sum = a0 + a1 + a2
    return (a0*q0 + a1*q1 + a2*q2) / a_sum


def reconstruct_weno_z(U, weight_fn):
    """Compute left/right interface states at every face using WENO."""
    n       = U.shape[1]
    n_faces = n - 1
    UL = np.empty((U.shape[0], n_faces))
    UR = np.empty((U.shape[0], n_faces))
    for k in range(2, n_faces - 2):
        UL[:, k] = weight_fn(
            U[:, k-2], U[:, k-1], U[:, k], U[:, k+1], U[:, k+2])
        UR[:, k] = weight_fn(
            U[:, k+3], U[:, k+2], U[:, k+1], U[:, k], U[:, k-1])
    for k in [0, 1, n_faces-2, n_faces-1]:
        UL[:, k] = U[:, k]
        UR[:, k] = U[:, k+1]
    return UL, UR


# ─────────────────────────────────────────────────────────
#  Boundary conditions
# ─────────────────────────────────────────────────────────

def apply_bc(U, bc="transmissive"):
    """Apply BCs using two ghost cells on each side."""
    if bc == "transmissive":
        U[:, 0]  = U[:, 2];  U[:, 1]  = U[:, 2]
        U[:, -1] = U[:, -3]; U[:, -2] = U[:, -3]
    elif bc == "reflective":
        U[:, 0]  = U[:, 2];  U[1, 0]  = -U[1, 2]
        U[:, 1]  = U[:, 2];  U[1, 1]  = -U[1, 2]
        U[:, -1] = U[:, -3]; U[1, -1] = -U[1, -3]
        U[:, -2] = U[:, -3]; U[1, -2] = -U[1, -3]
    elif bc in ("periodic", "shear_periodic"):
        U[:, 0]  = U[:, -4]
        U[:, 1]  = U[:, -3]
        U[:, -2] = U[:, 2]
        U[:, -1] = U[:, 3]
    else:
        raise ValueError(f"Unknown BC type: {bc!r}. "
                         "Choose: transmissive | reflective | periodic | shear_periodic")
    return U

def apply_bc_gen(U, bc="transmissive"):
    """Apply BCs using two ghost cells on each side."""
    if bc == "transmissive":
        U[ 0]  = U[ 2];  U[ 1]  = U[ 2]
        U[ -1] = U[ -3]; U[ -2] = U[ -3]
    elif bc == "reflective":
        U[ 0]  = U[ 2];  U[ 0]  = -U[ 2]
        U[ 1]  = U[ 2];  U[ 1]  = -U[ 2]
        U[ -1] = U[ -3]; U[ -1] = -U[ -3]
        U[ -2] = U[ -3]; U[ -2] = -U[ -3]
    elif bc in ("periodic", "shear_periodic"):
        U[ 0]  = U[ -4]
        U[ 1]  = U[ -3]
        U[ -2] = U[ 2]
        U[ -1] = U[ 3]
    else:
        raise ValueError(f"Unknown BC type: {bc!r}. "
                         "Choose: transmissive | reflective | periodic | shear_periodic")
    return U

# ─────────────────────────────────────────────────────────
#  Spatial residual  L(U) = -(1/dx)(F_{i+½} - F_{i-½})
# ─────────────────────────────────────────────────────────

def spatial_residual(U, dx, bc, recons='WENOZpI',
                     gas_sources=[], gas_source_params=[]):
    """
    Compute semi-discrete RHS L(U) using HLLC fluxes plus source terms.

    NOTE: When using the Yang-Johansen Strang splitting, drag and
    shearing-box forces are handled inside yang_johansen_drag_step.
    Do NOT pass gas_shearing_box_source here in that case — it would
    be double-counted.
    """
    U = apply_bc(U, bc)
    n_cells = U.shape[1]
    F = np.zeros_like(U)

    if recons in ('WENOZ', 'WENOZpI'):
        wfn = _weno_z_weights if recons == 'WENOZ' else _weno_zpi_weights
        UL_all, UR_all = reconstruct_weno_z(U, wfn)

    for i in range(1, n_cells - 1):
        match recons:
            case 'godunov':
                F[:, i] = riemann_hllc(U[:, i], U[:, i+1])
            case 'WENOZ' | 'WENOZpI':
                F[:, i] = riemann_hllc(UL_all[:, i], UR_all[:, i])
            case _:
                raise ValueError(
                    f"Unknown reconstruction {recons!r}. "
                    "Choose: godunov | WENOZ | WENOZpI")

    L = np.zeros_like(U)
    L[:, 2:-2] = -(1.0/dx) * (F[:, 2:-2] - F[:, 1:-3])

    for source, params in zip(gas_sources, gas_source_params):
        L += source(U, params)

    return L


# ─────────────────────────────────────────────────────────
#  CFL time-step
# ─────────────────────────────────────────────────────────

def compute_dt(U, dx, cfl=0.9, particles=None):
    """
    Compute maximum stable time step via the CFL condition.

    The Yang-Johansen drag operator is unconditionally stable in dt/τ_s
    so no drag-time limiter is needed — CFL on fluid signal speed and
    particle advection velocity is sufficient.
    """
    rho, ux, uy, p = cons_to_prim(U)
    c     = sound_speed(rho, p)
    S_max = np.max(np.abs(ux) + c)
    dt    = cfl * dx / S_max if S_max > 1.0e-14 else 1.0e-6

    if particles is not None:
        for p_type in particles:
            if p_type.np == 0:
                continue
            vx_max = np.max(np.abs(p_type.vxs))
            if vx_max > 1.0e-14:
                dt = min(dt, cfl * dx / vx_max)

    return dt


# ─────────────────────────────────────────────────────────
#  SSPRK tableaux  (Shu-Osher form)
# ─────────────────────────────────────────────────────────

def _ssprk_tableau(name):
    """Return (alpha, beta) for the requested SSPRK method."""
    name = name.lower()

    if name == "ssprk1":
        s = 1
        alpha = np.zeros((s+1, s)); beta = np.zeros((s+1, s))
        alpha[1,0] = 1.0; beta[1,0] = 1.0

    elif name == "ssprk2":
        s = 2
        alpha = np.zeros((s+1, s)); beta = np.zeros((s+1, s))
        alpha[1,0] = 1.0;               beta[1,0] = 1.0
        alpha[2,0] = 0.5; alpha[2,1] = 0.5
        beta [2,1] = 0.5

    elif name == "ssprk3":
        s = 3
        alpha = np.zeros((s+1, s)); beta = np.zeros((s+1, s))
        alpha[1,0] = 1.0;                            beta[1,0] = 1.0
        alpha[2,0] = 3/4; alpha[2,1] = 1/4;          beta[2,1] = 1/4
        alpha[3,0] = 1/3; alpha[3,2] = 2/3;          beta[3,2] = 2/3

    elif name == "ssprk4":
        s = 4
        alpha = np.zeros((s+1, s)); beta = np.zeros((s+1, s))
        alpha[1,0] = 1.0;                             beta[1,0] = 1/2
        alpha[2,1] = 1.0;                             beta[2,1] = 1/2
        alpha[3,0] = 2/3; alpha[3,2] = 1/3;          beta[3,2] = 1/6
        alpha[4,3] = 1.0;                             beta[4,3] = 1/2

    elif name == "ssprk5":
        s = 5
        alpha = np.zeros((s+1, s)); beta = np.zeros((s+1, s))
        alpha[1,0] = 1.0
        beta [1,0] = 0.39175222700392
        alpha[2,0] = 0.44437049406734; alpha[2,1] = 0.55562950593266
        beta [2,1] = 0.36841059262959
        alpha[3,0] = 0.62010185138540; alpha[3,2] = 0.37989814861460
        beta [3,2] = 0.25189177424738
        alpha[4,0] = 0.17807995410773; alpha[4,3] = 0.82192004589227
        beta [4,3] = 0.54497475021237
        alpha[5,0] = 0.00683325884039; alpha[5,2] = 0.51723167208978
        alpha[5,4] = 0.47591583767062 - 1e-15
        beta [5,2] = 0.12759831133288; beta[5,4]  = 0.08460416338212

    else:
        raise ValueError(
            f"Unknown SSPRK method {name!r}. "
            "Choose: ssprk1 | ssprk2 | ssprk3 | ssprk4 | ssprk5")

    return alpha, beta


# ─────────────────────────────────────────────────────────
#  Triangular Shaped Cloud (TSC) particle ↔ grid mapping
# ─────────────────────────────────────────────────────────

def _tsc_weights(x_p, x0, dx):
    """Compute TSC (quadratic B-spline) weights."""
    x_p    = np.asarray(x_p, dtype=float)
    scalar = x_p.ndim == 0
    x_p    = np.atleast_1d(x_p)
    eta    = (x_p - x0) / dx
    i_c    = np.round(eta).astype(int)
    delta  = eta - i_c
    w      = np.empty((x_p.size, 3), dtype=float)
    w[:, 0] = 0.5 * (0.5 - delta)**2
    w[:, 1] = 0.75 - delta**2
    w[:, 2] = 0.5 * (0.5 + delta)**2
    if scalar:
        return i_c[0], w[0]
    return i_c, w


def grid_to_particles(q_grid, x_grid, x_particles):
    """Gather a grid quantity to particle locations using TSC."""
    q_grid      = np.asarray(q_grid,      dtype=float)
    x_grid      = np.asarray(x_grid,      dtype=float)
    x_particles = np.asarray(x_particles, dtype=float)
    N  = q_grid.size
    dx = x_grid[1] - x_grid[0]
    x0 = x_grid[0]
    i_c, w = _tsc_weights(x_particles, x0, dx)
    idx = np.stack([
        np.clip(i_c-1, 0, N-1),
        np.clip(i_c,   0, N-1),
        np.clip(i_c+1, 0, N-1),
    ], axis=1)
    return np.einsum("pj,pj->p", w, q_grid[idx])


def particles_to_grid(q_particles, weights_p, x_particles, x_grid, bc='periodic'):
    q_particles  = np.asarray(q_particles,  dtype=float)
    x_particles  = np.asarray(x_particles,  dtype=float)
    x_grid       = np.asarray(x_grid,       dtype=float)
    N_p = q_particles.size
    N   = x_grid.size
    dx  = x_grid[1] - x_grid[0]
    x0  = x_grid[0]
    weights_p = (np.ones(N_p, dtype=float)
                 if weights_p is None
                 else np.asarray(weights_p, dtype=float))
    i_c, w    = _tsc_weights(x_particles, x0, dx)
    eff_w     = w * weights_p[:, np.newaxis]
    q_grid    = np.zeros(N, dtype=float)
    for s in range(3):
        idx = i_c + s - 1
        if bc in ('periodic', 'shear_periodic'):
            idx = idx % N
        else:
            idx = np.clip(idx, 0, N-1)
        np.add.at(q_grid, idx, eff_w[:, s] * q_particles)
    return q_grid


def _fold_ghost_deposits(deposit, Nx, bc):
    """Fold ghost-cell deposits back into physical cells for periodic BCs."""
    if bc in ('periodic', 'shear_periodic'):
        deposit[Nx  ] += deposit[0];  deposit[0]    = 0.0
        deposit[Nx+1] += deposit[1];  deposit[1]    = 0.0
        deposit[2]    += deposit[Nx+2]; deposit[Nx+2] = 0.0
        deposit[3]    += deposit[Nx+3]; deposit[Nx+3] = 0.0
    return deposit


# ─────────────────────────────────────────────────────────
#  Shearing-box gas source term
#  (only used in L(dt) if NOT using Yang-Johansen splitting,
#   since YJ handles Coriolis+epicyclic inside yang_johansen_drag_step)
# ─────────────────────────────────────────────────────────

def gas_shearing_box_source(U, params):
    """
    Coriolis + epicyclic + pressure-gradient source terms for the gas.

    WARNING: Do not pass this to spatial_residual when using the
    Yang-Johansen Strang splitting — those forces are already applied
    analytically inside yang_johansen_drag_step.  Passing them here
    as well would double-count them.
    """
    sb, eta = params
    rho = U[0, 2:-2]
    ux  = U[1, 2:-2] / rho
    uy  = U[2, 2:-2] / rho
    S = np.zeros_like(U)
    S[1, 2:-2] = rho * (2.0 * sb.Omega * uy  +  2.0 * eta * sb.Omega**2)
    S[2, 2:-2] = rho * (-(2.0 - sb.q) * sb.Omega * ux)
    return S


# ─────────────────────────────────────────────────────────
#  Particle position drift  P(h)
# ─────────────────────────────────────────────────────────

def _particle_drift(particles, h, bc, x_grid, shear_box=None):
    """
    Pure position drift operator P(h): x += v * h.

    This is the position half of the Verlet / leapfrog splitting.
    It uses whatever velocity is currently stored in each particle —
    the caller is responsible for ensuring velocities are at the
    correct time level before calling.

    Boundary handling is applied only when apply_bc=True (i.e. on
    the FINAL drift of the step so that t^{n+1} positions are wrapped).
    Wrapping mid-step would corrupt the TSC deposit in the next D(h)
    call for particles near the domain boundary.

    The shear_periodic azimuthal velocity kick (Δvᵧ = ±q·Ω·Lₓ) is
    also applied here when a particle crosses the radial boundary,
    and only on the final drift.

    Parameters
    ----------
    particles : list of Particle_type   velocities must be current
    h         : float                   drift duration
    bc        : str                     boundary condition type
    x_grid    : ndarray (Nx,)           physical cell centres
    shear_box : ShearBox or None        needed only for shear_periodic kick

    Returns
    -------
    particles : list of Particle_type with updated positions
    apply_bc  is handled internally based on the `apply_bc` parameter
    """
    dx         = x_grid[1] - x_grid[0]
    x_phys_min = x_grid[0] - dx/2
    x_phys_max = x_grid[-1] + dx/2
    L_domain   = x_phys_max - x_phys_min

    for p_type in particles:
        p_type.xs += p_type.vxs * h

    return particles


def _particle_drift_with_bc(particles, h, bc, x_grid, shear_box=None):
    """
    Position drift P(h) followed by boundary wrapping.

    Identical to _particle_drift but also applies periodic wrapping
    and the shear_periodic azimuthal velocity kick.  Call this only
    on the FINAL half-drift of the step, after all velocity updates
    are complete, so that the wrapped positions at t^{n+1} are
    deposited correctly on the next step.
    """
    dx         = x_grid[1] - x_grid[0]
    x_phys_min = x_grid[0] - dx/2
    x_phys_max = x_grid[-1] + dx/2
    L_domain   = x_phys_max - x_phys_min

    for p_type in particles:
        p_type.xs += p_type.vxs * h

        if bc == 'periodic':
            p_type.xs = (x_phys_min
                         + (p_type.xs - x_phys_min) % L_domain)
        elif bc == 'shear_periodic':
            # Detect crossings before wrapping
            crossed_right = p_type.xs >= x_phys_min + L_domain
            crossed_left  = p_type.xs <  x_phys_min
            p_type.xs = (x_phys_min
                         + (p_type.xs - x_phys_min) % L_domain)
            if shear_box is not None:
                # Azimuthal velocity kick when a particle crosses the
                # radial boundary (shearing-box boundary condition).
                delta_vy = shear_box.q * shear_box.Omega * L_domain
                p_type.vys[crossed_right] += delta_vy
                p_type.vys[crossed_left ] -= delta_vy

    return particles


# ─────────────────────────────────────────────────────────
#  Yang & Johansen (2016) drag operator — helpers + main
# ─────────────────────────────────────────────────────────
#
# Algorithm overview (Yang & Johansen §2.1–2.3):
#
# Step 1 — Per-cell analytical solve (§2.2):
#   Each particle is split into sub-clouds, one per overlapping TSC cell.
#   Inside each cell the gas + all its sub-clouds form a closed multi-fluid
#   system whose ODE is solved exactly.  After all cells are integrated,
#   particle velocity changes are collected back to parent particles via a
#   TSC-weighted sum (Eq. 27):
#       Δvⱼ = Σ_k  W(rₚⱼ − rₖ) · Δvⱼ^(k)
#
# Step 2 — Particle-mesh back reaction (PMBR, §2.3):
#   Having Δvⱼ for every particle, each particle is now treated as a
#   UNIFIED cloud (not split) carrying total momentum change mₚⱼ Δvⱼ.
#   This is scattered onto the grid by standard TSC assignment.
#   The gas velocity change per cell is then recovered from momentum
#   conservation in the centre-of-mass frame (Eq. 31):
#
#       ρg,k Δuₖ = −Σⱼ mₚⱼ · W(rₚⱼ − rₖ) · Δvⱼ
#
#   i.e. minus the TSC-scattered particle momentum change divided by dx.
#   This inter-cell coupling of the gas via PMBR is what allows the
#   algorithm to achieve convergence rates matching explicit integration
#   (Yang & Johansen §4.2.1, Figs 9–12).  Without it convergence requires
#   4–8× higher resolution for equivalent accuracy.
#
# Predictor-corrector for second-order accuracy:
#   A single evaluation of the per-cell ODE at t^n has O(h²) local error
#   because the gas velocity changes during h.  Using the midpoint gas
#   velocity (estimated by a predictor half-step) gives O(h³) local error
#   → O(h²) global convergence.  The predictor uses the cheap local gas
#   update (no PMBR) since it only needs to be a reasonable estimate.

def _yj_per_cell_solve(ugx_pad, ugy_pad, eps_grid, p_type, i_c, w, h,
                        shear_box, N_pad):
    """
    Per-cell analytical drag solve — pure function, no side effects.

    Loops over the three TSC stencil offsets s = -1, 0, +1.  For each
    offset, particle j interacts with padded cell k = i_c[j] + s with
    TSC weight w[j, s].  The per-cell ODE (Eqs. 7–8 or 13–16) is solved
    analytically for that (particle, cell) pair.

    Particle velocity changes are accumulated as a TSC-weighted sum over
    each particle's sub-clouds (Eq. 27):
        delta_vx[j] = Σ_s  w[j,s] * Δvx_jk(s)

    The local gas velocity changes (dug_x_grid, dug_y_grid) are a
    by-product computed from local momentum conservation and are used
    ONLY for the predictor half-step estimate.  The corrector's gas
    update is handled by PMBR in yang_johansen_drag_step (§2.3).

    Parameters
    ----------
    ugx_pad  : (N_pad,)   gas vx on padded grid
    ugy_pad  : (N_pad,)   gas vy on padded grid
    eps_grid : (N_pad,)   dust-to-gas ratio on padded grid
    p_type   : Particle_type   read-only particle state
    i_c      : (Np,)      TSC nearest-cell indices in padded grid
    w        : (Np,3)     TSC weights
    h        : float      sub-step duration
    shear_box: ShearBox or None
    N_pad    : int

    Returns
    -------
    delta_vx   : (Np,)     particle vx change (TSC-weighted over sub-clouds)
    delta_vy   : (Np,)     particle vy change
    dug_x_grid : (N_pad,)  local gas vx change (predictor use only)
    dug_y_grid : (N_pad,)  local gas vy change (predictor use only)
    cell_weight: (N_pad,)  total TSC weight deposited per cell
    """
    delta_vx    = np.zeros(p_type.np)
    delta_vy    = np.zeros(p_type.np)
    dug_x_grid  = np.zeros(N_pad)
    dug_y_grid  = np.zeros(N_pad)
    cell_weight = np.zeros(N_pad)

    for s in range(3):
        k_idx = np.clip(i_c + s - 1, 0, N_pad - 1)   # (Np,) cell index per particle

        ugx_k     = ugx_pad[k_idx]      # gas vx at cell k
        ugy_k     = ugy_pad[k_idx]      # gas vy at cell k
        eps_k     = eps_grid[k_idx]     # dust-to-gas ratio at cell k
        one_p_eps = 1.0 + eps_k

        # Coupled two-body decay: the relative velocity decays at rate
        # (1+ε)/τ_s, not 1/τ_s.  This is the stability mechanism of
        # Yang & Johansen — using the single-body rate causes instability
        # at high ε (the Bai & Stone chi-cap problem).
        coupled_decay = np.exp(-(1.0 + eps_k) * h / p_type.tstop)

        if shear_box is None:
            # ── Pure drag (eqs 7-8) ────────────────────────────────────
            # Velocity difference decays at the coupled rate; cm conserved.
            # Particle change (bounded):
            dvx_jk = (ugx_k - p_type.vxs) * (1.0 - coupled_decay) / one_p_eps
            dvy_jk = (ugy_k - p_type.vys) * (1.0 - coupled_decay) / one_p_eps

            # Gas change from eq. 31 (local, no scatter):
            #   Δu_k = -ε̃_{jk} Δv_{jk}  summed over sub-clouds in cell k.
            # Here ε̃_{jk} = eps_k * w[j,s] (the sub-cloud epsilon fraction).
            # Bounded: |Δu_k| ≤ eps_k/(1+eps_k) * |vp - ug| ≤ |vp - ug|.
            dug_x_jk = -eps_k * dvx_jk
            dug_y_jk = -eps_k * dvy_jk

        else:
            # ── Shearing box: Coriolis + epicyclic + drag (eqs 13-16) ──
            # Solution decomposes into three parts:
            #   1. NSH equilibrium (preserved exactly — no drift in equilibrium)
            #   2. CM deviation: pure epicyclic rotation, no decay
            #   3. Relative velocity deviation: decaying epicyclic
            sb   = shear_box
            St_k = p_type.tstop * sb.Omega
            D_k  = one_p_eps**2 + St_k**2

            # NSH equilibrium velocities (eqs 17-20, ax=0)
            Delta_v = sb.eta * sb.Omega          # headwind speed (= 0 for epicyclic test)
            ux_eq =  2.0 * eps_k * St_k / D_k * Delta_v
            uy_eq = -(one_p_eps  + St_k**2) / D_k * Delta_v
            vx_eq = -2.0 * St_k / D_k           * Delta_v
            vy_eq = -one_p_eps   / D_k          * Delta_v

            # Deviations from equilibrium
            dug_x = ugx_k      - ux_eq
            dug_y = ugy_k      - uy_eq
            dvp_x = p_type.vxs - vx_eq
            dvp_y = p_type.vys - vy_eq

            # CM deviation: evolves under pure epicyclic rotation (no drag)
            U_cm_x = (dug_x + eps_k * dvp_x) / one_p_eps
            U_cm_y = (dug_y + eps_k * dvp_y) / one_p_eps

            # Relative velocity deviation: decays at coupled rate
            V_x = dvp_x - dug_x
            V_y = dvp_y - dug_y

            omega_epi  = sb.Omega * np.sqrt(np.maximum(4.0 - 2.0*sb.q, 0.0))
            beta       = 2.0 - sb.q
            omega_safe = np.where(omega_epi > 1.0e-30, omega_epi, 1.0e-30)
            cos_t = np.cos(omega_epi * h)
            sin_t = np.sin(omega_epi * h)

            Ucm_x_new = ( U_cm_x * cos_t + U_cm_y * 2.0*sb.Omega / omega_safe * sin_t)
            Ucm_y_new = (-U_cm_x * beta*sb.Omega / omega_safe * sin_t + U_cm_y * cos_t)
            V_x_new   = coupled_decay * ( V_x * cos_t + V_y * 2.0*sb.Omega / omega_safe * sin_t)
            V_y_new   = coupled_decay * (-V_x * beta*sb.Omega / omega_safe * sin_t + V_y * cos_t)

            # New particle velocity
            vx_new = vx_eq + Ucm_x_new + V_x_new
            vy_new = vy_eq + Ucm_y_new + V_y_new
            dvx_jk = vx_new - p_type.vxs
            dvy_jk = vy_new - p_type.vys    

            # New gas velocity from same decomposition (also bounded)
            ug_x_new  = ux_eq + Ucm_x_new - V_x_new / one_p_eps
            ug_y_new  = uy_eq + Ucm_y_new - V_y_new / one_p_eps
            dug_x_jk  = ug_x_new - ugx_k
            dug_y_jk  = ug_y_new - ugy_k

        # Accumulate particle velocity change: weighted sum over sub-clouds
        delta_vx += w[:, s] * dvx_jk
        delta_vy += w[:, s] * dvy_jk

        # Accumulate gas velocity change: local eq. 31, weighted by TSC w
        # Multiple particles can contribute to the same cell; we accumulate
        # and normalise below so the update represents the average Δu from
        # all overlapping sub-clouds.
        np.add.at(dug_x_grid,  k_idx, w[:, s] * dug_x_jk)
        np.add.at(dug_y_grid,  k_idx, w[:, s] * dug_y_jk)
        np.add.at(cell_weight, k_idx, w[:, s])

    return delta_vx, delta_vy, dug_x_grid, dug_y_grid, cell_weight


def _pmbr_gas_update(U, p_type, delta_vx, delta_vy, x_grid_pad, dx, Nx, bc):
    """
    Particle-mesh back reaction (PMBR) gas velocity update (§2.3, Eq. 31).

    Given finalised particle velocity changes delta_vx/vy, scatter the
    total particle momentum change back to the gas grid via TSC and recover
    the gas velocity change from momentum conservation:

        ρg,k Δuₖ = −Σⱼ  mₚⱼ · W(rₚⱼ − rₖ) · Δvⱼ   / dx

    Ghost cells are refreshed after the update so that any subsequent
    apply_bc call does not overwrite the just-updated boundary physical
    cells with stale ghost values.

    Parameters
    ----------
    U          : ndarray (4, N_pad)   conserved state, modified in place
    p_type     : Particle_type        read-only particle state (positions used)
    delta_vx   : (Np,)               particle vx changes to scatter
    delta_vy   : (Np,)               particle vy changes to scatter
    x_grid_pad : (N_pad,)            padded grid cell centres
    dx         : float               cell width
    Nx         : int                 number of physical cells
    bc         : str                 boundary condition type

    Returns
    -------
    U : ndarray (4, N_pad)  with updated rows 1 (ρuₓ) and 2 (ρuᵧ),
        ghost cells refreshed to match updated boundary physical cells.
    """
    dp_x = particles_to_grid(
        p_type.mass * delta_vx, None, p_type.xs, x_grid_pad) / dx
    dp_y = particles_to_grid(
        p_type.mass * delta_vy, None, p_type.xs, x_grid_pad) / dx

    dp_x = _fold_ghost_deposits(dp_x, Nx, bc)
    dp_y = _fold_ghost_deposits(dp_y, Nx, bc)

    # Apply momentum conservation to physical cells only (indices 2..Nx+1).
    # ρg Δu = −Δp_particles  (momentum conservation in each cell)
    U[1, 2:Nx+2] -= dp_x[2:Nx+2]
    U[2, 2:Nx+2] -= dp_y[2:Nx+2]

    # Refresh ghost cells so they match the newly updated boundary physical
    # cells.  Without this, a subsequent apply_bc call would overwrite the
    # boundary physical cells with stale ghost values (the original bug).
    U = apply_bc(U, bc)

    return U


def yang_johansen_drag_step(U, particles, x_grid, h, bc, shear_box=None):
    """
    Yang & Johansen (2016) §2 drag operator D(h) with PMBR (§2.3).

    Updates particle VELOCITIES and gas MOMENTA only — positions are
    advanced separately by _particle_drift in ssprk_step.

    Algorithm
    ---------
    For each particle species:

    1. Compute eps_grid and TSC weights from current particle positions.

    2. Predictor (half-step with PMBR):
       - Run _yj_per_cell_solve for h/2 using gas state at t^n
         → delta_vx_pred, delta_vy_pred
       - Apply PMBR to get the predicted gas velocity at t^n + h/2.
         Using PMBR here (rather than the cheap local update) is essential
         for second-order convergence: the local update has O(1) error in
         the gas velocity which propagates as O(h) error in the corrector.
       - Extract ugx_mid from the predicted U^*.

    3. Corrector (full-step with PMBR):
       - Run _yj_per_cell_solve for h using ugx_mid
         → delta_vx, delta_vy  (final particle velocity changes)
       - Apply PMBR to update gas momenta from delta_vx/vy.

    4. Apply delta_vx/vy to particle velocities.

    The Strang-split outer structure (P D L D P) ensures the combined
    scheme is second-order accurate in time.

    Parameters
    ----------
    U         : ndarray (4, Nx+4)   padded conserved state (modified in place)
    particles : list of Particle_type
    x_grid    : ndarray (Nx,)       physical cell centres
    h         : float               sub-step duration (dt/2 for Strang splitting)
    bc        : str                 boundary condition type
    shear_box : ShearBox or None    enables Coriolis+epicyclic solve

    Returns
    -------
    U         : ndarray (4, Nx+4)   updated gas momenta
    particles : list of Particle_type with updated velocities
    """
    dx    = x_grid[1] - x_grid[0]
    Nx    = len(x_grid)
    N_pad = Nx + 4

    x_grid_pad = np.concatenate([
        [x_grid[0]-2*dx, x_grid[0]-dx],
        x_grid,
        [x_grid[-1]+dx,  x_grid[-1]+2*dx],
    ])
    x0 = x_grid_pad[0]

    for p_type in particles:

        # ── 1. Shared quantities ───────────────────────────────────────
        rho_dust = particles_to_grid(
            p_type.mass * np.ones(p_type.np),
            None, p_type.xs, x_grid_pad) / dx
        rho_dust = apply_bc_gen(_fold_ghost_deposits(rho_dust, Nx, bc),bc)
        rho_gas  = np.maximum(U[0], 1.0e-14)
        eps_grid = rho_dust / rho_gas

        i_c, w = _tsc_weights(p_type.xs, x0, dx)

        # ── 2. Single drag solve + PMBR ───────────────────────────────
        ugx_n = U[1] / rho_gas
        ugy_n = U[2] / rho_gas

        delta_vx, delta_vy, _, _, _ = _yj_per_cell_solve(
            ugx_n, ugy_n, eps_grid, p_type, i_c, w, h, shear_box, N_pad)

        U = _pmbr_gas_update(
            U, p_type, delta_vx, delta_vy, x_grid_pad, dx, Nx, bc)

        # ── 3. Update particle velocities ──────────────────────────────
        p_type.vxs += delta_vx
        p_type.vys += delta_vy

    return U, particles


# ─────────────────────────────────────────────────────────
#  SSPRK time step with Yang-Johansen Strang splitting
# ─────────────────────────────────────────────────────────

def ssprk_step(U, dx, dt, bc="transmissive", tableau="ssprk3",
               recons='WENOZpI', particles=None, x_grid=None,
               shear_box=None,
               gas_sources=[], gas_source_params=[],
               analytic_x=False,
               cs_iso=None):
    """
    Advance U by one time step dt.

    When particles are present, uses a Verlet-ordered Strang splitting:

      P(dt/2)   — x += v^n * dt/2            (drift, pre-drag velocity)
      D(dt/2)   — per-cell analytical drag [+Coriolis if shear_box],
                  velocities and gas momenta updated, positions unchanged
      L(dt)     — pure SSPRK2 hyperbolic advance, NO drag source terms
      D(dt/2)   — second half-drag, positions still at x^{n+1/2}
      P(dt/2)   — x += v^{n+1} * dt/2        (drift, post-drag velocity)
                  + boundary wrapping

    Why this ordering is second order
    ----------------------------------
    D(h) is a pure velocity operator: [D, P] is O(h²) because D changes
    v by O(h), shifting P·x by O(h²).  This satisfies the Strang
    requirement.

    Previously positions were advanced inside D(h), making D a compound
    operator.  Two half-steps D(h/2)∘D(h/2) were not equal to D(h)
    because the second D(h/2) saw a gas state already modified by PMBR
    from the first, giving an O(dt) error per step — first order.

    NOTE on gas_sources: if shear_box is set, do NOT include
    gas_shearing_box_source in gas_sources — those forces are handled
    analytically inside yang_johansen_drag_step and would be
    double-counted here.

    For pure-fluid problems (particles=None), the full SSPRK family
    is available via the tableau argument.

    Parameters
    ----------
    U                : ndarray (4, N+4)   conserved state with ghost cells
    dx               : float              cell width
    dt               : float              time step
    bc               : str                boundary condition type
    tableau          : str                SSPRK method (pure fluid only)
    recons           : str                reconstruction scheme
    particles        : list or None       Particle_type instances
    x_grid           : ndarray (N,)       physical cell centres
    shear_box        : ShearBox or None   enables Coriolis+epicyclic in D(h)
    gas_sources      : list               gas source functions for L(dt)
    gas_source_params: list               params for each gas source
    analytic_x       : bool               use analytic position update
    cs_iso           : float or None      isothermal sound speed

    Returns
    -------
    U_new               if particles is None
    (U_new, particles)  otherwise
    """
    U_n = U.copy()

    # ══════════════════════════════════════════════════════════════════
    #  Particle path: Yang & Johansen Strang splitting
    # ══════════════════════════════════════════════════════════════════
    if particles is not None:

        if cs_iso is not None:
            enforce_isothermal(U_n, cs_iso)
        U_n = apply_bc(U_n, bc)

        # ── Full step structure (Verlet / Strang order) ────────────────
        #
        #   P(dt/2)   x += v^n * dt/2          (drift with pre-drag vel)
        #   D(dt/2)   velocity + gas momentum update
        #   L(dt)     SSPRK2 fluid advance
        #   D(dt/2)   velocity + gas momentum update
        #   P(dt/2)   x += v^{n+1} * dt/2      (drift with post-drag vel)
        #             + boundary wrapping
        #
        # Why this ordering achieves second order
        # ----------------------------------------
        # D(h) is a pure velocity operator: it changes v by O(h) and
        # leaves x unchanged.  P(h) is x += v*h, which changes x by O(h)
        # and leaves v unchanged.  The commutator [D, P] is O(h²) because
        # D changes v by O(h), which shifts P(h)·x by O(h²) — exactly
        # the requirement for Strang splitting to give second-order error.
        #
        # If positions were updated inside D(h) instead (as was done
        # previously), D becomes a compound operator that is not self-
        # consistent under splitting: D(h/2)∘D(h/2) ≠ D(h) because the
        # second D(h/2) sees a gas state already modified by PMBR from
        # the first, giving O(dt) error — first order.

        # ── P(dt/2): first position half-drift ────────────────────────
        # Advance positions with the current (pre-drag) velocity.
        # No boundary wrapping here — mid-step wrapping would corrupt
        # the TSC deposit for particles near the domain edge.

        particles = _particle_drift(particles, 0.5*dt, bc, x_grid)

        # ── D(dt): full drag  ───────────────────────────────────
        # Velocities and gas momenta updated; positions unchanged.
        U_n, particles = yang_johansen_drag_step(
            U_n, particles, x_grid, dt, bc, shear_box)
        if cs_iso is not None:
            enforce_isothermal(U_n, cs_iso)
        U_n = apply_bc(U_n, bc)

        # ── L(dt): SSPRK2 pure hyperbolic fluid advance ───────────────
        # Drag and Coriolis/epicyclic are fully operator-split into D(h).
        # spatial_residual sees only the flux divergence and any remaining
        # gas_sources (e.g. external gravity NOT handled by shear_box).
        # Particle positions are frozen during this stage.
        L_n    = spatial_residual(U_n, dx, bc, recons,
                                  gas_sources, gas_source_params)
        U_star = U_n + dt * L_n
        if cs_iso is not None:
            enforce_isothermal(U_star, cs_iso)
        U_star = apply_bc(U_star, bc)

        L_star = spatial_residual(U_star, dx, bc, recons,
                                  gas_sources, gas_source_params)
        U_new  = 0.5*(U_n + U_star) + 0.5*dt*L_star
        if cs_iso is not None:
            enforce_isothermal(U_new, cs_iso)
        U_new  = apply_bc(U_new, bc)

        # ── P(dt/2): second position half-drift + boundary wrapping ───
        # Advance positions with the post-drag velocity v^{n+1}.
        # Together with the first P(dt/2) this gives:
        #   x^{n+1} = x^n + v^n*dt/2 + v^{n+1}*dt/2
        # which is the trapezoidal rule — second-order accurate.
        # Boundary wrapping and the shear_periodic velocity kick are
        # applied here, now that t^{n+1} positions are fully determined.
        particles = _particle_drift_with_bc(
            particles, 0.5*dt, bc, x_grid, shear_box)

        return U_new, particles

    # ══════════════════════════════════════════════════════════════════
    #  Fluid only: general SSPRK stages loop
    # ══════════════════════════════════════════════════════════════════
    alpha, beta = _ssprk_tableau(tableau)
    s = alpha.shape[0] - 1
    stages = [None] * (s + 1)
    stages[0] = U_n

    for i in range(1, s + 1):
        acc = np.zeros_like(U)
        for k in range(i):
            if alpha[i, k] != 0.0:
                acc += alpha[i, k] * stages[k]
            if beta[i, k] != 0.0:
                acc += beta[i, k] * dt * spatial_residual(
                    stages[k], dx, bc, recons,
                    gas_sources, gas_source_params)
        if cs_iso is not None:
            enforce_isothermal(acc, cs_iso)
        stages[i] = acc

    return apply_bc(stages[s], bc)


# ─────────────────────────────────────────────────────────
#  Solver driver
# ─────────────────────────────────────────────────────────

def solve(
    rho_init, ux_init, p_init,
    x,
    t_end,
    uy_init           = None,
    cfl               = 0.9,
    bc                = "transmissive",
    ssprk             = "ssprk3",
    verbose           = True,
    recons            = 'WENOZpI',
    particles         = None,
    shear_box         = None,
    gas_sources       = [],
    gas_source_params = [],
    analytic_x        = False,
    cs_iso            = 'auto',
):
    """
    Run the 1.5-D Godunov solver from t=0 to t=t_end.

    When particles are present, the Yang & Johansen (2016) Strang-split
    drag operator is used.  The timestep is limited only by the
    hydrodynamic CFL condition and the particle advection velocity —
    NOT by the drag timescale, since the YJ operator is unconditionally
    stable for any dt/τ_s and any eps.

    If shear_box is set, Coriolis and epicyclic forces are handled
    analytically inside yang_johansen_drag_step.  In that case,
    gas_shearing_box_source must NOT be included in gas_sources.

    Parameters
    ----------
    rho_init     : array-like (N,)    initial density
    ux_init      : array-like (N,)    initial radial velocity
    p_init       : array-like (N,)    initial pressure
    x            : array-like (N,)    cell-centre positions (uniform)
    t_end        : float              end time
    uy_init      : array-like or None initial azimuthal velocity
    cfl          : float              Courant number
    bc           : str                boundary condition type
    ssprk        : str                SSPRK method (fluid-only path)
    verbose      : bool               print progress every 100 steps
    recons       : str                reconstruction scheme
    particles    : list or None       Particle_type instances
    shear_box    : ShearBox or None   enables Coriolis+epicyclic in D(h)
    gas_sources       : list          gas source functions for L(dt) only
    gas_source_params : list          params for each gas source
    analytic_x        : bool          analytic position update (unused in
                                      YJ leapfrog path; reserved)
    cs_iso       : float, 'auto', or None
        'auto': isothermal when particles present (cs from ICs), else None.

    Returns
    -------
    rho, ux, uy, p, t                if particles is None
    rho, ux, uy, p, t, particles     otherwise
    """
    cfl_eff = {"ssprk1":1.0, "ssprk2":1.0, "ssprk3":1.0,
               "ssprk4":2.0, "ssprk5":1.508}.get(ssprk.lower(), 1.0)

    rho_init = np.asarray(rho_init, dtype=float)
    ux_init  = np.asarray(ux_init,  dtype=float)
    p_init   = np.asarray(p_init,   dtype=float)
    x        = np.asarray(x,        dtype=float)
    uy_init  = (np.zeros_like(rho_init)
                if uy_init is None
                else np.asarray(uy_init, dtype=float))

    dx = x[1] - x[0]

    def pad(arr):
        return np.concatenate([[arr[0], arr[0]], arr, [arr[-1], arr[-1]]])

    U = prim_to_cons(pad(rho_init), pad(ux_init), pad(uy_init), pad(p_init))

    if cs_iso == 'auto':
        cs_iso = (np.sqrt(np.mean(p_init) / np.mean(rho_init))
                  if particles is not None else None)

    t    = 0.0
    step = 0

    while t < t_end:
        dt_cfl = compute_dt(U[:, 2:-2], dx, cfl * cfl_eff,
                            particles=particles)
        dt = min(dt_cfl, t_end - t)

        if particles is None:
            U = ssprk_step(U, dx, dt, bc, ssprk, recons,
                           gas_sources=gas_sources,
                           gas_source_params=gas_source_params,
                           cs_iso=cs_iso)
        else:
            U, particles = ssprk_step(
                U, dx, dt, bc, ssprk, recons,
                particles, x,
                shear_box=shear_box,
                gas_sources=gas_sources,
                gas_source_params=gas_source_params,
                analytic_x=analytic_x,
                cs_iso=cs_iso,
            )

        t    += dt
        step += 1
        if verbose and step % 100 == 0:
            print(f"  step {step:5d}  t = {t:.4f}  dt = {dt:.2e}")

    rho_f, ux_f, uy_f, p_f = cons_to_prim(U[:, 2:-2])

    if particles is None:
        return rho_f, ux_f, uy_f, p_f, t
    else:
        return rho_f, ux_f, uy_f, p_f, t, particles


# ─────────────────────────────────────────────────────────
#  Test problems
# ─────────────────────────────────────────────────────────

def sod_shock_tube(N=200):
    x   = np.linspace(0.0, 1.0, N)
    rho = np.where(x < 0.5, 1.0,   0.125)
    ux  = np.zeros(N);  uy = np.zeros(N)
    p   = np.where(x < 0.5, 1.0,   0.1)
    return x, rho, ux, uy, p, 0.2


def lax_problem(N=200):
    x   = np.linspace(0.0, 1.0, N)
    rho = np.where(x < 0.5, 0.445,  0.5)
    ux  = np.where(x < 0.5, 0.698,  0.0)
    uy  = np.zeros(N)
    p   = np.where(x < 0.5, 3.528,  0.571)
    return x, rho, ux, uy, p, 0.14


def shu_osher(N=400):
    x   = np.linspace(-5.0, 5.0, N)
    rho = np.where(x < -4.0, 3.857143, 1.0 + 0.2*np.sin(5.0*x))
    ux  = np.where(x < -4.0, 2.629369, 0.0)
    uy  = np.zeros(N)
    p   = np.where(x < -4.0, 10.3333,  1.0)
    return x, rho, ux, uy, p, 1.8


def two_blast_waves(N=400):
    x   = np.linspace(0.0, 1.0, N)
    rho = np.ones(N);  ux = np.zeros(N);  uy = np.zeros(N)
    p   = np.where(x < 0.1, 1000.0, np.where(x > 0.9, 100.0, 0.01))
    return x, rho, ux, uy, p, 0.038


def particle_drag_single(N=100, vgx=0., vgy=0., mp=1.,
                         vpx=1., vpy=0., tstop=1.):
    """Single particle decelerating in uniform gas — exact solution available."""
    x         = np.linspace(0.0, 100.0, N)
    rho       = np.ones(N)
    ux        = np.full(N, vgx);  uy = np.full(N, vgy)
    p         = np.ones(N) * 1e-4
    particles = [Particle_type(np.array([10.0]), np.array([vpx]),
                               np.array([vpy]), tstop, mp)]
    return x, rho, ux, uy, p, 5.0, particles


def nsh_equilibrium(N=128, epsilon=1.0, St=0.1, eta=0.05, Omega=1.0, q=1.5,
                    N_particles=None, L=1.0, p0=1.0):
    """
    NSH equilibrium drift test (Nakagawa, Sekiya & Hayashi 1986).

    The system is initialised exactly at the NSH equilibrium.  With the
    Yang-Johansen shearing-box solve, this equilibrium should be
    preserved to machine precision at any timestep.
    """
    if N_particles is None:
        N_particles = N

    v_K     = Omega
    Delta_v = eta * v_K
    t_stop  = St / Omega
    D       = (1.0 + epsilon)**2 + St**2

    u_gx = +2.0 * epsilon * St * Delta_v / D
    u_gy = -(1.0 + epsilon + St**2) * Delta_v / D
    v_px = -2.0 * St * Delta_v / D
    v_py = -(1.0 + epsilon) * Delta_v / D

    x   = np.linspace(0.0, L, N, endpoint=False)
    rho = np.ones(N)
    ux  = np.full(N, u_gx);  uy = np.full(N, u_gy)
    p   = np.full(N, p0)

    x_p   = np.linspace(0.0, L, N_particles, endpoint=False) + 0.5*L/N_particles
    m_p   = epsilon * L / N_particles
    particles = [Particle_type(x_p,
                               np.full(N_particles, v_px),
                               np.full(N_particles, v_py),
                               tstop=t_stop, mass=m_p)]
    shear_box = ShearBox(Omega=Omega, q=q)
    t_end     = 10.0 * (2.0 * np.pi / Omega)

    return x, rho, ux, uy, p, t_end, particles, shear_box


def clumpy_particle_drift(N=128, N_particles=512, clump_fraction=0.1,
                           epsilon=1.0, tstop=0.1, vp=1.0, vg=0.0):
    """
    Clumped particle distribution streaming through uniform gas.

    Good test for momentum conservation at high local eps (clump centre
    can have eps >> 1) and for the unconditional stability of the
    Yang-Johansen drag operator.

    epsilon = the maximum local epsilon near the particles at the start
    """
    L  = 1.0
    x  = np.linspace(0.0, L, N, endpoint=False)
    rho = np.ones(N);  ux = np.full(N, vg);  uy = np.zeros(N)
    p   = np.ones(N) * 1e-2

    clump_centre = 0.25 * L
    clump_half   = 0.5 * clump_fraction * L
    x_lo = clump_centre - clump_half
    x_hi = clump_centre + clump_half

    x_p = (np.linspace(x_lo, x_hi, N_particles, endpoint=False)
           + 0.5*(x_hi-x_lo)/N_particles)
    m_p = epsilon * L * clump_fraction / N_particles

    particles = [Particle_type(
        xs=x_p, vxs=np.full(N_particles, vp),
        vys=np.zeros(N_particles), tstop=tstop, mass=m_p)]

    return x, rho, ux, uy, p, 5.0*tstop, particles


def sod_exact(x, t, gamma=1.4):
    """Exact solution of the Sod shock tube."""
    from scipy.optimize import brentq

    rho_L, u_L, p_L = 1.0,   0.0, 1.0
    rho_R, u_R, p_R = 0.125, 0.0, 0.1
    x_0 = 0.5

    c_L = np.sqrt(gamma*p_L/rho_L);  c_R = np.sqrt(gamma*p_R/rho_R)
    gm1 = gamma - 1.0;               gp1 = gamma + 1.0

    def f(p, p_k, rho_k, c_k):
        if p <= p_k:
            return 2.*c_k/gm1 * ((p/p_k)**(gm1/(2.*gamma)) - 1.)
        A = 2./(gp1*rho_k);  B = gm1/gp1*p_k
        return (p - p_k)*np.sqrt(A/(p+B))

    p_star = brentq(
        lambda p: f(p,p_L,rho_L,c_L)+f(p,p_R,rho_R,c_R)+(u_R-u_L),
        1e-10, 10*max(p_L,p_R), xtol=1e-12, rtol=1e-12)
    u_star = u_L - f(p_star, p_L, rho_L, c_L)

    if p_star <= p_L:
        rho_star_L = rho_L*(p_star/p_L)**(1./gamma)
        c_star_L   = c_L  *(p_star/p_L)**(gm1/(2.*gamma))
    else:
        rho_star_L = rho_L*(p_star/p_L+gm1/gp1)/(gm1/gp1*p_star/p_L+1.)
        c_star_L   = None

    rho_star_R = (rho_R*(p_star/p_R+gm1/gp1)/(gm1/gp1*p_star/p_R+1.)
                  if p_star > p_R else rho_R*(p_star/p_R)**(1./gamma))

    S_head    = u_L - c_L
    S_tail    = u_star - c_star_L
    S_shock   = u_R + c_R*np.sqrt(gp1/(2.*gamma)*p_star/p_R+gm1/(2.*gamma))
    S_contact = u_star

    xi = (x - x_0) / t
    rho_out = np.empty_like(x);  u_out = np.empty_like(x);  p_out = np.empty_like(x)

    for i, xi_i in enumerate(xi):
        if xi_i <= S_head:
            rho_out[i], u_out[i], p_out[i] = rho_L, u_L, p_L
        elif xi_i <= S_tail:
            u_fan   = 2./gp1*(c_L + gm1/2.*u_L + xi_i)
            c_fan   = c_L - gm1/2.*(u_fan - u_L)
            rho_fan = rho_L*(c_fan/c_L)**(2./gm1)
            p_fan   = p_L*(rho_fan/rho_L)**gamma
            rho_out[i], u_out[i], p_out[i] = rho_fan, u_fan, p_fan
        elif xi_i <= S_contact:
            rho_out[i], u_out[i], p_out[i] = rho_star_L, u_star, p_star
        elif xi_i <= S_shock:
            rho_out[i], u_out[i], p_out[i] = rho_star_R, u_star, p_star
        else:
            rho_out[i], u_out[i], p_out[i] = rho_R, u_R, p_R

    return rho_out, u_out, np.zeros_like(rho_out), p_out


# ─────────────────────────────────────────────────────────
#  Command-line entry point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    problem_map = {
        "sod":   sod_shock_tube,
        "lax":   lax_problem,
        "shu":   shu_osher,
        "blast": two_blast_waves,
    }
    problem_name = sys.argv[1] if len(sys.argv) > 1 else "sod"
    ssprk_name   = sys.argv[2] if len(sys.argv) > 2 else "ssprk3"

    if problem_name not in problem_map:
        print(f"Unknown problem {problem_name!r}. Choose: {list(problem_map)}")
        sys.exit(1)

    x, rho0, ux0, uy0, p0, t_end = problem_map[problem_name]()

    print(f"Running {problem_name} with {ssprk_name} …")
    rho, ux, uy, p, t = solve(
        rho0, ux0, p0, x, t_end, uy_init=uy0,
        ssprk=ssprk_name, cfl=0.9, verbose=True,
    )
    print(f"Done.  t_final = {t:.6f}")

    exact_fn = sod_exact if problem_name == "sod" else None

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    fig.subplots_adjust(hspace=0.4)
    fields  = [(rho,r"$\rho$"),(ux,r"$u_x$"),(uy,r"$u_y$"),(p,r"$p$")]
    colours = ["#2171b5","#238b45","#fd8d3c","#cb181d"]
    for ax, (q,lbl), col in zip(axes, fields, colours):
        ax.plot(x, q, color=col, lw=1.5, label="numerical")
        ax.set_ylabel(lbl, fontsize=10)
        ax.grid(True, lw=0.4, alpha=0.4)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    if exact_fn is not None:
        erho,eux,euy,ep = exact_fn(x, t)
        for ax, eq in zip(axes, [erho,eux,euy,ep]):
            ax.plot(x, eq, "k--", lw=1.0, alpha=0.6, label="exact")
        axes[0].legend(fontsize=9)
    axes[-1].set_xlabel("x", fontsize=10)
    fig.suptitle(f"{problem_name}  |  {ssprk_name}  |  t = {t:.4f}", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{problem_name}_{ssprk_name}.png", dpi=150)
    print(f"Figure saved to {problem_name}_{ssprk_name}.png")
