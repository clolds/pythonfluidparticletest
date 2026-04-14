"""
1.5D Godunov Finite Volume Fluid–Particle Solver
=================================================
Solves the 1.5-D Euler equations coupled to Lagrangian dust particles
via aerodynamic (Epstein/Stokes) drag.  The fluid is advanced with a
Godunov finite-volume scheme using an HLLC Riemann solver, and
particles are integrated with exponential semi-analytic methods that
are unconditionally stable in the stiff-drag regime (Δt ≫ τ_s).

The gas–particle coupling follows the predictor-corrector scheme of
Mignone et al. (2018, ApJ 859:13, "Paper I") §3.1.2 and Mignone et al.
(2019, ApJS 244:38, "Paper II") §3.1, which is second-order accurate
in time and conserves total momentum to machine precision.

Grid & Equations
----------------
The grid is 1-D (scalar x positions) but the velocity field is a
2-component vector (uₓ, uᵧ).  uᵧ is advected passively by uₓ with
no y-pressure gradient ("1.5-D" or "2-velocity" model).

Governing equations (conservative form):
  ∂U/∂t + ∂F/∂x = S_gas + S_drag

  U = [ρ, ρuₓ, ρuᵧ, E]ᵀ                      (conserved variables)
  F = [ρuₓ, ρuₓ²+p, ρuₓuᵧ, uₓ(E+p)]ᵀ        (fluxes)
  E = p/(γ-1) + ½ρ(uₓ²+uᵧ²)                  (total energy, ideal gas)

Equation of state:
  Adiabatic:   p = (γ-1)(E - ½ρ(uₓ²+uᵧ²))    (default for pure fluid)
  Isothermal:  p = ρ·cs²                        (default when particles present;
                                                 controlled by cs_iso parameter)

Particle Module
---------------
Particles carry scalar position x and 2-component velocity (vₓ, vᵧ).
Each species has a constant stopping time τ_s and per-particle mass m_p.
Drag coupling (linear in relative velocity) acts on both components:

  dvₚ/dt = f(x,v) + (v_gas − vₚ) / τ_s

where f includes external forces (gravity, Coriolis, etc.).

Particle–grid coupling uses the Triangular Shaped Cloud (TSC) weighting
function for both deposition (particle → grid) and interpolation
(grid → particle), ensuring adjoint consistency.

Particle integrators (set via integrator='EM' or 'SSA'):

  EM   – Exponential Midpoint (Mignone et al. 2019, §3.2, eq. 32-34).
         Implicit in the midpoint velocity; solved analytically for pure
         drag and for the shearing-box 2×2 system (Coriolis + epicyclic),
         or by fixed-point iteration (with Newton fallback) for general
         nonlinear velocity-dependent forces.  Default.

  SSA  – Staggered Semi-Analytic (Fung & Muley 2019, ApJS 244:42).
         Explicit two-evaluation method using a staggered half-step to
         improve the terminal-velocity estimate.  Second-order accurate,
         symplectic in the drag-free limit.

Both integrators are unconditionally stable for arbitrarily stiff drag
(Δt/τ_s → ∞): the exponential factor exp(−Δt/τ_s) ∈ [0,1] ensures
the velocity update is always a bounded convex combination.

Position update options (set via analytic_x=True/False):

  Leapfrog (default):  x_{n+1} = x_{1/2} + v_{n+1}·Δt/2
  Analytic:            x_{n+1} = x_n + v_t·Δt + (v_n − v_t)·τ_s·(1−e^{−Δt/τ_s})

  where v_t = f·τ_s + v_gas is the terminal velocity.  The analytic form
  is exact for constant force and gas velocity and provides bounded
  position errors regardless of stiffness.

Fluid–Particle Coupling (Mignone predictor-corrector)
-----------------------------------------------------
When particles are present, the time step uses SSPRK2 with the
following predictor-corrector structure:

  1. L^n     = spatial_residual(U^n)
  2. pred    = predicted drag source S^n_D (exponential estimate)
  3. U*      = U^n + Δt·L^n + pred          (predictor)
  4. U_half  = (U^n + U*) / 2               (half-time state)
  5. Push particles using U_half → deposit   (actual momentum transfer)
  6. L*      = spatial_residual(U*)
  7. U^{n+1} = U^n + Δt/2·(L^n + L*) + deposit/Δx   (corrector)

The predictor drag (step 2) enters U* so that L(U*) is evaluated at a
better state, but only the ACTUAL particle deposit (step 5) appears in
the final answer.  This guarantees machine-precision total momentum
conservation (gas + dust) by Newton's 3rd law.

Fluid Time Integration — SSPRK (Shu-Osher form)
------------------------------------------------
For pure-fluid problems (no particles), the general SSPRK family is
available in Shu-Osher form:

  U^(0)    = U^n
  U^(i)    = Σ_{k=0}^{i-1}  α_{ik} U^(k)  +  β_{ik} Δt L(U^(k))
  U^{n+1}  = U^(s)

Built-in tableaux (Gottlieb, Shu & Tadmor 2001; Gottlieb 2005):
  ssprk1   – forward Euler          (1st order, 1 stage,  CFL ≤ 1)
  ssprk2   – Heun / optimal SSP-RK2 (2nd order, 2 stages, CFL ≤ 1)
  ssprk3   – Shu-Osher SSP-RK3      (3rd order, 3 stages, CFL ≤ 1)
  ssprk4   – optimal SSP-RK(4,3)    (3rd order, 4 stages, CFL ≤ 2)
  ssprk5   – optimal SSP-RK(5,4)    (4th order, 5 stages, CFL ≤ 1.508)

Boundary Conditions
-------------------
  transmissive   – zero-gradient (copy nearest interior cell)
  reflective     – reflect normal velocity
  periodic       – wrap ghost cells from opposite end
  shear_periodic – periodic + azimuthal velocity kick Δvᵧ = ±q·Ω·Lₓ
                   when a particle crosses the radial domain boundary

Reconstruction Schemes
----------------------
  godunov  – piecewise-constant (1st order)
  WENOZ    – WENO-Z 5th-order (Borges et al. 2008)
  WENOZpI  – WENO-Z+I (Luo & Wu 2021)

Shearing-box Physics
--------------------
Gas source terms:
  gas_shearing_box_source  –  Coriolis + epicyclic + pressure-gradient
                              body force for the gas momentum equations

Particle forces:
  shearing_box_force       –  Coriolis + epicyclic acceleration for particles
                              (pass via forces=[shearing_box_force])

Test Problems
-------------
  sod_shock_tube      – Sod (1978) shock tube
  lax_problem         – Lax strong shock
  shu_osher           – Shu-Osher shock/entropy wave interaction
  two_blast_waves     – Woodward & Colella interacting blast waves
  particle_drag_single – single particle decelerating in uniform gas
  nsh_equilibrium     – Nakagawa-Sekiya-Hayashi equilibrium drift
  clumpy_particle_drift – concentrated particle clump (conservation test)

References
----------
  Mignone et al. (2018) ApJ 859:13  – MHD-PIC, RK2 predictor-corrector
  Mignone et al. (2019) ApJS 244:38 – dust module, exponential midpoint
  Fung & Muley (2019) ApJS 244:42   – SSA particle integrator
  Gottlieb, Shu & Tadmor (2001)     – SSPRK methods
  Borges et al. (2008)              – WENO-Z reconstruction
  Luo & Wu (2021)                   – WENO-Z+I reconstruction
  Toro (2009)                       – HLLC Riemann solver

Usage
-----
  # Pure fluid
  rho, ux, uy, p, t = gp.solve(rho0, ux0, p0, x, t_end)

  # With particles (isothermal by default)
  rho, ux, uy, p, t, particles = gp.solve(
      rho0, ux0, p0, x, t_end,
      particles=particles, integrator='EM')

  # Command line (fluid only)
  python godunov1p5d_particles.py [problem] [ssprk_order]
"""

from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# ─────────────────────────────────────────────────────────
#  Global constants
# ─────────────────────────────────────────────────────────

GAMMA = 1.4   # ratio of specific heats (ideal diatomic gas)

# Orbital shearing-box parameters.
#   Omega : angular frequency at the reference radius
#   q     : shear parameter  (q = 3/2 for a Keplerian disc)
ShearBox = namedtuple("ShearBox", ["Omega", "q"])


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
    """
    Convert primitive variables (ρ, uₓ, uᵧ, p) to conserved [ρ, ρuₓ, ρuᵧ, E].

    Total energy includes both velocity components:
        E = p/(γ-1) + ½ρ(uₓ²+uᵧ²)
    """
    E = p / (GAMMA - 1.0) + 0.5 * rho * (ux**2 + uy**2)
    return np.array([rho, rho*ux, rho*uy, E])


def cons_to_prim(U):
    """
    Convert conserved vector U = [ρ, ρuₓ, ρuᵧ, E] to primitives (ρ, uₓ, uᵧ, p).

    uᵧ does not appear in the x-direction pressure gradient, but contributes
    ½ρuᵧ² to the total energy.

    Returns
    -------
    rho, ux, uy, p : 1-D arrays
    """
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
    """
    Recompute the energy row (row 3) of the conserved state U so that
    the pressure satisfies the isothermal equation of state p = ρ·cs².

    This replaces the energy equation with the isothermal constraint,
    ensuring pressure is always positive and consistent with density.
    The velocity field is unchanged.

    Parameters
    ----------
    U      : ndarray (4, N)   conserved state (modified in place)
    cs_iso : float            isothermal sound speed

    Returns
    -------
    U : the same array, modified in place
    """
    rho = np.maximum(U[0], 1.0e-14)
    p   = rho * cs_iso**2 / GAMMA
    U[3] = p / (GAMMA - 1.0) + 0.5 * (U[1]**2 + U[2]**2) / rho
    return U


def euler_flux(rho, ux, uy, p):
    """
    Physical flux vector F(U) in the x-direction for the 1.5-D Euler equations.

    uᵧ is advected by uₓ with no y-pressure gradient:

        F = [ ρuₓ,  ρuₓ²+p,  ρuₓuᵧ,  uₓ(E+p) ]ᵀ
    """
    E = p / (GAMMA - 1.0) + 0.5 * rho * (ux**2 + uy**2)
    return np.array([
        rho * ux,
        rho * ux**2 + p,
        rho * ux * uy,        # uᵧ advected passively; no ∂p/∂y term
        ux  * (E + p),
    ])


# ─────────────────────────────────────────────────────────
#  Riemann solver  (HLLC, 1.5-D)
# ─────────────────────────────────────────────────────────

def riemann_hllc(UL, UR):
    """
    HLLC approximate Riemann solver for the 1.5-D Euler equations.

    Wave speeds are determined solely by uₓ and c (x-propagating signals).
    uᵧ is a contact-wave field: it is copied from the respective side into
    each HLLC star state unchanged (Toro 2009, §10.4).

    The returned numerical flux has four components: [ρ, ρuₓ, ρuᵧ, E].

    Reference: Toro (2009) "Riemann Solvers and Numerical Methods for Fluid
    Dynamics", Chapter 10.
    """
    rhoL, uxL, uyL, pL = cons_to_prim(UL)
    rhoR, uxR, uyR, pR = cons_to_prim(UR)

    cL = sound_speed(rhoL, pL)
    cR = sound_speed(rhoR, pR)

    # Pressure estimate via PVRS (primitive variable Riemann solver)
    rho_bar = 0.5 * (rhoL + rhoR)
    c_bar   = 0.5 * (cL   + cR  )
    p_star  = np.maximum(
        0.0, 0.5*(pL + pR) - 0.5*(uxR - uxL)*rho_bar*c_bar)

    # Wave-speed estimates (uₓ only — uᵧ does not propagate in x)
    qL = np.where(p_star > pL,
                  np.sqrt(1.0 + (GAMMA+1)/(2*GAMMA)*(p_star/pL - 1)), 1.0)
    qR = np.where(p_star > pR,
                  np.sqrt(1.0 + (GAMMA+1)/(2*GAMMA)*(p_star/pR - 1)), 1.0)

    SL = uxL - cL*qL    # left  wave speed
    SR = uxR + cR*qR    # right wave speed

    # Contact / middle wave speed S*
    denom  = rhoL*(SL - uxL) - rhoR*(SR - uxR)
    S_star = (pR - pL
              + rhoL*uxL*(SL - uxL)
              - rhoR*uxR*(SR - uxR)) / denom

    FL = euler_flux(rhoL, uxL, uyL, pL)
    FR = euler_flux(rhoR, uxR, uyR, pR)

    def star_state(U, rho, ux, uy, p, S):
        """
        HLLC star state.  The tangential velocity uᵧ is constant across
        the contact wave, so (ρuᵧ)* = factor · uᵧ.
        Energy uses the standard HLLC formula (Toro eq. 10.38).
        """
        factor = rho * (S - ux) / (S - S_star)
        E_spec = U[3] / rho                       # specific total energy E/ρ
        return np.array([
            factor,
            factor * S_star,
            factor * uy,                           # uᵧ unchanged across contact
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
    """
    WENO-Z nonlinear weights for the left-biased reconstruction u⁻_{i+½}.

    Three candidate 3rd-order polynomials (Jiang & Shu 1996):
      q0  stencil {i-2,i-1,i}
      q1  stencil {i-1, i,i+1}
      q2  stencil { i,i+1,i+2}

    WENO-Z global smoothness indicator (Borges et al. 2008):
      τ5 = |b0 - b2|,   αk = dk·(1 + τ5/(ε+bk))
    """
    eps = 1.0e-36

    q0 = ( 2.0*vm2 -  7.0*vm1 + 11.0*v0 ) / 6.0
    q1 = (         -  1.0*vm1 +  5.0*v0  +  2.0*vp1) / 6.0
    q2 = ( 2.0*v0  +  5.0*vp1 -  1.0*vp2) / 6.0

    b0 = (13./12.)*(vm2 - 2.*vm1 + v0 )**2 + (1./4.)*(vm2 - 4.*vm1 + 3.*v0)**2
    b1 = (13./12.)*(vm1 - 2.*v0  + vp1)**2 + (1./4.)*(vm1 - vp1)**2
    b2 = (13./12.)*(v0  - 2.*vp1 + vp2)**2 + (1./4.)*(3.*v0 - 4.*vp1 + vp2)**2

    tau5 = np.abs(b0 - b2)
    d0, d1, d2 = 1./10., 3./5., 3./10.

    a0 = d0 * (1. + tau5/(eps+b0))
    a1 = d1 * (1. + tau5/(eps+b1))
    a2 = d2 * (1. + tau5/(eps+b2))
    a_sum = a0 + a1 + a2

    return (a0*q0 + a1*q1 + a2*q2) / a_sum


def _weno_zpi_weights(vm2, vm1, v0, vp1, vp2):
    """
    WENO-Z+I nonlinear weights for the left-biased reconstruction u⁻_{i+½}.

    Implements Eq. (12) of Luo & Wu, Computers and Fluids 218 (2021) 104855.

    Differences from WENO-Z:
    1. τ5 term is squared: (τ5/(βk+ε))²
    2. Extra term  λ·α·βk/(βmax+ε)  raises weight of less-smooth stencils
    3. Adaptive factor α = 1 - βmin/(βmax+ε)  suppresses boost when smooth
    4. λ = 3.3  (grid-independent, from ADR analysis, Fig. 4 of Luo & Wu)
    """
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
    """
    Compute left (UL) and right (UR) interface states at every face
    using the supplied WENO weight function.

    Face k lies between padded cells k and k+1.
      UL[:,k]  = u⁻_{k+½}  (left-biased)
      UR[:,k]  = u⁺_{k+½}  (right-biased)

    Works for any state vector width (3 or 4 components).
    """
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
    """
    Apply boundary conditions using two ghost cells on each side.

    Ghost layout (2 ghost cells each side, N physical cells):
        index:   0    1  | 2 … N+1 |  N+2  N+3
                 gl0  gl1  phys       gr0   gr1

    Supported modes
    ---------------
    transmissive  – zero-gradient (copy nearest interior cell)
    reflective    – reflect normal velocity (uₓ sign-flip in ghost cells)
    periodic      – wrap ghost cells from the opposite physical end
    shear_periodic – identical to periodic for the fluid; the azimuthal
                    shear kick for particles is handled in particles_step
    """
    if bc == "transmissive":
        U[:, 0]  = U[:, 2];  U[:, 1]  = U[:, 2]
        U[:, -1] = U[:, -3]; U[:, -2] = U[:, -3]

    elif bc == "reflective":
        # Copy all components, then flip uₓ (row 1) in ghost cells
        U[:, 0]  = U[:, 2];  U[1, 0]  = -U[1, 2]
        U[:, 1]  = U[:, 2];  U[1, 1]  = -U[1, 2]
        U[:, -1] = U[:, -3]; U[1, -1] = -U[1, -3]
        U[:, -2] = U[:, -3]; U[1, -2] = -U[1, -3]

    elif bc in ("periodic", "shear_periodic"):
        # Left  ghosts ← last two physical cells (right end)
        U[:, 0]  = U[:, -4]   # gl0 ← phys[N-1]
        U[:, 1]  = U[:, -3]   # gl1 ← phys[N]
        # Right ghosts ← first two physical cells (left end)
        U[:, -2] = U[:, 2]    # gr0 ← phys[1]
        U[:, -1] = U[:, 3]    # gr1 ← phys[2]

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
    Compute the semi-discrete right-hand side L(U) using HLLC fluxes,
    plus any supplied volumetric source terms.

    Only the interior cells (indices 2:-2) are updated; ghost cells are
    filled by apply_bc before flux evaluation.

    Gas source functions are called as:
        S = source(U, params)   →  ndarray, same shape as U
    and must return non-zero values only in the physical cells (2:-2).
    They are evaluated at the current stage state U and added to L,
    so they participate fully in the SSPRK time integration.

    Returns
    -------
    L : np.ndarray, same shape as U   (dU/dt = L(U); ghost entries = 0)
    """
    U = apply_bc(U, bc)
    n_cells = U.shape[1]
    F = np.zeros_like(U)

    # Pre-compute WENO states once per residual call (not per face)
    if recons in ('WENOZ', 'WENOZpI'):
        wfn = _weno_z_weights if recons == 'WENOZ' else _weno_zpi_weights
        UL_all, UR_all = reconstruct_weno_z(U, wfn)

    for i in range(1, n_cells - 1):
        match recons:
            case 'godunov':
                F[:, i] = riemann_hllc(U[:, i], U[:, i+1])
            case 'WENOZ':
                F[:, i] = riemann_hllc(UL_all[:, i], UR_all[:, i])
            case 'WENOZpI':
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
    Compute the maximum stable time step via the CFL condition.

    Fluid:
        dt_fluid = cfl * dx / max(|uₓ| + c)

    Particles (if supplied):
        dt_particle = cfl * dx / max(|vₓ|)

    Only vₓ limits the particle time step because positions advance only
    in x.  vᵧ does not contribute to radial displacement.

    The returned dt is the minimum of the fluid and all particle
    constraints, so this must be called every step with the current
    particle list to reflect velocities that change during the simulation.

    Parameters
    ----------
    U         : ndarray (4, Nx)    physical-cell conserved state (no ghosts)
    dx        : float              cell width
    cfl       : float              Courant number
    particles : list of Particle_type or None

    Returns
    -------
    dt : float
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
    """
    Return (alpha, beta) coefficient arrays for the requested SSPRK method
    in Shu-Osher form.

    Each stage i (1-indexed) is:
        U^(i) = sum_{k=0}^{i-1}  alpha[i,k] * U^(k)
                                + beta[i,k]  * dt * L(U^(k))

    Built-in methods
    ----------------
    ssprk1  : Forward Euler, 1 stage, order 1, CFL_eff = 1
    ssprk2  : Optimal SSP-RK2 (Heun), 2 stages, order 2, CFL_eff = 1
    ssprk3  : Shu-Osher SSP-RK3, 3 stages, order 3, CFL_eff = 1
    ssprk4  : Optimal SSP-RK(4,3), 4 stages, order 3, CFL_eff = 2
    ssprk5  : Optimal SSP-RK(5,4), 5 stages, order 4, CFL_eff ≈ 1.508

    References
    ----------
    Gottlieb, Shu & Tadmor (2001) SIAM Review 43(1):89-112
    Gottlieb (2005) J. Sci. Comput. 25:105-128  (ssprk5 coefficients)
    """
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
    """
    Compute TSC (quadratic B-spline) weights for a single or array of particles.

    The TSC shape function W(ξ), ξ = (x_p - x_j)/dx:
        ¾ - ξ²              |ξ| < ½
        ½(3/2 - |ξ|)²       ½ ≤ |ξ| < 3/2
        0                    |ξ| ≥ 3/2

    Returns
    -------
    i_c : ndarray of int   index of nearest grid node per particle
    w   : ndarray (N_p,3)  weights [W_{i_c-1}, W_{i_c}, W_{i_c+1}]
    """
    x_p    = np.asarray(x_p, dtype=float)
    scalar = x_p.ndim == 0
    x_p    = np.atleast_1d(x_p)

    eta = (x_p - x0) / dx
    i_c = np.round(eta).astype(int)
    delta = eta - i_c

    w = np.empty((x_p.size, 3), dtype=float)
    w[:, 0] = 0.5 * (0.5 - delta)**2
    w[:, 1] = 0.75 - delta**2
    w[:, 2] = 0.5 * (0.5 + delta)**2

    if scalar:
        return i_c[0], w[0]
    return i_c, w


def grid_to_particles(q_grid, x_grid, x_particles):
    """
    Gather (interpolate) a scalar grid quantity to particle locations
    using the TSC weighting function.

    Particles outside the grid are clamped to the first/last cell
    (transmissive boundary padding).
    """
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


def particles_to_grid(q_particles, weights_p, x_particles, x_grid):
    """
    Scatter (deposit) a particle quantity onto the grid using the TSC
    weighting function, with optional per-particle mass/volume weights.

    The result is the unnormalised weighted sum (adjoint of gather):
        q_j = Σ_p  W_j(x_p) · m_p · q_p

    Total deposited quantity is conserved: Σ_j q_j = Σ_p m_p · q_p.
    """
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
    eff_w     = w * weights_p[:, np.newaxis]     # (N_p, 3)
    q_grid    = np.zeros(N, dtype=float)

    for s in range(3):
        idx = np.clip(i_c + s - 1, 0, N-1)
        np.add.at(q_grid, idx, eff_w[:, s] * q_particles)
        #np.add.at(q_grid, i_c + s - 1, eff_w[:, s] * q_particles)

    return q_grid


def _fold_ghost_deposits(deposit, Nx, bc):
    """
    For periodic / shear_periodic BCs, fold momentum deposited into
    ghost cells back into the corresponding physical cells.

    Padded layout (2 ghosts each side, Nx physical cells):
        index:  0    1  | 2 … Nx+1 |  Nx+2  Nx+3
                gl0  gl1   physical    gr0    gr1

    Periodic wrapping maps:
        gl0 (pos x0-2dx)  → physical cell Nx   (second-to-last)
        gl1 (pos x0-dx )  → physical cell Nx+1 (last)
        gr0 (pos xN+dx )  → physical cell 2    (first)
        gr1 (pos xN+2dx)  → physical cell 3    (second)
    """
    if bc in ('periodic', 'shear_periodic'):
        # Left ghosts → right physical cells
        deposit[Nx  ] += deposit[0]
        deposit[Nx+1] += deposit[1]
        deposit[0]     = 0.0
        deposit[1]     = 0.0
        # Right ghosts → left physical cells
        deposit[2]    += deposit[Nx+2]
        deposit[3]    += deposit[Nx+3]
        deposit[Nx+2]  = 0.0
        deposit[Nx+3]  = 0.0
    return deposit


# ─────────────────────────────────────────────────────────
#  Shearing-box physics  (particle + gas source terms)
# ─────────────────────────────────────────────────────────

def shearing_box_force(x_p, vx_p, vy_p, vgx_p, vgy_p, params):
    """
    Coriolis and epicyclic acceleration in the shearing box (rotating frame).

    Equations of motion for perturbation velocities (Bai & Stone 2010, eq. 1-2):

        dvₓ/dt =  2Ω vᵧ
        dvᵧ/dt = -(2-q)Ω vₓ

    For a Keplerian disc (q = 3/2) this reduces to:

        dvₓ/dt =  2Ω vᵧ
        dvᵧ/dt = -½Ω vₓ

    Parameters
    ----------
    x_p    : ndarray (N_p,)   particle positions       (unused; signature req.)
    vx_p   : ndarray (N_p,)   current particle vₓ
    vy_p   : ndarray (N_p,)   current particle vᵧ
    vgx_p  : ndarray (N_p,)   gas vₓ at particle pos.  (unused; signature req.)
    vgy_p  : ndarray (N_p,)   gas vᵧ at particle pos.  (unused; signature req.)
    params : ShearBox          namedtuple with fields Omega and q

    Returns
    -------
    fx, fy : ndarray (N_p,), ndarray (N_p,)
        Acceleration components due to Coriolis + epicyclic terms.
    """
    sb = params
    fx =  2.0 * sb.Omega * vy_p
    fy = -(2.0 - sb.q) * sb.Omega * vx_p
    return fx, fy


def gas_shearing_box_source(U, params):
    """
    Volumetric source terms for the gas in the shearing box (rotating frame).

    Adds the Coriolis + epicyclic accelerations and the background radial
    pressure-gradient force to the gas momentum equations:

        d(ρuₓ)/dt += +2Ω ρuᵧ  +  2η Ω² ρ
        d(ρuᵧ)/dt += -(2-q)Ω ρuₓ

    Parameters
    ----------
    U      : ndarray (4, N+4)   padded conserved state at current stage
    params : tuple (ShearBox, eta)
        ShearBox  namedtuple with .Omega and .q
        eta       float, dimensionless radial pressure gradient

    Returns
    -------
    S : ndarray (4, N+4), non-zero only in rows 1-2, physical cells (2:-2)
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
#  Particle integrator
# ─────────────────────────────────────────────────────────

def particles_step(particles, U_half, x_grid, dt,
                   bc='outflow', integrator="EM",
                   forces=[], force_params=[],
                   shear_box=None, analytic_x=False):
    """
    Advance all particle species by dt and deposit momentum back onto
    the grid (Newton's 3rd law).

    The drag update uses the exact exponential integrator for Stokes drag.
    External forces (e.g. Coriolis) are supplied via forces=[]/force_params[]
    and evaluated at the intermediate half-drag state for 2nd-order accuracy.

    Parameters
    ----------
    particles   : list of Particle_type
    U_half      : ndarray (4, Nx+4)   conserved state at the half-step
    x_grid      : ndarray (Nx,)       physical cell-centre positions
    dt          : float               time step
    bc          : str                 'outflow' | 'periodic' | 'shear_periodic'
    integrator : str
        'SSA' — Staggered Semi-Analytic method (Fung & Muley 2019).
    forces      : list of callables   external force functions
    force_params: list                parameter objects for each force
    shear_box   : ShearBox or None    used only for shear_periodic BC kick

    Returns
    -------
    particles     : updated list
    mom_x_deposit : ndarray (Nx+4,)   Δ(ρuₓ) deposited per cell
    mom_y_deposit : ndarray (Nx+4,)   Δ(ρuᵧ) deposited per cell
    """
    dx     = x_grid[1] - x_grid[0]
    x_grid = np.copy(x_grid)
    Nx     = len(x_grid)           # physical cell count — save BEFORE padding
    nghost = 2

    x_phys_min = x_grid[0] - dx/2
    x_phys_max = x_grid[-1] + dx/2
    L_domain   = x_phys_max - x_phys_min

    x_grid = np.concatenate([
        [x_grid[0] - 2*dx, x_grid[0] - dx],
        x_grid,
        [x_grid[-1] + dx,  x_grid[-1] + 2*dx],
    ])

    mom_x_deposit = np.zeros_like(x_grid)
    mom_y_deposit = np.zeros_like(x_grid)

    for p_type in particles:

        # Save initial state for analytic_x position update
        xn  = p_type.xs.copy()
        vxn = p_type.vxs.copy()

        # ── half-step positions (wrapped for periodic BCs) ────────────
        x_p_halfs = p_type.xs + p_type.vxs * dt * 0.5
        if bc in ('periodic', 'shear_periodic'):
            x_p_halfs = (x_phys_min
                         + (x_p_halfs - x_phys_min) % L_domain)

        # ── gas velocity at particle positions ────────────────────────
        vgx_halfs = U_half[1] / U_half[0]
        vgy_halfs = U_half[2] / U_half[0]

        vgx_at_p = grid_to_particles(vgx_halfs, x_grid, x_p_halfs)
        vgy_at_p = grid_to_particles(vgy_halfs, x_grid, x_p_halfs)

        match integrator:
            case "SSA":
                # Staggered Semi-Analytic method — Fung & Muley (2019).
                f1x = np.zeros_like(p_type.vxs)
                f1y = np.zeros_like(p_type.vys)
                for i, force in enumerate(forces):
                    dfx, dfy = force(
                        x_p_halfs,
                        p_type.vxs, p_type.vys,
                        vgx_at_p,  vgy_at_p,
                        force_params[i],
                    )
                    f1x += dfx
                    f1y += dfy

                exp_half = 1.0 - np.exp(-dt / (2.0 * p_type.tstop))
                v1x = p_type.vxs + (f1x*p_type.tstop + vgx_at_p - p_type.vxs) * exp_half
                v1y = p_type.vys + (f1y*p_type.tstop + vgy_at_p - p_type.vys) * exp_half

                f2x = np.zeros_like(p_type.vxs)
                f2y = np.zeros_like(p_type.vys)
                for i, force in enumerate(forces):
                    dfx, dfy = force(
                        x_p_halfs, v1x, v1y,
                        vgx_at_p, vgy_at_p,
                        force_params[i],
                    )
                    f2x += dfx
                    f2y += dfy

                exp_full = 1.0 - np.exp(-dt / p_type.tstop)
                vxf = p_type.vxs + (f2x*p_type.tstop + vgx_at_p - p_type.vxs) * exp_full
                vyf = p_type.vys + (f2y*p_type.tstop + vgy_at_p - p_type.vys) * exp_full

                delta_px = p_type.mass * (vxf - p_type.vxs - f2x*dt)
                delta_py = p_type.mass * (vyf - p_type.vys - f2y*dt)

                # Force for analytic position update
                fx_pos = f2x
                fy_pos = f2y

            case "EM":
                # ──────────────────────────────────────────────────────
                # Exponential Midpoint method (Mignone et al. 2019,
                # §3.2, eq. 32–34).
                #
                # Kick equation (implicit in v_mid):
                #   v^{n+1} = E·v^n + h₁·G(x_{1/2}, v_mid)
                #
                # where  v_mid = (v^n + v^{n+1})/2
                #        E     = exp(−Δt/τs)
                #        h₁    = τs·(1 − E)
                #        G     = f(x,v) + v_g/τs
                #
                # The exponential captures the stiff drag exactly;
                # G is evaluated at the TRUE midpoint velocity,
                # giving second-order accuracy for the coupling.
                #
                # Three sub-paths depending on the force structure:
                #   1. No forces:       explicit (G independent of v)
                #   2. Shearing box:    analytic 2×2 solve (G linear in v)
                #   3. General f(x,v):  fixed-point, Newton fallback
                # ──────────────────────────────────────────────────────

                E_decay = np.exp(-dt / p_type.tstop)
                h1      = p_type.tstop * (1.0 - E_decay)
                one_mE  = 1.0 - E_decay            # = h1 / τs

                vxn = p_type.vxs
                vyn = p_type.vys

                if not forces:
                    # ── Case 1: Pure drag — explicit ──────────────
                    # G = v_g/τs, no v dependence.
                    # v^{n+1} = E·v^n + (1−E)·v_g
                    vxf = E_decay * vxn + one_mE * vgx_at_p
                    vyf = E_decay * vyn + one_mE * vgy_at_p
                    f_mid_x = np.zeros_like(vxn)
                    f_mid_y = np.zeros_like(vyn)

                elif (len(forces) == 1
                      and forces[0] is shearing_box_force):
                    # ── Case 2: Shearing box — analytic 2×2 ──────
                    # Forces: f_x = 2Ω·v_y,  f_y = −(2−q)Ω·v_x
                    #
                    # G_x(v_mid) = 2Ω·v_{y,mid} + v_{gx}/τs
                    # G_y(v_mid) = −(2−q)Ω·v_{x,mid} + v_{gy}/τs
                    #
                    # Substituting v_mid = (v^n + v^{n+1})/2 into
                    # the kick equation and collecting v^{n+1} terms
                    # on the left gives the 2×2 system:
                    #
                    #   [  1    −Ω₁ ] [vx^{n+1}]   [bx]
                    #   [  Ω₂    1  ] [vy^{n+1}] = [by]
                    #
                    # where Ω₁ = h₁·Ω,  Ω₂ = h₁·(2−q)·Ω/2
                    #
                    # (Mignone et al. 2019, Appendix B.1, eq. 86–90)

                    sb  = force_params[0]
                    Om1 = h1 * sb.Omega
                    Om2 = h1 * (2.0 - sb.q) * sb.Omega * 0.5

                    bx = (E_decay * vxn  +  Om1 * vyn
                          + one_mE * vgx_at_p)
                    by = (E_decay * vyn  -  Om2 * vxn
                          + one_mE * vgy_at_p)

                    det = 1.0 + Om1 * Om2
                    vxf = (bx + Om1 * by) / det
                    vyf = (by - Om2 * bx) / det

                    # Midpoint velocity and force for deposit
                    vmx = 0.5 * (vxn + vxf)
                    vmy = 0.5 * (vyn + vyf)
                    f_mid_x =  2.0 * sb.Omega * vmy
                    f_mid_y = -(2.0 - sb.q) * sb.Omega * vmx

                else:
                    # ── Case 3: General f(x,v) — fixed-point,
                    #    with Newton fallback if it diverges.
                    #
                    # Fixed-point iteration:
                    #   v^{n+1}_{k+1} = E·v^n
                    #                  + h₁·G(x_{1/2}, (v^n + v^{n+1}_k)/2)
                    #
                    # Cost: 1 force eval per iteration.
                    # Converges when h₁/2·ρ(∂f/∂v) < 1  (typical).
                    #
                    # Newton iteration (fallback):
                    #   F(v) = v − E·v^n − h₁·G(x, (v^n+v)/2) = 0
                    #   J_F  = I − (h₁/2)·∂G/∂v
                    #   v ← v − J_F⁻¹·F(v)
                    #
                    # Cost: 3 force evals per iteration (1 residual
                    #        + 2 finite-difference columns of ∂G/∂v).
                    # Converges quadratically — needed only when
                    # fixed-point diverges (large Δt with strong
                    # velocity-dependent forces).

                    def _eval_G(vx_eval, vy_eval):
                        """Sum of all forces at given velocity."""
                        fx_s = np.zeros_like(vxn)
                        fy_s = np.zeros_like(vyn)
                        for j, force_fn in enumerate(forces):
                            dfx, dfy = force_fn(
                                x_p_halfs, vx_eval, vy_eval,
                                vgx_at_p, vgy_at_p,
                                force_params[j])
                            fx_s += dfx
                            fy_s += dfy
                        return fx_s, fy_s

                    def _kick(fx, fy):
                        """Apply kick: E·v^n + h₁·(f + vg/τs)."""
                        return (E_decay * vxn + h1 * fx
                                + one_mE * vgx_at_p,
                                E_decay * vyn + h1 * fy
                                + one_mE * vgy_at_p)

                    # Initial guess: forces at v^n
                    fx0, fy0 = _eval_G(vxn, vyn)
                    vxf, vyf = _kick(fx0, fy0)

                    # ── Fixed-point iterations ────────────────────
                    _fp_max   = 5
                    _fp_conv  = False
                    _fp_prev_err = np.inf
                    for _fp_k in range(_fp_max):
                        vmx = 0.5 * (vxn + vxf)
                        vmy = 0.5 * (vyn + vyf)
                        fx_s, fy_s = _eval_G(vmx, vmy)
                        vxf_new, vyf_new = _kick(fx_s, fy_s)

                        err = (np.max(np.abs(vxf_new - vxf))
                               + np.max(np.abs(vyf_new - vyf)))
                        vxf, vyf = vxf_new, vyf_new

                        if err < 1e-13 * (
                                1.0 + np.max(np.abs(vxf))
                                    + np.max(np.abs(vyf))):
                            _fp_conv = True
                            break

                        # Detect divergence: error growing
                        if err > 2.0 * _fp_prev_err and _fp_k >= 1:
                            break
                        _fp_prev_err = err

                    # ── Newton fallback if fixed-point failed ─────
                    if not _fp_conv:
                        # Restart from SA1 guess
                        fx0, fy0 = _eval_G(vxn, vyn)
                        vxf, vyf = _kick(fx0, fy0)

                        _eps_fd = 1e-7  # finite-difference step
                        _nw_max = 6
                        for _nw_k in range(_nw_max):
                            # Residual  F(v) = v − kick(G(v_mid))
                            vmx = 0.5 * (vxn + vxf)
                            vmy = 0.5 * (vyn + vyf)
                            fx_s, fy_s = _eval_G(vmx, vmy)
                            kx, ky = _kick(fx_s, fy_s)
                            Fx = vxf - kx
                            Fy = vyf - ky

                            res = (np.max(np.abs(Fx))
                                   + np.max(np.abs(Fy)))
                            if res < 1e-13 * (
                                    1.0 + np.max(np.abs(vxf))
                                        + np.max(np.abs(vyf))):
                                break

                            # Jacobian J_F by forward finite diffs.
                            # Column 1: perturb vx
                            sc_x = _eps_fd * (1.0 + np.abs(vxf))
                            fx_p, fy_p = _eval_G(
                                vmx + 0.5 * sc_x, vmy)
                            kxp, kyp = _kick(fx_p, fy_p)
                            # J_F = I − ∂kick/∂v; ∂kick/∂v_x col:
                            dkx_dvx = (kxp - kx) / sc_x
                            dky_dvx = (kyp - ky) / sc_x

                            # Column 2: perturb vy
                            sc_y = _eps_fd * (1.0 + np.abs(vyf))
                            fx_p, fy_p = _eval_G(
                                vmx, vmy + 0.5 * sc_y)
                            kxp, kyp = _kick(fx_p, fy_p)
                            dkx_dvy = (kxp - kx) / sc_y
                            dky_dvy = (kyp - ky) / sc_y

                            # J_F = [[1−dkx_dvx, −dkx_dvy],
                            #        [−dky_dvx,  1−dky_dvy]]
                            J00 = 1.0 - dkx_dvx
                            J01 =     - dkx_dvy
                            J10 =     - dky_dvx
                            J11 = 1.0 - dky_dvy

                            det_J = J00 * J11 - J01 * J10
                            det_J = np.where(
                                np.abs(det_J) < 1e-30,
                                np.sign(det_J) * 1e-30 + 1e-30,
                                det_J)

                            # δv = −J_F⁻¹ · F
                            dvx = -(J11 * Fx - J01 * Fy) / det_J
                            dvy = -(J00 * Fy - J10 * Fx) / det_J

                            vxf = vxf + dvx
                            vyf = vyf + dvy

                        # Recompute midpoint forces at converged v
                        vmx = 0.5 * (vxn + vxf)
                        vmy = 0.5 * (vyn + vyf)
                        fx_s, fy_s = _eval_G(vmx, vmy)

                    f_mid_x = fx_s
                    f_mid_y = fy_s

                # ── Back-reaction deposit (all EM sub-cases) ─────
                # Isolate drag contribution: subtract external force
                delta_px = p_type.mass * (vxf - vxn - f_mid_x * dt)
                delta_py = p_type.mass * (vyf - vyn - f_mid_y * dt)

                # Force for analytic position update
                fx_pos = f_mid_x
                fy_pos = f_mid_y

            case _:
                raise ValueError(f"Unknown integrator {integrator!r}")

        p_type.vxs = vxf
        p_type.vys = vyf

        if analytic_x:
            # Analytic position update assuming constant force and gas
            # velocity over the step (Fung & Muley 2019, eq. 9).
            ts  = p_type.tstop
            emf = 1.0 - np.exp(-dt / ts)    # = (1 - e^{-τ})
            # Terminal velocity asymptote: v_term = f*ts + vg
            vtx = fx_pos * ts + vgx_at_p
            vty = fy_pos * ts + vgy_at_p
            p_type.xs = xn + vtx * dt + (vxn - vtx) * ts * emf
        else:
            # Standard leapfrog second drift
            p_type.xs = x_p_halfs + p_type.vxs * dt * 0.5

        # ── boundary handling for final positions (all integrators) ──
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

        # ── deposit ───────────────────────────────────────────────────
        mom_x_deposit += particles_to_grid(
            -delta_px, None, x_p_halfs, x_grid)
        mom_y_deposit += particles_to_grid(
            -delta_py, None, x_p_halfs, x_grid)

    # Fold ghost-cell deposits back into physical domain for periodic BCs
    mom_x_deposit = _fold_ghost_deposits(mom_x_deposit, Nx, bc)
    mom_y_deposit = _fold_ghost_deposits(mom_y_deposit, Nx, bc)

    return particles, mom_x_deposit, mom_y_deposit


# ─────────────────────────────────────────────────────────
#  Particle drag predictor  (fluid first-stage source term)
# ─────────────────────────────────────────────────────────

def _particle_drag_predictor(particles, x_grid, U, dt, bc='periodic'):
    """
    Predictor drag source term for the fluid's first SSPRK stage.

    Computes the estimated momentum transfer from particles to gas over dt
    by working at the particle level (Mignone et al. 2019, §3.1, eq. 18-19):

      1. Interpolate gas velocity to each particle position using TSC.
      2. Compute the predicted velocity change per particle:
             Δvₓ_p = (ugx_at_p − vₓ_p) · (1 − exp(−dt/tstop))
             Δvᵧ_p = (ugy_at_p − vᵧ_p) · (1 − exp(−dt/tstop))
      3. Deposit the per-particle momentum change back onto the grid
         using the same TSC shape function (adjoint of gather):
             Δ(ρuₓ)_j += Σ_p  m_p · Δvₓ_p · W_j(x_p)  /  dx
             Δ(ρuᵧ)_j += Σ_p  m_p · Δvᵧ_p · W_j(x_p)  /  dx

    The gas receives the NEGATIVE of the particle momentum change
    (Newton's 3rd law), so this function deposits -m_p·Δv onto the
    gas grid.

    Parameters
    ----------
    particles : list of Particle_type   particle species at time n
    x_grid    : ndarray (Nx,)           physical cell-centre positions
    U         : ndarray (4, Nx+4)       padded conserved fluid state at time n
    dt        : float                   time step

    Returns
    -------
    source : ndarray (4, Nx+4)
        Δ(ρu) array; non-zero only in rows 1 (ρuₓ) and 2 (ρuᵧ),
        physical cells (indices 2:-2) only.
    """
    dx = x_grid[1] - x_grid[0]
    Nx = len(x_grid)

    # Build padded x_grid to match the convention in particles_step so that
    # grid_to_particles can reach ghost-cell gas velocities near boundaries.
    x_grid_pad = np.concatenate([
        [x_grid[0] - 2*dx, x_grid[0] - dx],
        x_grid,
        [x_grid[-1] + dx,  x_grid[-1] + 2*dx],
    ])

    source = np.zeros_like(U)          # shape (4, Nx+4)

    # Padded gas velocity fields for interpolation (includes ghost cells)
    ugx_pad = U[1] / U[0]             # shape (Nx+4,)
    ugy_pad = U[2] / U[0]

    for p_type in particles:
        exp_full = 1.0 - np.exp(-dt / p_type.tstop)

        # ── 1. Interpolate gas velocity to each particle position ─────────
        ugx_at_p = grid_to_particles(ugx_pad, x_grid_pad, p_type.xs)
        ugy_at_p = grid_to_particles(ugy_pad, x_grid_pad, p_type.xs)

        # ── 2. Predicted velocity change per particle ─────────────────────
        delta_vx = (ugx_at_p - p_type.vxs) * exp_full
        delta_vy = (ugy_at_p - p_type.vys) * exp_full

        # ── 3. Deposit per-particle Δ(mv) onto padded grid ───────────────
        # Use the PADDED grid for deposition so TSC stencils near
        # boundaries deposit into ghost cells instead of being clipped.
        # Then fold ghost deposits back into physical cells (periodic BC).
        dep_x = particles_to_grid(
            -p_type.mass * delta_vx, None, p_type.xs, x_grid_pad)
        dep_y = particles_to_grid(
            -p_type.mass * delta_vy, None, p_type.xs, x_grid_pad)
        dep_x = _fold_ghost_deposits(dep_x, Nx, bc)
        dep_y = _fold_ghost_deposits(dep_y, Nx, bc)
        source[1] += dep_x / dx
        source[2] += dep_y / dx

    return source


# ─────────────────────────────────────────────────────────
#  General SSPRK time step
# ─────────────────────────────────────────────────────────

def ssprk_step(U, dx, dt, bc="transmissive", tableau="ssprk3",
               recons='WENOZpI', particles=None, x_grid=None,
               dim=1.0, shear_box=None, forces=[], force_params=[],
               gas_sources=[], gas_source_params=[],
               integrator='EM', analytic_x=False,
               cs_iso=None):
    """
    Advance the 1.5-D conserved state U (shape 4 × N+4) by one time step dt
    using the requested SSPRK method in Shu-Osher form.

    When particles are present, the Mignone predictor-corrector scheme is
    used (Mignone et al. 2018 §3.1.2, 2019 §3.1):

      1. L^n  = spatial_residual(U^n)
      2. pred = predictor drag estimate ≈ Δt·S^n_D
      3. U*   = U^n + Δt·L^n + pred
      4. U_half = (U^n + U*) / 2
      5. Push particles using U_half → actual deposit
      6. L*   = spatial_residual(U*)
      7. U^{n+1} = U^n + Δt/2·(L^n + L*) + deposit/dx

    Second-order accurate in time and space; momentum conserved to
    machine precision.

    Parameters
    ----------
    U           : ndarray (4, N+4)   conserved state with ghost cells
    dx          : float              cell width
    dt          : float              time step
    bc          : str                boundary condition type
    tableau     : str                SSPRK method name
    recons      : str                reconstruction scheme
    particles   : list or None       Particle_type instances
    x_grid      : ndarray (N,)       physical cell centres (required if particles)
    shear_box        : ShearBox or None   used only for shear_periodic BC kick
    forces           : list               particle force functions
    force_params     : list               params for each particle force
    gas_sources      : list               gas source functions for spatial_residual
    gas_source_params: list               params for each gas source
    integrator       : str                'SSA' | 'EM'
    analytic_x       : bool               use analytic position update
    cs_iso           : float or None       isothermal sound speed; if set,
                                           enforce p = ρ·cs² (no energy eqn)

    Returns
    -------
    U_new               if particles is None
    (U_new, particles)  otherwise
    """
    if particles is not None:
        tableau = "ssprk2"   # particle coupling is designed for ssprk2 only

    alpha, beta = _ssprk_tableau(tableau)
    s = alpha.shape[0] - 1

    U_n = U.copy()

    # ══════════════════════════════════════════════════════════════════════
    #  Particle coupling: Mignone predictor-corrector
    # ══════════════════════════════════════════════════════════════════════
    if particles is not None:

        # Enforce isothermal EOS on initial state
        if cs_iso is not None:
            enforce_isothermal(U_n, cs_iso)

        # ── 1. Spatial residual at time n ─────────────────────────────
        L_n = spatial_residual(U_n, dx, bc, recons,
                               gas_sources, gas_source_params)

        # ── 2. Predictor drag source ─────────────────────────────────
        pred = _particle_drag_predictor(particles, x_grid, U_n, dt, bc)

        # ── 3. Predictor step: U* = U^n + Δt·L^n + pred ─────────────
        U_star = U_n + dt * L_n + pred
        #if cs_iso is not None:
            #enforce_isothermal(U_star, cs_iso)

        # ── 4. Half-time state: U_half = (U^n + U*) / 2 ─────────────
        U_half = 0.5 * (U_n + U_star)
        #if cs_iso is not None:
            #enforce_isothermal(U_half, cs_iso)
        U_half = apply_bc(U_half, bc)

        # ── 5. Push particles using U_half ────────────────────────────
        particles, Mx_deposit, My_deposit = particles_step(
            particles, U_half, x_grid, dt,
            bc=bc, shear_box=shear_box,
            forces=forces, force_params=force_params,
            integrator=integrator, analytic_x=analytic_x,
        )

        # ── 6. Spatial residual at predicted state ────────────────────
        L_star = spatial_residual(U_star, dx, bc, recons,
                                  gas_sources, gas_source_params)

        # ── 7. Corrector: U^{n+1} = U^n + Δt/2·(L^n + L*) + deposit ─
        U_new = U_n + 0.5 * dt * (L_star + L_n)
        U_new[1, 2:-2] += Mx_deposit[2:-2] / dx
        U_new[2, 2:-2] += My_deposit[2:-2] / dx
        if cs_iso is not None:
            enforce_isothermal(U_new, cs_iso)
        U_new = apply_bc(U_new, bc)

        return U_new, particles

    # ══════════════════════════════════════════════════════════════════════
    #  Fluid only: general SSPRK stages loop (no particles)
    # ══════════════════════════════════════════════════════════════════════
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

    U_new = apply_bc(stages[s], bc)

    return U_new


# ─────────────────────────────────────────────────────────
#  Solver driver
# ─────────────────────────────────────────────────────────

def solve(
    rho_init, ux_init, p_init,
    x,
    t_end,
    uy_init      = None,
    dt           = 1.e-2,
    cfl          = 0.9,
    bc           = "transmissive",
    ssprk        = "ssprk3",
    verbose      = True,
    recons       = 'WENOZpI',
    particles    = None,
    shear_box    = None,
    forces           = [],
    force_params     = [],
    gas_sources      = [],
    gas_source_params= [],
    integrator       = 'EM',
    analytic_x       = False,
    cs_iso           = 'auto',
):
    """
    Run the 1.5-D Godunov solver from t=0 to t=t_end.

    Parameters
    ----------
    rho_init     : array-like (N,)    initial density
    ux_init      : array-like (N,)    initial radial   velocity
    p_init       : array-like (N,)    initial pressure
    x            : array-like (N,)    cell-centre positions (uniform)
    t_end        : float              end time
    uy_init      : array-like or None initial azimuthal velocity; zeros if None
    cfl          : float              Courant number
    bc           : str                boundary condition type
    ssprk        : str                SSPRK method
    verbose      : bool               print progress every 100 steps
    recons       : str                reconstruction scheme
    particles    : list or None       Particle_type instances
    shear_box    : ShearBox or None   used only for shear_periodic BC kick
    forces            : list          particle force functions
    force_params      : list          params for each particle force
    gas_sources       : list          gas source functions for spatial_residual
    gas_source_params : list          params for each gas source
    integrator        : str           'SSA' | 'EM'
    analytic_x        : bool          use analytic position update (eq. 9, Fung & Muley 2019)
    cs_iso            : float, 'auto', or None
        Isothermal sound speed.  When set, enforce p = ρ·cs² after
        every sub-step (no energy equation).
        'auto' (default): enable isothermal when particles are present,
        computing cs from the initial conditions as sqrt(p₀/ρ₀).
        None: adiabatic (solve the full energy equation).

    Returns
    -------
    rho, ux, uy, p, t                 if particles is None
    rho, ux, uy, p, t, particles      otherwise
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

    # Resolve isothermal sound speed
    if cs_iso == 'auto':
        if particles is not None:
            # Compute from initial conditions: cs = sqrt(p/ρ)
            cs_iso = np.sqrt(np.mean(p_init) / np.mean(rho_init))
        else:
            cs_iso = None   # adiabatic for pure fluid problems

    t    = 0.0
    step = 0

    while t < t_end:
        dt_cfl = compute_dt(U[:, 2:-2], dx, cfl * cfl_eff,
                            particles=particles)
        dt     = min(dt_cfl, t_end - t)

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
                forces=forces, force_params=force_params,
                gas_sources=gas_sources,
                gas_source_params=gas_source_params,
                integrator=integrator,
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
    """Classic Sod shock tube (Sod 1978)."""
    x   = np.linspace(0.0, 1.0, N)
    rho = np.where(x < 0.5, 1.0,   0.125)
    ux  = np.zeros(N)
    uy  = np.zeros(N)
    p   = np.where(x < 0.5, 1.0,   0.1)
    return x, rho, ux, uy, p, 0.2


def lax_problem(N=200):
    """Lax problem — strong shock test."""
    x   = np.linspace(0.0, 1.0, N)
    rho = np.where(x < 0.5, 0.445,  0.5)
    ux  = np.where(x < 0.5, 0.698,  0.0)
    uy  = np.zeros(N)
    p   = np.where(x < 0.5, 3.528,  0.571)
    return x, rho, ux, uy, p, 0.14


def shu_osher(N=400):
    """Shu-Osher — shock/entropy wave interaction."""
    x   = np.linspace(-5.0, 5.0, N)
    rho = np.where(x < -4.0, 3.857143, 1.0 + 0.2*np.sin(5.0*x))
    ux  = np.where(x < -4.0, 2.629369, 0.0)
    uy  = np.zeros(N)
    p   = np.where(x < -4.0, 10.3333,  1.0)
    return x, rho, ux, uy, p, 1.8


def two_blast_waves(N=400):
    """Woodward & Colella two interacting blast waves."""
    x   = np.linspace(0.0, 1.0, N)
    rho = np.ones(N)
    ux  = np.zeros(N)
    uy  = np.zeros(N)
    p   = np.where(x < 0.1, 1000.0, np.where(x > 0.9, 100.0, 0.01))
    return x, rho, ux, uy, p, 0.038


def particle_drag_single(N=100, vgx=0., vgy=0., mp=1.,
                         vpx=1., vpy=0., tstop=1.):
    """Single particle decelerating by Stokes drag in a uniform gas."""
    x         = np.linspace(0.0, 100.0, N)
    rho       = np.ones(N)
    ux        = np.full(N, vgx)
    uy        = np.full(N, vgy)
    p         = np.ones(N) * 1e-4
    particles = [Particle_type(
        np.array([10.0]),
        np.array([vpx]),
        np.array([vpy]),
        tstop, mp,
    )]
    return x, rho, ux, uy, p, 5.0, particles


def nsh_equilibrium(N=128, epsilon=1.0, St=0.1, eta=0.05, Omega=1.0, q=1.5,
                    N_particles=None, L=1.0, p0=1.0):
    """
    Initial conditions for the Nakagawa-Sekiya-Hayashi (NSH) equilibrium
    drift test (Nakagawa, Sekiya & Hayashi 1986).
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
    ux  = np.full(N, u_gx)
    uy  = np.full(N, u_gy)
    p   = np.full(N, p0)

    x_p   = np.linspace(0.0, L, N_particles, endpoint=False) + 0.5*L/N_particles
    m_p   = epsilon * L / N_particles
    vxs_p = np.full(N_particles, v_px)
    vys_p = np.full(N_particles, v_py)

    particles = [Particle_type(x_p, vxs_p, vys_p, tstop=t_stop, mass=m_p)]
    shear_box = ShearBox(Omega=Omega, q=q)
    t_end     = 10.0 * (2.0 * np.pi / Omega)

    return x, rho, ux, uy, p, t_end, particles, shear_box


def clumpy_particle_drift(N=128, N_particles=512, clump_fraction=0.1,
                           epsilon=1.0, tstop=0.1, vp=1.0, vg=0.0):
    """
    Clumpy particle distribution test for momentum conservation.
    """
    L  = 1.0
    x  = np.linspace(0.0, L, N, endpoint=False)
    dx = x[1] - x[0]

    rho = np.ones(N)
    ux  = np.full(N, vg)
    uy  = np.zeros(N)
    p   = np.ones(N) * 1e-2

    clump_centre = 0.25 * L
    clump_half   = 0.5 * clump_fraction * L
    x_lo = clump_centre - clump_half
    x_hi = clump_centre + clump_half

    x_p = np.linspace(x_lo, x_hi, N_particles, endpoint=False) \
          + 0.5 * (x_hi - x_lo) / N_particles

    m_p = epsilon * L / N_particles

    particles = [Particle_type(
        xs    = x_p,
        vxs   = np.full(N_particles, vp),
        vys   = np.zeros(N_particles),
        tstop = tstop,
        mass  = m_p,
    )]

    t_end = 5.0 * tstop

    return x, rho, ux, uy, p, t_end, particles


# ─────────────────────────────────────────────────────────
#  Exact Sod solution
# ─────────────────────────────────────────────────────────

def sod_exact(x, t, gamma=1.4):
    """
    Exact solution of the Sod shock tube problem.

    Returns (rho, ux, uy, p) where uy is identically zero.
    """
    from scipy.optimize import brentq

    rho_L, u_L, p_L = 1.0,   0.0, 1.0
    rho_R, u_R, p_R = 0.125, 0.0, 0.1
    x_0 = 0.5

    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    def f(p, p_k, rho_k, c_k):
        if p <= p_k:
            return 2.0*c_k/gm1 * ((p/p_k)**(gm1/(2.0*gamma)) - 1.0)
        else:
            A = 2.0 / (gp1*rho_k)
            B = gm1/gp1 * p_k
            return (p - p_k) * np.sqrt(A/(p + B))

    p_star = brentq(
        lambda p: f(p,p_L,rho_L,c_L) + f(p,p_R,rho_R,c_R) + (u_R - u_L),
        1e-10, 10*max(p_L, p_R), xtol=1e-12, rtol=1e-12)

    u_star = u_L - f(p_star, p_L, rho_L, c_L)

    if p_star <= p_L:
        rho_star_L = rho_L * (p_star/p_L)**(1.0/gamma)
        c_star_L   = c_L   * (p_star/p_L)**(gm1/(2.0*gamma))
    else:
        rho_star_L = rho_L * (p_star/p_L + gm1/gp1) / (gm1/gp1*p_star/p_L + 1.0)
        c_star_L   = None

    if p_star > p_R:
        rho_star_R = rho_R * (p_star/p_R + gm1/gp1) / (gm1/gp1*p_star/p_R + 1.0)
    else:
        rho_star_R = rho_R * (p_star/p_R)**(1.0/gamma)

    S_head    = u_L - c_L
    S_tail    = u_star - c_star_L
    S_shock   = u_R + c_R*np.sqrt(gp1/(2.0*gamma)*p_star/p_R + gm1/(2.0*gamma))
    S_contact = u_star

    xi      = (x - x_0) / t
    rho_out = np.empty_like(x)
    u_out   = np.empty_like(x)
    p_out   = np.empty_like(x)

    for i, xi_i in enumerate(xi):
        if xi_i <= S_head:
            rho_out[i], u_out[i], p_out[i] = rho_L, u_L, p_L
        elif xi_i <= S_tail:
            u_fan   = 2.0/gp1 * (c_L + gm1/2.0*u_L + xi_i)
            c_fan   = c_L - gm1/2.0 * (u_fan - u_L)
            rho_fan = rho_L * (c_fan/c_L)**(2.0/gm1)
            p_fan   = p_L   * (rho_fan/rho_L)**gamma
            rho_out[i], u_out[i], p_out[i] = rho_fan, u_fan, p_fan
        elif xi_i <= S_contact:
            rho_out[i], u_out[i], p_out[i] = rho_star_L, u_star, p_star
        elif xi_i <= S_shock:
            rho_out[i], u_out[i], p_out[i] = rho_star_R, u_star, p_star
        else:
            rho_out[i], u_out[i], p_out[i] = rho_R, u_R, p_R

    return rho_out, u_out, np.zeros_like(rho_out), p_out


# ─────────────────────────────────────────────────────────
#  Animation
# ─────────────────────────────────────────────────────────

def animate_euler(
    rho_init, ux_init, p_init,
    x,
    t_end,
    uy_init   = None,
    n_frames  = 100,
    variables = "primitive",
    cfl       = 0.9,
    bc        = "transmissive",
    ssprk     = "ssprk3",
    figsize   = (10, 9),
    exact_fn  = None,
    interval  = 40,
    recons    = 'WENOZpI',
):
    """
    Animate the 1.5-D Euler solution from t=0 to t=t_end.

    Four-panel figure: ρ, uₓ, uᵧ, p  (primitive)  or  ρ, ρuₓ, ρuᵧ, E  (conserved).
    """
    cfl_eff = {"ssprk1":1.0,"ssprk2":1.0,"ssprk3":1.0,
               "ssprk4":2.0,"ssprk5":1.508}.get(ssprk.lower(), 1.0)

    x        = np.asarray(x,        dtype=float)
    rho_init = np.asarray(rho_init, dtype=float)
    ux_init  = np.asarray(ux_init,  dtype=float)
    p_init   = np.asarray(p_init,   dtype=float)
    uy_init  = (np.zeros_like(rho_init)
                if uy_init is None
                else np.asarray(uy_init, dtype=float))
    dx = x[1] - x[0]

    def pad(arr):
        return np.concatenate([[arr[0], arr[0]], arr, [arr[-1], arr[-1]]])

    if variables == "primitive":
        labels  = [r"$\rho$  (density)",
                   r"$u_x$  (radial velocity)",
                   r"$u_y$  (azimuthal velocity)",
                   r"$p$  (pressure)"]
        colours = ["#2171b5", "#238b45", "#fd8d3c", "#cb181d"]
        def extract(U_):
            rho, ux, uy, p = cons_to_prim(U_[:, 2:-2])
            return rho, ux, uy, p
    elif variables == "conserved":
        labels  = [r"$\rho$  (density)",
                   r"$\rho u_x$  (x-momentum)",
                   r"$\rho u_y$  (y-momentum)",
                   r"$E$  (total energy)"]
        colours = ["#2171b5", "#238b45", "#fd8d3c", "#cb181d"]
        def extract(U_):
            return (U_[0,2:-2].copy(), U_[1,2:-2].copy(),
                    U_[2,2:-2].copy(), U_[3,2:-2].copy())
    else:
        raise ValueError("variables must be 'primitive' or 'conserved'")

    q_init = extract(prim_to_cons(pad(rho_init), pad(ux_init), pad(uy_init), pad(p_init)))

    rho_f, ux_f, uy_f, p_f, _ = solve(
        rho_init, ux_init, p_init, x, t_end,
        uy_init=uy_init, cfl=cfl, bc=bc, ssprk=ssprk,
        verbose=False, recons=recons,
    )
    q_end = (rho_f, ux_f, uy_f, p_f) if variables == "primitive" else \
            extract(prim_to_cons(pad(rho_f), pad(ux_f), pad(uy_f), pad(p_f)))

    def _lims(a, b, margin=0.08):
        lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
        span = hi - lo if hi > lo else 1.0
        return lo - margin*span, hi + margin*span

    ylims       = [_lims(qi, qf) for qi, qf in zip(q_init, q_end)]
    frame_times = np.linspace(0.0, t_end, n_frames + 1)

    U = prim_to_cons(pad(rho_init), pad(ux_init), pad(uy_init), pad(p_init))
    t = 0.0

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs  = gridspec.GridSpec(4, 1, hspace=0.50,
                            left=0.10, right=0.95, top=0.93, bottom=0.06)
    axes        = [fig.add_subplot(gs[i]) for i in range(4)]
    lines       = []
    exact_lines = []

    for ax, q, label, col, ylim in zip(axes, q_init, labels, colours, ylims):
        ln, = ax.plot(x, q, color=col, lw=1.5, solid_capstyle="round")
        lines.append(ln)
        if exact_fn is not None:
            ex, = ax.plot(x, q, color="0.65", lw=1.0, ls="--", zorder=0, label="exact")
            exact_lines.append(ex)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=8)
        ax.grid(True, lw=0.4, alpha=0.4)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    if exact_fn is not None:
        axes[0].legend(fontsize=8, loc="upper right", framealpha=0.6, edgecolor="none")

    title = fig.suptitle("t = 0.0000", fontsize=10, x=0.52)

    def update(frame_idx):
        nonlocal U, t
        t_target = frame_times[frame_idx]
        while t < t_target:
            dt_step = compute_dt(U[:, 2:-2], dx, cfl * cfl_eff)
            dt_step = min(dt_step, t_target - t)
            U  = ssprk_step(U, dx, dt_step, bc, ssprk, recons=recons)
            t += dt_step
        q1, q2, q3, q4 = extract(U)
        for ln, q in zip(lines, (q1, q2, q3, q4)):
            ln.set_ydata(q)
        if exact_fn is not None:
            erho, eux, euy, ep = exact_fn(x, max(t, 1.0e-12))
            if variables == "conserved":
                eE = ep/(GAMMA-1.) + 0.5*erho*(eux**2 + euy**2)
                exact_qs = (erho, erho*eux, erho*euy, eE)
            else:
                exact_qs = (erho, eux, euy, ep)
            for ex_ln, eq in zip(exact_lines, exact_qs):
                ex_ln.set_ydata(eq)
        title.set_text(f"t = {t:.4f}  (frame {frame_idx} / {n_frames})")
        return lines + exact_lines + [title]

    anim = FuncAnimation(fig, update, frames=n_frames+1, interval=interval, blit=True)
    plt.close(fig)
    return anim, HTML(anim.to_jshtml())


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

    fields  = [(rho, r"$\rho$"), (ux, r"$u_x$"), (uy, r"$u_y$"), (p, r"$p$")]
    colours = ["#2171b5", "#238b45", "#fd8d3c", "#cb181d"]

    for ax, (q, lbl), col in zip(axes, fields, colours):
        ax.plot(x, q, color=col, lw=1.5, label="numerical")
        ax.set_ylabel(lbl, fontsize=10)
        ax.grid(True, lw=0.4, alpha=0.4)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    if exact_fn is not None:
        erho, eux, euy, ep = exact_fn(x, t)
        for ax, eq in zip(axes, [erho, eux, euy, ep]):
            ax.plot(x, eq, "k--", lw=1.0, alpha=0.6, label="exact")
        axes[0].legend(fontsize=9)

    axes[-1].set_xlabel("x", fontsize=10)
    fig.suptitle(f"{problem_name}  |  {ssprk_name}  |  t = {t:.4f}", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{problem_name}_{ssprk_name}.png", dpi=150)
    print(f"Figure saved to {problem_name}_{ssprk_name}.png")
