"""
lid_driven_cavity.py
====================
2D Lid-Driven Cavity Flow Solver
---------------------------------
Method  : Finite Volume Method (FVM), collocated grid
Coupling: SIMPLE algorithm (Semi-Implicit Method for Pressure-Linked Equations)
Scheme  : First-order upwind convection, central differencing for diffusion
Grid    : Uniform structured NxN, cell-centred collocated storage
Pressure: Rhie-Chow momentum interpolation to suppress checkerboard oscillations

--------------
Ghia, Ghia & Shin (1982) -- benchmark data for Re = 100
"""

import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class LidDrivenCavity:
    """
    Parameters
    ----------
    N        : number of cells in each direction (grid is N x N)
    Re       : Reynolds number
    alpha_u  : velocity under-relaxation factor  (typical 0.5-0.8)
    alpha_p  : pressure under-relaxation factor  (typical 0.2-0.4)
    max_iter : maximum SIMPLE outer iterations
    tol      : convergence tolerance (L2 norm of continuity residual and delta u/v)
    """

    def __init__(
        self,
        N: int         = 64,
        Re: float      = 100.0,
        alpha_u: float = 0.7,
        alpha_p: float = 0.3,
        max_iter: int  = 5000,
        tol: float     = 1e-5,
    ):
        self.N        = N
        self.Re       = Re
        self.alpha_u  = alpha_u
        self.alpha_p  = alpha_p
        self.max_iter = max_iter
        self.tol      = tol

        # Non-dimensional physical constants (rho=U_lid=L=1)
        self.rho   = 1.0
        self.mu    = 1.0 / Re
        self.U_lid = 1.0

        # Uniform grid
        self.dx = 1.0 / N
        self.dy = 1.0 / N

        # Cell-centre coordinate arrays
        self.xc = np.linspace(self.dx / 2, 1.0 - self.dx / 2, N)
        self.yc = np.linspace(self.dy / 2, 1.0 - self.dy / 2, N)

        # Primary flow fields (all cell-centred)
        self.u = np.zeros((N, N))
        self.v = np.zeros((N, N))
        self.p = np.zeros((N, N))

        # Momentum matrix diagonal entry (relaxed: a_P / alpha_u).
        # Stored for use in Rhie-Chow interpolation and pressure correction.
        self.aP_u = np.ones((N, N))
        self.aP_v = np.ones((N, N))

        # Convergence log: list of (res_cont, res_u, res_v) per iteration
        self.residuals: list = []

    # ------------------------------------------------------------------
    # MOMENTUM EQUATION ASSEMBLY 
    # ------------------------------------------------------------------

    def assemble_momentum(self, component: str):
        """
        Build the sparse coefficient matrix A and RHS vector b for the
        u- or v-momentum equation.

        Discretization:  first-order upwind convection + central diffusion.
        Pressure:  explicit source using the current self.p field.

        Returns
        -------
        A   : scipy CSR matrix  (n x n),  n = N^2
        b   : 1-D numpy array  (n,)
        aP  : (N, N) array -- the RELAXED diagonal  a_P / alpha_u
        """
        N  = self.N
        dx, dy    = self.dx, self.dy
        rho, mu   = self.rho, self.mu
        alpha_u   = self.alpha_u

        # Select previous velocity field for under-relaxation source
        phi_old = self.u if component == 'u' else self.v

        # ---- Face velocities for convective mass fluxes ---------------
        # Linear interpolation on internal faces; walls carry zero flux.
        u_fe = np.zeros((N, N))
        u_fe[:, :-1] = 0.5 * (self.u[:, :-1] + self.u[:, 1:])

        u_fw = np.zeros((N, N))
        u_fw[:, 1:]  = 0.5 * (self.u[:, :-1] + self.u[:, 1:])

        v_fn = np.zeros((N, N))
        v_fn[:-1, :] = 0.5 * (self.v[:-1, :] + self.v[1:, :])

        v_fs = np.zeros((N, N))
        v_fs[1:, :]  = 0.5 * (self.v[:-1, :] + self.v[1:, :])

        # Mass flux: F = rho * u_normal * face_area
        F_e = rho * u_fe * dy
        F_w = rho * u_fw * dy
        F_n = rho * v_fn * dx
        F_s = rho * v_fs * dx

        # ---- Diffusion conductances  D = mu * A_face / delta -----------
        # Boundary faces: half distance to wall -> D is doubled.
        D_e = np.full((N, N), mu * dy / dx);  D_e[:, -1] *= 2   # right wall
        D_w = np.full((N, N), mu * dy / dx);  D_w[:,  0] *= 2   # left  wall
        D_n = np.full((N, N), mu * dx / dy);  D_n[-1, :] *= 2   # top wall/lid
        D_s = np.full((N, N), mu * dx / dy);  D_s[ 0, :] *= 2   # bottom wall

        # ---- Upwind neighbour coefficients -----------------------------
        # a_E = D_e + max(-F_e, 0)   etc.
        # Zero at boundary faces (no neighbour on that side).
        a_E = D_e + np.maximum(-F_e, 0.0);  a_E[:, -1] = 0.0
        a_W = D_w + np.maximum( F_w, 0.0);  a_W[:,  0] = 0.0
        a_N = D_n + np.maximum(-F_n, 0.0);  a_N[-1, :] = 0.0
        a_S = D_s + np.maximum( F_s, 0.0);  a_S[ 0, :] = 0.0

        # ---- Diagonal coefficient (without relaxation) -----------------
        # Collects the "own cell" contribution from every face.
        a_P = (D_e + np.maximum( F_e, 0.0)
             + D_w + np.maximum(-F_w, 0.0)
             + D_n + np.maximum( F_n, 0.0)
             + D_s + np.maximum(-F_s, 0.0))

        # ---- Pressure-gradient source (explicit, linear face interp.) --
        # Neumann BC at walls: face pressure = cell-centre pressure.
        if component == 'u':
            p_fe = self.p.copy()
            p_fe[:, :-1] = 0.5 * (self.p[:, :-1] + self.p[:,  1:])
            p_fw = self.p.copy()
            p_fw[:,  1:] = 0.5 * (self.p[:, :-1] + self.p[:,  1:])
            b_pres = -(p_fe - p_fw) * dy
        else:  # 'v'
            p_fn = self.p.copy()
            p_fn[:-1, :] = 0.5 * (self.p[:-1, :] + self.p[ 1:, :])
            p_fs = self.p.copy()
            p_fs[ 1:, :] = 0.5 * (self.p[:-1, :] + self.p[ 1:, :])
            b_pres = -(p_fn - p_fs) * dx

        # ---- Dirichlet BC source (non-zero wall velocity) --------------
        b_bc = np.zeros((N, N))
        if component == 'u':
            b_bc[-1, :] = D_n[-1, :] * self.U_lid   # top lid: u = U_lid

        # Replace a_P -> a_P / alpha_u  (diagonal of matrix)
        b_relax  = (1.0 - alpha_u) / alpha_u * a_P * phi_old
        a_P_diag = a_P / alpha_u                     # larger diagonal: more stable

        b_total = (b_pres + b_bc + b_relax).ravel()
        aP      = a_P_diag                           # returned for Rhie-Chow

        # ---- Sparse matrix assembly (COO -> CSR) -----------------------
        idx   = np.arange(N * N)
        j_idx = idx // N
        i_idx = idx  % N

        rows = [idx];        cols = [idx];        data = [a_P_diag.ravel()]

        for mask, shift, coeff in [
            (i_idx < N-1,  1,  a_E),
            (i_idx >    0, -1, a_W),
            (j_idx < N-1,  N,  a_N),
            (j_idx >    0, -N, a_S),
        ]:
            m = mask
            rows.append(idx[m])
            cols.append((idx + shift)[m])
            data.append(-coeff.ravel()[m])

        A = coo_matrix(
            (np.concatenate(data),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(N * N, N * N),
        ).tocsr()

        return A, b_total, aP

    # ------------------------------------------------------------------
    # RHIE-CHOW FACE FLUXES  ->  continuity residual
    # ------------------------------------------------------------------

    def rhie_chow_divergence(self, u_star, v_star):
        """
        Compute net outward volumetric face-flux per cell using Rhie-Chow momentum interpolation.

        Returns
        -------
        div : (N, N) array -- net outward volumetric flux per cell [m^2/s]
        """
        N  = self.N
        dx, dy = self.dx, self.dy
        p  = self.p

        # East faces (N x N-1 internal faces)
        u_e_lin = 0.5 * (u_star[:, :-1] + u_star[:, 1:])
        d_e     = 0.5 * (dy / self.aP_u[:, :-1] + dy / self.aP_u[:, 1:])
        u_e_rc  = u_e_lin - d_e * (p[:, 1:] - p[:, :-1])
        F_x     = u_e_rc * dy

        # North faces (N-1 x N internal faces)
        v_n_lin = 0.5 * (v_star[:-1, :] + v_star[1:, :])
        d_n     = 0.5 * (dx / self.aP_v[:-1, :] + dx / self.aP_v[1:, :])
        v_n_rc  = v_n_lin - d_n * (p[1:, :] - p[:-1, :])
        F_y     = v_n_rc * dx

        # Divergence: net outward flux per cell
        # F_x[j,i] leaves cell (j,i) eastward and enters cell (j,i+1) westward
        div = np.zeros((N, N))
        div[:, :-1] += F_x
        div[:,  1:] -= F_x
        div[:-1, :] += F_y
        div[ 1:, :] -= F_y

        return div

    # ------------------------------------------------------------------
    # PRESSURE-CORRECTION EQUATION
    # ------------------------------------------------------------------

    def assemble_pressure_correction(self):
        """
        Build the pressure-correction matrix A_p.

        BCs: Neumann at all walls (boundary face coefficients = 0).

        Returns
        -------
        A_p : scipy CSR matrix (n x n)
        """
        N  = self.N
        dx, dy = self.dx, self.dy

        # Pressure-response coefficients at internal faces
        d_x = 0.5 * (dy / self.aP_u[:, :-1] + dy / self.aP_u[:,  1:])  # (N, N-1)
        d_y = 0.5 * (dx / self.aP_v[:-1, :] + dx / self.aP_v[ 1:, :])  # (N-1, N)

        # Off-diagonal coefficients (Neumann -> 0 at boundary faces)
        a_E = np.zeros((N, N));  a_E[:, :-1] = d_x * dy
        a_W = np.zeros((N, N));  a_W[:,  1:] = d_x * dy
        a_N = np.zeros((N, N));  a_N[:-1, :] = d_y * dx
        a_S = np.zeros((N, N));  a_S[ 1:, :] = d_y * dx

        a_P = -(a_E + a_W + a_N + a_S)    # negative definite

        # COO assembly
        idx   = np.arange(N * N)
        j_idx = idx // N
        i_idx = idx  % N

        rows = [idx];  cols = [idx];  data = [a_P.ravel()]

        for mask, shift, coeff in [
            (i_idx < N-1,  1,  a_E),
            (i_idx >    0, -1, a_W),
            (j_idx < N-1,  N,  a_N),
            (j_idx >    0, -N, a_S),
        ]:
            m = mask
            rows.append(idx[m])
            cols.append((idx + shift)[m])
            data.append(coeff.ravel()[m])

        A = coo_matrix(
            (np.concatenate(data),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(N * N, N * N),
        ).tolil()

        # Fix reference pressure: p'[0,0] = 0  (removes null space)
        A[0, :] = 0.0
        A[0, 0] = 1.0

        return A.tocsr()

    # ------------------------------------------------------------------
    # CELL-CENTRE VELOCITY CORRECTION
    # ------------------------------------------------------------------

    def _correct_velocities(self, u_star, v_star, p_prime):
        """
        Correct cell-centre velocities using the pressure correction p'.

        Neumann at walls: p'_ghost = p'_interior  (dp'/dn = 0).

        Returns
        -------
        u_new, v_new : (N, N) corrected velocity arrays
        """
        dy = self.dy
        dx = self.dx

        # East/West neighbours of p' with Neumann at boundaries
        pp_E = np.hstack([p_prime[:,  1:], p_prime[:, [-1]]])
        pp_W = np.hstack([p_prime[:, [ 0]], p_prime[:, :-1]])
        u_new = u_star - (dy / self.aP_u) * 0.5 * (pp_E - pp_W)

        # North/South neighbours of p' with Neumann at boundaries
        pp_N = np.vstack([p_prime[ 1:, :], p_prime[[-1], :]])
        pp_S = np.vstack([p_prime[[ 0], :], p_prime[:-1, :]])
        v_new = v_star - (dx / self.aP_v) * 0.5 * (pp_N - pp_S)

        return u_new, v_new

    # ------------------------------------------------------------------
    # MAIN SIMPLE LOOP
    # ------------------------------------------------------------------

    def run(self):
        """
        Execute the SIMPLE iteration loop.

        Per-iteration steps
        -------------------
        1.  Assemble & solve u-momentum  ->  u*  (saves aP_u = a_P/alpha_u)
        2.  Assemble & solve v-momentum  ->  v*  (saves aP_v = a_P/alpha_u)
        3.  Compute Rhie-Chow face-flux divergence  div(u*)
        4.  Assemble & solve pressure-correction equation  ->  p'
        5.  Update pressure:    p  <-  p + alpha_p * p'
        6.  Correct velocities: u, v  using p' and relaxed aP
        7.  Compute residuals and check convergence

        """
        N   = self.N
        SEP = "-" * 65

        print(f"\n{SEP}")
        print(f"  Lid-Driven Cavity Flow  |  N={N}x{N}   Re={self.Re}")
        print(f"  alpha_u={self.alpha_u}   alpha_p={self.alpha_p}   "
              f"tol={self.tol}   max_iter={self.max_iter}")
        print(SEP)
        print(f"  {'Iter':>6}  {'Continuity':>12}  {'Delta-u':>12}  {'Delta-v':>12}")
        print(SEP)

        t0 = time.perf_counter()

        for it in range(1, self.max_iter + 1):

            u_old = self.u.copy()
            v_old = self.v.copy()

            # -- Steps 1 & 2: Momentum (with implicit relaxation) ------
            A_u, b_u, aP_u = self.assemble_momentum('u')
            u_star = spsolve(A_u, b_u).reshape(N, N)
            self.aP_u = aP_u

            A_v, b_v, aP_v = self.assemble_momentum('v')
            v_star = spsolve(A_v, b_v).reshape(N, N)
            self.aP_v = aP_v

            # -- Step 3: Rhie-Chow face-flux divergence ----------------
            div_star = self.rhie_chow_divergence(u_star, v_star)

            # -- Step 4: Pressure correction ---------------------------
            A_p = self.assemble_pressure_correction()
            b_p = div_star.ravel().copy()
            b_p[0] = 0.0                   # reference: p'[0,0] = 0
            p_prime = spsolve(A_p, b_p).reshape(N, N)

            # -- Step 5: Update pressure --------------------------------
            self.p += self.alpha_p * p_prime

            # -- Step 6: Correct velocities (no extra relaxation needed) -
            self.u, self.v = self._correct_velocities(u_star, v_star, p_prime)

            # -- Step 7: Convergence ------------------------------------
            res_cont = float(np.sqrt(np.mean(div_star ** 2)))
            res_u    = float(np.sqrt(np.mean((self.u - u_old) ** 2)))
            res_v    = float(np.sqrt(np.mean((self.v - v_old) ** 2)))
            self.residuals.append((res_cont, res_u, res_v))

            if it % 50 == 0 or it == 1:
                elapsed = time.perf_counter() - t0
                print(f"  {it:>6}  {res_cont:>12.3e}  {res_u:>12.3e}  "
                      f"{res_v:>12.3e}  [{elapsed:5.1f}s]")

            if res_cont < self.tol and res_u < self.tol and res_v < self.tol:
                elapsed = time.perf_counter() - t0
                print(f"\n  Converged at iteration {it}  (total: {elapsed:.1f} s)")
                break
        else:
            elapsed = time.perf_counter() - t0
            print(f"\n  Max iterations ({self.max_iter}) reached.  ({elapsed:.1f} s)")

    # ------------------------------------------------------------------
    # POST-PROCESSING
    # ------------------------------------------------------------------

    def plot_results(self, show: bool = True):
        N     = self.N
        i_mid = N // 2
        j_mid = N // 2
        
        # Reference data
        exp_y, exp_u = np.loadtxt("u_data.txt", unpack=True)
        exp_x, exp_v = np.loadtxt("v_data.txt", unpack=True)

        fig, ax = plt.subplots(figsize=(5, 5))

        # Velocity profiles
        ax.plot(self.u[:, i_mid], self.yc, 'b-',  lw=2, label=r'$u(x=0.5,\;y)$')
        ax.plot(self.v[j_mid, :], self.xc, 'r--', lw=2, label=r'$v(x,\;y=0.5)$')
        ax.plot(exp_u, exp_y, 'ro', label='Reference u data')
        ax.plot(exp_v, exp_x, 'go', label='Reference v data')
        ax.axhline(0.5, color='0.75', ls=':', lw=0.8)
        ax.axvline(0.0, color='0.75', ls=':', lw=0.8)
        ax.set_xlabel('Velocity', fontsize=11)
        ax.set_ylabel('Position', fontsize=11)
        ax.set_title('Centreline Velocity Profiles')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f'Exp_comparison_Re{int(self.Re)}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        print(f"  Saved -> {fname}")
        return fig

# ======================================================================
# Call solver
# ======================================================================

if __name__ == "__main__":
    solver = LidDrivenCavity(
        N        = 64,
        Re       = 100,
        alpha_u  = 0.7,
        alpha_p  = 0.3,
        max_iter = 5000,
        tol      = 1e-5,
    )
    solver.run()
    solver.plot_results()
