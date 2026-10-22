"""
Task 1: 2D Transient Heat Advection-Diffusion in Channel Flow
Author: Jacob Andrew Astorga (jacobastorga)
Repository: computational-physics-benchmarks

Solves the non-dimensional 2D advection-diffusion PDE:
    dT/dt + u(y) * dT/dx = (1 / Pe) * (d^2T/dx^2 + d^2T/dy^2)

Domain:
    x in [0, L], y in [0, 1]
    Laminar Poiseuille flow: u(y) = 6 * y * (1 - y)

Boundary Conditions:
    - Inlet (x = 0): Dirichlet T(0, y, t) = T_inlet = 1.0 (for 0 < y < 1)
    - Walls (y = 0, y = 1): Homogeneous Dirichlet T(x, 0, t) = T(x, 1, t) = 0.0 (isothermal chilled walls)
    - Outlet (x = L): Neumann zero-gradient dT/dx(L, y, t) = 0.0
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Compatibility helper for NumPy 1.x and 2.x
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

class AdvectionDiffusion2DSolver:
    def __init__(self, L=2.0, Pe=10.0, nx=41, ny=21):
        self.L = float(L)
        self.Pe = float(Pe)
        self.nx = int(nx)
        self.ny = int(ny)

        self.dx = self.L / (self.nx - 1)
        self.dy = 1.0 / (self.ny - 1)

        self.x = np.linspace(0, self.L, self.nx)
        self.y = np.linspace(0, 1.0, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Parabolic velocity profile u(y) = 6*y*(1-y)
        self.u = 6.0 * self.Y * (1.0 - self.Y)

    def solve_steady_state(self):
        """
        Solves the steady-state 2D advection-diffusion boundary value problem:
            u(y) * dT/dx - (1 / Pe) * (d^2T/dx^2 + d^2T/dy^2) = 0
        using a 2nd-order central difference / upwind scheme and sparse matrix inversion.
        """
        N = self.nx * self.ny
        A = sparse.lil_matrix((N, N))
        b = np.zeros(N)

        diff_coeff = 1.0 / self.Pe

        for i in range(self.nx):
            for j in range(self.ny):
                idx = i * self.ny + j

                # Wall boundaries take precedence at corners
                if j == 0:
                    A[idx, idx] = 1.0
                    b[idx] = 0.0
                elif j == self.ny - 1:
                    A[idx, idx] = 1.0
                    b[idx] = 0.0

                # Inlet (x = 0) interior
                elif i == 0:
                    A[idx, idx] = 1.0
                    b[idx] = 1.0

                # Outlet (x = L) -> Neumann dT/dx = 0 (Backward difference)
                elif i == self.nx - 1:
                    A[idx, idx] = 1.0
                    A[idx, (i - 1) * self.ny + j] = -1.0
                    b[idx] = 0.0

                # Interior nodes
                else:
                    u_val = self.u[i, j]

                    idx_im1 = (i - 1) * self.ny + j
                    idx_ip1 = (i + 1) * self.ny + j
                    idx_jm1 = i * self.ny + (j - 1)
                    idx_jp1 = i * self.ny + (j + 1)

                    # Upwind advection for stability: u * (T_{i,j} - T_{i-1,j}) / dx
                    adv_diag = u_val / self.dx
                    adv_im1 = -u_val / self.dx

                    diff_diag = -2.0 * diff_coeff / (self.dx**2) - 2.0 * diff_coeff / (self.dy**2)
                    diff_x = diff_coeff / (self.dx**2)
                    diff_y = diff_coeff / (self.dy**2)

                    A[idx, idx] = adv_diag - diff_diag
                    A[idx, idx_im1] = adv_im1 - diff_x
                    A[idx, idx_ip1] = -diff_x
                    A[idx, idx_jm1] = -diff_y
                    A[idx, idx_jp1] = -diff_y
                    b[idx] = 0.0

        A = A.tocsr()
        T_flat = spsolve(A, b)
        T_field = T_flat.reshape((self.nx, self.ny))
        return T_field

    def compute_energy_flux_balance(self, T_field):
        """
        Computes inlet convective energy flux, outlet convective energy flux,
        and wall conductive heat loss to verify global thermal energy balance.
        """
        inlet_flux = _trapz(self.u[0, :] * T_field[0, :], self.y)
        outlet_flux = _trapz(self.u[-1, :] * T_field[-1, :], self.y)

        grad_y_bottom = (T_field[:, 1] - T_field[:, 0]) / self.dy
        grad_y_top = (T_field[:, -1] - T_field[:, -2]) / self.dy

        wall_loss = (1.0 / self.Pe) * (_trapz(grad_y_bottom, self.x) - _trapz(grad_y_top, self.x))

        return {
            'inlet_flux': float(inlet_flux),
            'outlet_flux': float(outlet_flux),
            'wall_loss': float(wall_loss),
            'net_residual': float(abs(inlet_flux - (outlet_flux + wall_loss)))
        }
