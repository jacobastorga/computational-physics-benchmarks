"""
Task 3: Nonlinear Post-Buckling Bifurcation of an Imperfect Column (Computational Mathematics & Nonlinear ODEs)
Author: Jacob Andrew Astorga (jacobastorga)
Repository: computational-physics-benchmarks

Solves the exact geometrically nonlinear Elastica two-point boundary value problem 
for an axially compressed slender elastic column with initial geometric imperfection:
    d^2theta/ds^2 + (P / EI) * sin(theta + theta_0(s)) = 0
subject to moment-free simply supported boundary conditions:
    dtheta/ds(0) = 0,  dtheta/ds(L) = 0

Lateral deflection is obtained via kinematic integration:
    w(s) = integral_0^s sin(theta(s') + theta_0(s')) ds'
"""

import numpy as np
from scipy.integrate import solve_bvp

# Compatibility helper for NumPy 1.x and 2.x
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

class NonlinearBeamBucklingSolver:
    def __init__(self, L=1.0, EI=100.0, alpha_0=0.02):
        """
        Parameters:
            L: Column length (m)
            EI: Bending stiffness (N*m^2)
            alpha_0: Initial geometric angular imperfection amplitude (radians)
        """
        self.L = float(L)
        self.EI = float(EI)
        self.alpha_0 = float(alpha_0)

        # Classical Euler critical buckling load
        self.P_cr = (np.pi**2 * self.EI) / (self.L**2)

    def theta0(self, s):
        """Initial imperfection angle distribution along arc length s."""
        return self.alpha_0 * np.cos(np.pi * s / self.L)

    def solve_bvp(self, P, n_points=101):
        """
        Solves the nonlinear two-point BVP using scipy.integrate.solve_bvp.
        State vector: y[0] = theta (angle), y[1] = dtheta/ds (curvature/moment).
        """
        s_mesh = np.linspace(0, self.L, n_points)
        k2 = P / self.EI

        def ode_system(s, y):
            th = y[0]
            th_prime = y[1]
            th0_val = self.theta0(s)
            d2th_ds2 = -k2 * np.sin(th + th0_val)
            return np.vstack((th_prime, d2th_ds2))

        def boundary_conditions(ya, yb):
            # Moment-free at pinned ends: dtheta/ds(0) = 0, dtheta/ds(L) = 0
            return np.array([ya[1], yb[1]])

        # Initial guess based on perturbation
        p_ratio = P / self.P_cr
        amp = self.alpha_0 / max(abs(1.0 - p_ratio), 0.15)
        y_guess = np.zeros((2, s_mesh.size))
        y_guess[0] = amp * np.cos(np.pi * s_mesh / self.L)
        y_guess[1] = -amp * (np.pi / self.L) * np.sin(np.pi * s_mesh / self.L)

        sol = solve_bvp(ode_system, boundary_conditions, s_mesh, y_guess, tol=1e-5, max_nodes=5000)

        if not sol.success:
            raise RuntimeError(f"Elastica BVP solver failed to converge: {sol.message}")

        theta_field = sol.sol(s_mesh)[0]
        total_theta = theta_field + self.theta0(s_mesh)

        # Kinematic integration of lateral deflection w(s) = integral_0^s sin(total_theta) ds
        w_field = np.zeros_like(s_mesh)
        for i in range(1, len(s_mesh)):
            w_field[i] = _trapz(np.sin(total_theta[:i+1]), s_mesh[:i+1])

        mid_idx = n_points // 2
        mid_w = float(w_field[mid_idx])

        return {
            'success': sol.success,
            's': s_mesh,
            'theta': theta_field,
            'total_theta': total_theta,
            'w': w_field,
            'midspan_deflection': mid_w,
            'P': float(P),
            'P_cr': float(self.P_cr),
            'P_ratio': float(p_ratio)
        }
