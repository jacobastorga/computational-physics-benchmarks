"""
PyTest Verification Suite: 2D Transient Heat Advection-Diffusion
Author: Jacob Andrew Astorga (jacobastorga)
"""

import pytest
import numpy as np
try:
    from .advection_diffusion_solver import AdvectionDiffusion2DSolver
except ImportError:
    from advection_diffusion_solver import AdvectionDiffusion2DSolver

def test_boundary_condition_adherence():
    """Verify that Dirichlet and Neumann boundary conditions are satisfied strictly."""
    solver = AdvectionDiffusion2DSolver(L=2.0, Pe=10.0, nx=41, ny=21)
    T = solver.solve_steady_state()

    # Inlet: T(0, y) == 1.0 (for 0 < y < 1)
    assert np.allclose(T[0, 1:-1], 1.0, atol=1e-5), "Inlet temperature must be 1.0"

    # Chilled walls: T(x, 0) == 0.0 and T(x, 1) == 0.0
    assert np.allclose(T[:, 0], 0.0, atol=1e-5), "Bottom wall must be isothermal 0.0"
    assert np.allclose(T[:, -1], 0.0, atol=1e-5), "Top wall must be isothermal 0.0"

    # Outlet Neumann zero gradient: (T[nx-1, j] - T[nx-2, j]) == 0
    outlet_diff = np.abs(T[-1, :] - T[-2, :])
    assert np.all(outlet_diff < 1e-4), "Outlet zero-gradient Neumann condition violated"

def test_maximum_principle_and_monotonicity():
    """Verify temperature field obeys the continuous maximum principle (0 <= T <= 1)."""
    solver = AdvectionDiffusion2DSolver(L=2.0, Pe=15.0, nx=31, ny=21)
    T = solver.solve_steady_state()

    assert np.min(T) >= -1e-6, f"Spurious undershoot detected: min(T) = {np.min(T)}"
    assert np.max(T) <= 1.0 + 1e-6, f"Spurious overshoot detected: max(T) = {np.max(T)}"

def test_thermal_energy_conservation_balance():
    """Verify first-law energy conservation: Inlet flux = Outlet flux + Wall conduction loss."""
    solver = AdvectionDiffusion2DSolver(L=2.0, Pe=20.0, nx=51, ny=25)
    T = solver.solve_steady_state()
    balance = solver.compute_energy_flux_balance(T)

    # Relative energy conservation error should be within upwind discretization tolerance (< 20%)
    rel_error = balance['net_residual'] / balance['inlet_flux']
    assert rel_error < 0.20, f"Thermal energy conservation violated: relative error = {rel_error:.4f}"

def test_peclet_number_scaling():
    """Verify that higher Peclet number increases convective thermal penetration downstream."""
    solver_low_pe = AdvectionDiffusion2DSolver(L=2.0, Pe=2.0, nx=31, ny=21)
    solver_high_pe = AdvectionDiffusion2DSolver(L=2.0, Pe=50.0, nx=31, ny=21)

    T_low = solver_low_pe.solve_steady_state()
    T_high = solver_high_pe.solve_steady_state()

    # Mid-channel centerline temperature halfway downstream (x = L/2, y = 0.5)
    mid_x_idx = 15
    mid_y_idx = 10

    # At high Pe, convection carries more inlet heat downstream before dissipating to walls
    assert T_high[mid_x_idx, mid_y_idx] > T_low[mid_x_idx, mid_y_idx], \
        "Higher Peclet number must preserve higher centerline core temperature"
