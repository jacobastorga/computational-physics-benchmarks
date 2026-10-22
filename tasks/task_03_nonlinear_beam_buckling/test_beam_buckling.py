"""
PyTest Verification Suite: Nonlinear Beam Buckling & Bifurcation
Author: Jacob Andrew Astorga (jacobastorga)
"""

import pytest
import numpy as np
try:
    from .beam_buckling_solver import NonlinearBeamBucklingSolver
except ImportError:
    from beam_buckling_solver import NonlinearBeamBucklingSolver

def test_simply_supported_boundary_conditions():
    """Verify pinned boundary conditions w(0) = 0 and w(L) = 0 are satisfied to high precision."""
    solver = NonlinearBeamBucklingSolver(L=1.0, EI=100.0, alpha_0=0.02)
    P_test = 0.8 * solver.P_cr
    res = solver.solve_bvp(P_test)

    assert np.isclose(res['w'][0], 0.0, atol=1e-6), "w(0) boundary condition violated"
    assert np.isclose(res['w'][-1], 0.0, atol=1e-5), "w(L) boundary condition violated"

def test_subcritical_deflection_magnification():
    """Verify that below P_cr, midspan deflection increases monotonically with axial load."""
    solver = NonlinearBeamBucklingSolver(L=1.0, EI=100.0, alpha_0=0.02)
    res_low = solver.solve_bvp(0.3 * solver.P_cr)
    res_mid = solver.solve_bvp(0.8 * solver.P_cr)

    assert abs(res_mid['midspan_deflection']) > abs(res_low['midspan_deflection']), \
        "Lateral midspan deflection magnitude must increase as compressive load approaches Euler load"

def test_post_buckling_finite_amplitude_stability():
    """Verify that supercritical load P > P_cr does not diverge to infinity, but stabilizes along finite post-buckling branch."""
    solver = NonlinearBeamBucklingSolver(L=1.0, EI=100.0, alpha_0=0.02)
    P_super = 1.2 * solver.P_cr
    res = solver.solve_bvp(P_super)

    # Deflection magnitude must be finite and physically bounded (finite elastica amplitude)
    assert res['success'] is True, "BVP solver must converge on supercritical branch"
    mid_amp = abs(res['midspan_deflection'])
    assert 0.01 < mid_amp < 0.5, \
        f"Post-buckling deflection amplitude should be stably bounded, got {mid_amp}"

def test_symmetric_deflection_profile():
    """Verify that the column deflection profile is symmetric: |w(s)| == |w(L - s)|."""
    solver = NonlinearBeamBucklingSolver(L=1.0, EI=100.0, alpha_0=0.02)
    res = solver.solve_bvp(0.9 * solver.P_cr, n_points=101)

    w = np.abs(res['w'])
    w_flipped = np.abs(np.flip(res['w']))

    assert np.allclose(w, w_flipped, atol=1e-4), "Deflection profile violated bilateral symmetry"
