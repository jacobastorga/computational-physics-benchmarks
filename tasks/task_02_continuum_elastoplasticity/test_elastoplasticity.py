"""
PyTest Verification Suite: Elastoplastic Stress Concentration Around a Cutout
Author: Jacob Andrew Astorga (jacobastorga)
"""

import pytest
import numpy as np
try:
    from .elastoplasticity_solver import ContinuumElastoplasticitySolver
except ImportError:
    from elastoplasticity_solver import ContinuumElastoplasticitySolver

def test_theoretical_stress_concentration_factor():
    """Verify that linear elastic stress concentration Kt matches Kirsch analytical solution (Kt = 3.0)."""
    solver = ContinuumElastoplasticitySolver(a=1.0, W=10.0, sigma_inf=50.0, sigma_y0=300.0)
    kt = solver.compute_stress_concentration_factor()

    assert np.isclose(kt, 3.0, atol=1e-5), f"Theoretical Kt should equal 3.0, got {kt}"

def test_far_field_asymptotic_recovery():
    """Verify that stresses decay back to uniform remote tension sigma_inf as r -> infinity."""
    solver = ContinuumElastoplasticitySolver(a=1.0, W=50.0, sigma_inf=100.0)
    r_far = np.array([50.0])
    theta_zero = np.array([0.0])

    res = solver.compute_elastic_stress_field(r_far, theta_zero)

    # In far field (r >> a), sigma_xx -> sigma_inf, sigma_yy -> 0, tau_xy -> 0
    assert np.isclose(res['sigma_xx'][0], 100.0, rtol=0.01), "Far field sigma_xx must recover remote tension"
    assert np.isclose(res['sigma_yy'][0], 0.0, atol=2.0), "Far field transverse stress must vanish"
    assert np.isclose(res['tau_xy'][0], 0.0, atol=2.0), "Far field shear stress must vanish"

def test_plastic_zone_confinement_and_yield_surface():
    """Verify that when remote stress causes local yield (sigma_vm > sigma_y0), the stress is bounded by the hardening yield surface."""
    # Under sigma_inf = 100 MPa, elastic peak is 300 MPa > sigma_y0 = 220 MPa
    solver = ContinuumElastoplasticitySolver(a=1.0, W=10.0, sigma_inf=100.0, sigma_y0=220.0, E=200000.0, H=15000.0)
    r_ligament = np.linspace(1.0, 10.0, 200)

    ep_res = solver.evaluate_elastoplastic_yield(r_ligament)

    # Plastic zone must form at notch root
    assert ep_res['plastic_zone_radius'] > 1.0, "Plastic yield zone must develop at notch tip"

    # Effective stress must never exceed the updated yield surface: sigma_y0 + H * plastic_strain
    computed_yield_surface = solver.sigma_y0 + solver.H * ep_res['plastic_strain']
    assert np.all(ep_res['effective_stress'] <= computed_yield_surface + 1e-4), \
        "Effective stress exceeded the isotropic hardening yield surface!"

def test_monotonic_strain_decay():
    """Verify that plastic strain is strictly monotonically decreasing away from the notch root."""
    solver = ContinuumElastoplasticitySolver(a=1.0, W=10.0, sigma_inf=110.0, sigma_y0=200.0)
    r_ligament = np.linspace(1.0, 5.0, 50)
    ep_res = solver.evaluate_elastoplastic_yield(r_ligament)

    ep_diff = np.diff(ep_res['plastic_strain'])
    assert np.all(ep_diff <= 1e-8), "Plastic strain must decrease monotonically with radial distance from cutout"
