"""
Task 2: Elastoplastic Stress Concentration Around a Cutout (Continuum Mechanics & Elasticity)
Author: Jacob Andrew Astorga (jacobastorga)
Repository: computational-physics-benchmarks

Evaluates 2D stress tensor field sigma_ij and elastoplastic progression
around a cutout in a plate subject to remote uniaxial tension sigma_inf.
Implements the Kirsch analytical elasticity equations paired with 
Von Mises yield criteria and isotropic linear strain hardening.
"""

import numpy as np

class ContinuumElastoplasticitySolver:
    def __init__(self, a=1.0, W=10.0, sigma_inf=100.0, sigma_y0=250.0, E=200000.0, H=10000.0):
        """
        Parameters:
            a: Cutout radius (mm)
            W: Half-width of plate (mm)
            sigma_inf: Remote tensile stress (MPa)
            sigma_y0: Initial yield strength (MPa)
            E: Young's Modulus (MPa)
            H: Plastic hardening modulus (MPa)
        """
        self.a = float(a)
        self.W = float(W)
        self.sigma_inf = float(sigma_inf)
        self.sigma_y0 = float(sigma_y0)
        self.E = float(E)
        self.H = float(H)

    def compute_elastic_stress_field(self, r_grid, theta_grid):
        """
        Computes 2D Kirsch stress field in polar and Cartesian coordinates.
        r >= a, theta in [0, 2*pi]
        """
        r = np.maximum(r_grid, self.a)
        th = theta_grid
        s_inf = self.sigma_inf
        a = self.a

        # Kirsch closed-form polar stresses
        factor = (a / r)**2
        sigma_rr = 0.5 * s_inf * (1.0 - factor) + 0.5 * s_inf * (1.0 - 4.0 * factor + 3.0 * factor**2) * np.cos(2.0 * th)
        sigma_tt = 0.5 * s_inf * (1.0 + factor) - 0.5 * s_inf * (1.0 + 3.0 * factor**2) * np.cos(2.0 * th)
        tau_rt = -0.5 * s_inf * (1.0 + 2.0 * factor - 3.0 * factor**2) * np.sin(2.0 * th)

        # Coordinate transformation to Cartesian (x, y)
        cos_t = np.cos(th)
        sin_t = np.sin(th)

        sigma_xx = sigma_rr * cos_t**2 + sigma_tt * sin_t**2 - 2.0 * tau_rt * sin_t * cos_t
        sigma_yy = sigma_rr * sin_t**2 + sigma_tt * cos_t**2 + 2.0 * tau_rt * sin_t * cos_t
        tau_xy = (sigma_rr - sigma_tt) * sin_t * cos_t + tau_rt * (cos_t**2 - sin_t**2)

        # Plane stress Von Mises equivalent stress
        sigma_vm = np.sqrt(sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3.0 * tau_xy**2)

        return {
            'sigma_rr': sigma_rr,
            'sigma_tt': sigma_tt,
            'tau_rt': tau_rt,
            'sigma_xx': sigma_xx,
            'sigma_yy': sigma_yy,
            'tau_xy': tau_xy,
            'sigma_vm': sigma_vm
        }

    def compute_stress_concentration_factor(self):
        """Returns peak elastic hoop stress at hole boundary (r = a, theta = pi/2) normalized by sigma_inf."""
        res = self.compute_elastic_stress_field(np.array([self.a]), np.array([np.pi / 2.0]))
        peak_hoop = float(res['sigma_tt'][0])
        return peak_hoop / self.sigma_inf

    def evaluate_elastoplastic_yield(self, r_array):
        """
        Evaluates yield criteria and plastic strain accumulation along the ligament (theta = pi/2)
        from r = a to r = W under linear isotropic hardening.
        """
        theta_ligament = np.full_like(r_array, np.pi / 2.0)
        elastic_res = self.compute_elastic_stress_field(r_array, theta_ligament)
        vm_stress = elastic_res['sigma_vm'].copy()

        plastic_strain = np.zeros_like(vm_stress)
        effective_stress = vm_stress.copy()

        # Check yield condition
        yield_mask = vm_stress > self.sigma_y0
        if np.any(yield_mask):
            # 1D radial return-mapping algorithm for isotropic hardening
            # delta_gamma = (sigma_trial - sigma_y0) / (E + H)
            delta_strain = (vm_stress[yield_mask] - self.sigma_y0) / (self.E + self.H)
            plastic_strain[yield_mask] = delta_strain
            # Updated yield surface: sigma_y = sigma_y0 + H * plastic_strain
            effective_stress[yield_mask] = self.sigma_y0 + self.H * delta_strain

        return {
            'r': r_array,
            'elastic_trial_vm': vm_stress,
            'effective_stress': effective_stress,
            'plastic_strain': plastic_strain,
            'plastic_zone_radius': float(r_array[yield_mask][-1]) if np.any(yield_mask) else 0.0
        }
