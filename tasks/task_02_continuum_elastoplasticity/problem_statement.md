# Task 2: Elastoplastic Stress Concentration Around a Cutout (Continuum Mechanics & Elasticity)

**Author:** Jacob Andrew Astorga (`jacobastorga`)  
**Domain:** Solid Mechanics, Continuum Elasticity & Constitutive Plasticity  
**Difficulty:** Advanced Research / Frontier AI Benchmark  

---

## 1. Problem Formulation
Consider an elastic-plastic plate containing a circular cutout of radius $a = 1.0\text{ mm}$ subjected to a uniform remote uniaxial tensile stress $\sigma_\infty$ along the $y$-axis.

In the linear elastic regime, the closed-form stress tensor distribution is given by the classical Kirsch equations in polar coordinates $(r, \theta)$:
$$\sigma_{rr}(r, \theta) = \frac{\sigma_\infty}{2} \left( 1 - \frac{a^2}{r^2} \right) + \frac{\sigma_\infty}{2} \left( 1 - 4\frac{a^2}{r^2} + 3\frac{a^4}{r^4} \right) \cos(2\theta)$$
$$\sigma_{\theta\theta}(r, \theta) = \frac{\sigma_\infty}{2} \left( 1 + \frac{a^2}{r^2} \right) - \frac{\sigma_\infty}{2} \left( 1 + 3\frac{a^4}{r^4} \right) \cos(2\theta)$$
$$\sigma_{r\theta}(r, \theta) = -\frac{\sigma_\infty}{2} \left( 1 + 2\frac{a^2}{r^2} - 3\frac{a^4}{r^4} \right) \sin(2\theta)$$

### Plasticity & Yield Condition
When equivalent stress exceeds the uniaxial material yield strength $\sigma_{y0}$, plastic flow initiates according to the plane stress Von Mises yield function:
$$f(\boldsymbol{\sigma}, \varepsilon_p) = \sigma_{\text{vM}} - \sigma_y(\varepsilon_p) \le 0$$
$$\sigma_{\text{vM}} = \sqrt{\sigma_{xx}^2 - \sigma_{xx}\sigma_{yy} + \sigma_{yy}^2 + 3\tau_{xy}^2}$$
$$\sigma_y(\varepsilon_p) = \sigma_{y0} + H \varepsilon_p$$

where $H$ is the plastic tangent hardening modulus and $\varepsilon_p$ is accumulated effective plastic strain.

---

## 2. Evaluation Subproblems for AI Model
1. **Tensor Transformation:** Implement the coordinate transformation mapping polar Kirsch stresses $(\sigma_{rr}, \sigma_{\theta\theta}, \sigma_{r\theta})$ to Cartesian Cauchy stress components $(\sigma_{xx}, \sigma_{yy}, \tau_{xy})$.
2. **Stress Concentration Proof:** Analytically and numerically demonstrate that at the equatorial boundary ($r = a, \theta = \pi/2$), the theoretical elastic stress concentration factor satisfies:
   $$K_t = \frac{\sigma_{\theta\theta}(a, \pi/2)}{\sigma_\infty} = 3.0$$
3. **Elastoplastic Integration:** Implement a radial return-mapping integration scheme computing the localized plastic zone boundary $r_p$ and enforcing that the stress state remains strictly bounded on the hardening yield surface.

---

## 3. Reference Implementation & Automated Verification
* Reference solver: `reference_solver.py` (`ContinuumElastoplasticitySolver`)
* PyTest verification suite: `test_elastoplasticity.py`
  * Stress concentration factor exact recovery ($K_t = 3.0$) (`test_theoretical_stress_concentration_factor`)
  * Far-field asymptotic decay back to remote tension (`test_far_field_asymptotic_recovery`)
  * Yield surface non-violation under monotonic loading (`test_plastic_zone_confinement_and_yield_surface`)
  * Monotonic spatial decay of accumulated plastic strain (`test_monotonic_strain_decay`)
