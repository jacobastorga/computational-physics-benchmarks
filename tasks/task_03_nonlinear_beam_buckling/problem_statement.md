# Task 3: Nonlinear Post-Buckling Bifurcation of an Imperfect Column (Computational Mathematics & ODEs)

**Author:** Jacob Andrew Astorga (`jacobastorga`)  
**Domain:** Nonlinear Mechanics, Elastic Stability & Two-Point Boundary Value ODEs  
**Difficulty:** Advanced Research / Frontier AI Benchmark  

---

## 1. Problem Formulation
Consider an elastic slender column of length $L = 1.0\text{ m}$ and flexural rigidity $EI = 100.0\text{ N}\cdot\text{m}^2$ simply supported at both ends, subjected to an axial compressive load $P$.

The column possesses an initial geometric angular imperfection:
$$\theta_0(s) = \alpha_0 \cos\left(\frac{\pi s}{L}\right)$$

According to classical linear Euler theory, the critical buckling load is:
$$P_{\text{cr}} = \frac{\pi^2 EI}{L^2}$$
However, linear theory erroneously predicts infinite lateral deflection at $P = P_{\text{cr}}$. To properly capture the post-critical pitchfork bifurcation and large-deflection equilibrium states, the geometrically exact nonlinear Elastica equation must be solved:
$$\frac{d^2\theta}{ds^2} + \frac{P}{EI} \sin(\theta(s) + \theta_0(s)) = 0$$

subject to moment-free simply supported boundary conditions:
$$\left. \frac{d\theta}{ds} \right|_{s=0} = 0, \quad \left. \frac{d\theta}{ds} \right|_{s=L} = 0$$

The lateral deflection profile $w(s)$ is obtained via exact kinematic integration along the arc length:
$$w(s) = \int_0^s \sin(\theta(s') + \theta_0(s')) \, ds'$$

---

## 2. Evaluation Subproblems for AI Model
1. **ODE State-Space Conversion:** Convert the second-order nonlinear Elastica ODE into a first-order state-space system $\mathbf{y}' = \mathbf{f}(s, \mathbf{y})$ suitable for adaptive collocation algorithms (`scipy.integrate.solve_bvp`).
2. **Subcritical Magnification:** Verify that for loads below Euler's load ($P < P_{\text{cr}}$), the numerical solution accurately reproduces the asymptotic linear magnification trend.
3. **Supercritical Pitchfork Bifurcation:** Solve the supercritical regime ($P > P_{\text{cr}}$), proving that geometric nonlinearity bounds the post-buckling amplitude without numerical divergence.
4. **Boundary & Symmetry Validation:** Demonstrate that the kinematic integration strictly preserves pinned boundary conditions ($w(0) = 0, w(L) = 0$) and bilateral symmetry ($|w(s)| = |w(L-s)|$).

---

## 3. Reference Implementation & Automated Verification
* Reference solver: `reference_solver.py` (`NonlinearBeamBucklingSolver`)
* PyTest verification suite: `test_beam_buckling.py`
  * Pinned boundary conditions recovery (`test_simply_supported_boundary_conditions`)
  * Subcritical load-deflection magnification (`test_subcritical_deflection_magnification`)
  * Supercritical finite amplitude stability proof (`test_post_buckling_finite_amplitude_stability`)
  * Bilateral symmetry verification (`test_symmetric_deflection_profile`)
