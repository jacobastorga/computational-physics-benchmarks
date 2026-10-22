# Task 1: 2D Heat Advection-Diffusion in Channel Flow (Fluid Dynamics & PDEs)

**Author:** Jacob Andrew Astorga (`jacobastorga`)  
**Domain:** Computational Fluid Dynamics (CFD), Transport Phenomena & Numerical PDEs  
**Difficulty:** Advanced Research / Frontier AI Benchmark  

---

## 1. Problem Formulation
Consider steady-state, laminar, incompressible flow through a two-dimensional rectangular channel of length $L$ and non-dimensional height $H = 1.0$, bounded by $x \in [0, L]$ and $y \in [0, 1]$.

The governing non-dimensional energy equation for convective and conductive heat transport is:
$$u(y) \frac{\partial T}{\partial x} = \frac{1}{\text{Pe}} \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right)$$

where:
* $\text{Pe} = \frac{U_{\text{max}} H}{\alpha}$ is the Péclet number representing the ratio of advective to diffusive transport rates.
* $u(y) = 6y(1-y)$ is the fully developed laminar Poiseuille velocity profile with maximum velocity $U_{\text{max}} = 1.5$ at the channel centerline $y = 0.5$.

### Boundary Conditions
1. **Inlet ($x = 0$):** Inflow enters at a uniform heated temperature:
   $$T(0, y) = 1.0 \quad \text{for } 0 < y < 1$$
2. **Channel Walls ($y = 0, y = 1$):** Chilled isothermal walls maintain zero temperature:
   $$T(x, 0) = 0.0, \quad T(x, 1) = 0.0$$
3. **Outlet ($x = L$):** Fully developed convective outflow (homogeneous Neumann):
   $$\frac{\partial T}{\partial x}(L, y) = 0.0$$

---

## 2. Evaluation Subproblems for AI Model
1. **Discretization Formulation:** Construct a sparse linear system $A \mathbf{T} = \mathbf{b}$ implementing upwind differencing for advection and second-order central differencing for diffusion.
2. **Boundary Condition Enforcement:** Demonstrate proper Dirichlet row-replacement and Neumann backward-difference stencils without violating matrix diagonal dominance.
3. **Physical Law Conservation Audit:** Compute the net thermal energy balance between inlet convective flux, outlet convective flux, and wall conduction loss:
   $$\Phi_{\text{in}} = \int_0^1 u(y) T(0, y) \, dy, \quad \Phi_{\text{out}} = \int_0^1 u(y) T(L, y) \, dy, \quad \Phi_{\text{loss}} = \frac{1}{\text{Pe}} \int_0^L \left( \left. \frac{\partial T}{\partial y} \right|_{y=0} - \left. \frac{\partial T}{\partial y} \right|_{y=1} \right) dx$$
   Verify that $\Phi_{\text{in}} \approx \Phi_{\text{out}} + \Phi_{\text{loss}}$.

---

## 3. Reference Implementation & Automated Verification
* Reference solver: `reference_solver.py` (`AdvectionDiffusion2DSolver`)
* PyTest verification suite: `test_advection_diffusion.py`
  * Dirichlet & Neumann boundary condition compliance (`test_boundary_condition_adherence`)
  * Physical maximum principle adherence ($0 \le T \le 1$) (`test_maximum_principle_and_monotonicity`)
  * Global thermal energy conservation balance (`test_thermal_energy_conservation_balance`)
  * Péclet number downstream penetration scaling (`test_peclet_number_scaling`)
