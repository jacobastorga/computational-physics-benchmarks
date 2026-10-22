# Computational Physics & Mathematics AI Benchmarks

[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)]()
[![Author](https://img.shields.io/badge/author-Jacob%20Andrew%20Astorga-orange.svg)](https://github.com/jacobastorga)
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

A research-grade benchmark suite designed to evaluate and stress-test frontier foundation models (LLMs and autonomous coding agents) on complex **computational physics, continuum mechanics, and applied mathematics** challenges.

Each benchmark task is structured according to frontier evaluation standards:
1. **Problem Formulation:** A rigorous physical problem defined by governing PDEs/ODEs, boundary value conditions, and constitutive material laws.
2. **Evaluation Subproblems:** Step-by-step mathematical reasoning challenges (discretization, tensor transformation, asymptotic scaling).
3. **Deterministic Reference Solver:** Ground-truth numerical solutions built in Python utilizing `numpy`, `scipy.integrate` (`solve_bvp`, `solve_ivp`), and `scipy.sparse`.
4. **Automated PyTest Test Harness:** Programmatic test assertions verifying boundary conditions (Dirichlet, Neumann, Robin), energy conservation balances, yield surfaces, and convergence stability.

---

## 🏛️ Benchmark Suite Overview

| Task ID | Physics Domain | Governing Equation | Numerical Architecture | Verification Assertions |
| :--- | :--- | :--- | :--- | :--- |
| **Task 1** | **Fluid Dynamics & Heat Transport** | 2D Advection-Diffusion: $u(y)\frac{\partial T}{\partial x} = \frac{1}{\text{Pe}}\nabla^2 T$ | Sparse FDM, upwind differencing, `scipy.sparse.linalg` | Boundary flux continuity, physical maximum principle ($0 \le T \le 1$), 1st-law thermal energy balance |
| **Task 2** | **Continuum Mechanics & Elasticity** | 2D Plane Stress Elasticity & Von Mises Yield | Kirsch closed-form polar tensor transformations + radial return mapping | Stress concentration factor $K_t = 3.0$, far-field decay, yield surface adherence |
| **Task 3** | **Nonlinear Mechanics & ODEs** | Exact Elastica BVP: $\theta''(s) + \frac{P}{EI}\sin(\theta + \theta_0) = 0$ | Two-point BVP collocation (`scipy.integrate.solve_bvp`) | Pinned boundary conditions $w(0)=w(L)=0$, post-buckling finite amplitude, bilateral symmetry |

---

## 📂 Repository Structure

```text
computational-physics-benchmarks/
├── README.md
├── requirements.txt
└── tasks/
    ├── task_01_advection_diffusion/
    │   ├── problem_statement.md
    │   ├── reference_solver.py
    │   └── test_advection_diffusion.py
    ├── task_02_continuum_elastoplasticity/
    │   ├── problem_statement.md
    │   ├── reference_solver.py
    │   └── test_elastoplasticity.py
    └── task_03_nonlinear_beam_buckling/
        ├── problem_statement.md
        ├── reference_solver.py
        └── test_beam_buckling.py
```

---

## 🚀 Installation & Running Automated Tests

```bash
# Clone the repository
git clone https://github.com/jacobastorga/computational-physics-benchmarks.git
cd computational-physics-benchmarks

# Install scientific dependencies
pip install -r requirements.txt

# Execute all benchmark verification suites
pytest tasks/ -v
```

---

## 🔬 Benchmark Task Summaries

### Task 1: 2D Heat Advection-Diffusion in Channel Flow
Solves non-dimensional convective-conductive heat transfer in a 2D channel with laminar Poiseuille flow $u(y) = 6y(1-y)$ and isothermal chilled walls ($T=0$). Tests whether AI models can correctly formulate the sparse matrix discretization, apply upwind differencing without numerical instability, and satisfy global first-law thermal conservation between inlet flux, outlet flux, and wall conductive dissipation.

### Task 2: Elastoplastic Stress Concentration Around a Cutout
Evaluates 2D stress tensor transformations $(\sigma_{rr}, \sigma_{\theta\theta}, \tau_{r\theta}) \to (\sigma_{xx}, \sigma_{yy}, \tau_{xy})$ under remote tensile loading. Benchmarks model ability to analytically verify the equatorial elastic stress concentration factor $K_t = 3.0$ and numerically integrate linear isotropic hardening under the Von Mises yield criterion without yield surface violations.

### Task 3: Nonlinear Post-Buckling Bifurcation of an Imperfect Column
Addresses the failure of linear Euler buckling theory ($P = P_{\text{cr}}$ singularity) by solving the geometrically exact nonlinear Elastica ODE BVP using `scipy.integrate.solve_bvp`. Tests model capabilities in state-space ODE formulation, adaptive collocation mesh refinement, and supercritical pitchfork bifurcation branch tracking.

---

## 👨‍💻 Author
**Jacob Andrew Astorga**  
*Computational Scientist & AI Evaluation Specialist*  
* LinkedIn: [linkedin.com/in/jacob-astorga-895405416](https://www.linkedin.com/in/jacob-astorga-895405416/)  
* GitHub: [@jacobastorga](https://github.com/jacobastorga)
