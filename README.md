# Plasma Surface Dynamics

## Introduction

This project addresses the modeling and analysis of plasma-surface kinetics, emphasizing the following key objectives:

- **Simulation** of plasma surface kinetics using the system of differential equations.
- **Optimization** of kinetic parameters based on experimental data.
- **Error analysis** via uncertainty propagation.

The theoretical foundations and detailed methodologies are based on works by:

- Pedro Viegas *et al.* 2024 *Plasma Sources Sci. Technol.* **33** 055003,
- V. Guerra, "Analytical Model of Heterogeneous Atomic Recombination on Silica," *IEEE Transactions on Plasma Science*, vol. 35, no. 5, pp. 1397–1403, Oct. 2007.
- José Afonso *et al* 2024 *J. Phys. D: Appl. Phys.* **57** 04LT01



## Theoretical formulation

#### Plasma Surface Kinetics (`SurfaceKineticsSimulator`  Class)

The physical system's technical details are presented in previous referenced papers.

The general formulation corresponds to:
$$
\frac{dy}{dt} = f(y, t;\theta), ~~y(t_0) = y_0
$$
where

- $y$: vector of species concentrations 
- $\theta$: parameter set governing kinetics
- $f(\cdot)$: non-linear function modeling interactions 

At steady state, the dynamics satisfy:
$$
A(\theta, \vec{x}) \vec{S}^* + \vec{S}^* B(\theta, \vec{x}) \vec{S}^* = \vec{0}
$$
with reactions categorized as:

- **Linear** reactions (matrix $A$):

$$
B(g) + S_i \rightarrow S_j + C(g) \newline
S_i \rightarrow S_j + C(g)
$$

- **Bimolecular** reactions (tensor $B$):

$$
S_i + S_j \rightarrow S_{i'} + S_{j'} + D(g)
$$

**Observable ($\gamma$)** is computed as:
$$
\gamma \equiv \hat{O} = \hat{T}(\theta, \vec{x}) \vec{S}^*
$$


**Numerical methods:**

- Initial solution approximation using implicit Radau integration method.
- Refinement via fixed-point iteration or Runge-Kutta methods.



#### Parameter Optimization (`Optimize` Class)

Optimization adjusts kinetic parameters $\theta$ to minimize discrepancies between the model predictions and experimental data ($D = \{(\vec{x}_i, y_i) \}_{i=0}^{N}$). The recombination probabilities ($\gamma$) form the output conditions:

##### Objective function formation:

$$
\min_{\theta}J(\theta) = \sum^N_{i=0} \left( \frac{y_{exp,i} - \hat{y}_i}{y_{exp,i}}\right)^2
$$

**Hybrid optimization strategy:**

1. Execute global optimization (e.g., Differential Evolution with local polishing) $N$ times.
2. Select the top $K$ solutions based on objective values.
3. Locally refine each candidate by perturbation and further optimization, repeating $N'$ times.
4. Choose the best-refined candidate with the minimal objective function value.

This hybrid method effectively balances exploration and exploitation, improving the likelihood of approaching a global optimum.



#### Uncertainty Propagation (`ErrorPropagation` class)

This module evaluates the uncertainty of model predictions resulting from experimental uncertainties in input parameters.

Assume a set of independent random variables:

$$
X = \{X_1, X_2, \dots, X_n \}
$$

with knowns pdfs $P(X_1), P(X_2), \dots, P(X_n)$. If we assume that the inputs are independent, the joint distribution is given by:

$$
P(X_1, X_2, \dots, X_n) = \prod_{i=1}^n P(X_i)
$$
The deterministic relationship between inputs and observable ($\gamma$) is defined as:

$$
\gamma = f(X_1, X_2, \dots, X_n)
$$

And so it defines the conditional probability $P(\gamma|X_1, \dots X_n)$, which in our deterministic case is given by:

$$
P(\gamma|X_1, \dots, X_n) = \delta (\gamma - f(X_1, X_2, \dots, X_n))
$$

The resulting PDF for $\gamma$ is calculated as:

$$
P(\gamma) = \int_{x_1}\dots \int_{x_n} \delta(\gamma - f(x_1, \dots, x_n)) P(x_1, \dots, x_n) dx_1 \dots dx_n
$$

This integral is numerically estimated using the **Monte Carlo stratified sampling method**.



### Project Structure

This project comprises:

- **Simulation Module**: `DMsimulator.py` containing `SurfaceKineticsSimulator` class.
- **Optimization Module**: `DMoptimize.py` containing `Optimize` class.
- **Error Analysis Module**: `DMerror.py` containing `ErrorPropagation` class.



### Additional Scripts and Examples:

- `main.py`: Demonstrates basic usage of simulation and optimization.
- `learnParametersPaper.ipynb`: Jupyter notebook illustrating parameter optimization.
- `error_class_exp.ipynb`: Notebook showing uncertainty propagation analysis.
- `readExcel.ipynb`: Utility script to convert Excel/TXT files into HDF5 input format and visualize outputs.

Data and results are managed and stored in structured HDF5 files.