# Steady-State Dynamics Optimization

## Steady-State Equations:

In general, the steady-state dynamics of the system is given by:
$$
\sum_j  \left[ A(\theta, \vec{x}) \right]_{ij} S^{*}_j + \sum_{mn} S^{*}_m \left[ B(\theta, \vec{x})\right]_{i,mn} S^{*}_n = 0
$$
having in mind the entries of the tensor $B(\theta)$ are almost all zero. The $\vec{x}$ is the vector of the inputs of the model.

In the vectorial notation, we have that:
$$
A(\theta, \vec{x}) \vec{S}^* + \vec{S}^* B(\theta, \vec{x}) \vec{S}^* = \vec{0} \newline
$$

$$
\vec{S}^T = [ F_v ~~ O_f ~~ O_{2f} ~~ S_v ~~ O_s ~~S^*_{v} ~~ O^*_v ~\dots ] \newline
\vec{x}^T = [T_w ~~ T_{nw} ~~ \phi_{ion} ~~ [O] ~~ p ~~ I ~~\dots  ]
$$

#### $A(\theta)$ -like reactions:

$$
B(g) + S_i \rightarrow S_j + C(g) \newline
S_i \rightarrow S_j + C(g)
$$

####  $B(\theta)$ -like reactions:

$$
S_i + S_j \rightarrow S_{i'} + S_{j'} + D(g)
$$

#### Observable - $\gamma$:

$$
\gamma \equiv \hat{O} = \hat{T}(\theta, \vec{x}) \vec{S}^* \newline
\hat{O} = \sum_j \hat{T}(\theta, \vec{x})_j S^*_j
$$

## Loss function:

$$
\mathcal{L}(\theta) = \sum_{X} (O - \hat{O})^2
$$

where $X$ corresponds to the experimental data: $X = \{ (\vec{x}_i, O_i) \}^N_{i}$



## Optimization Methods

Global Optimization Step:

​	Differential Evolution

​	Dual Annealing

Local Optimizer:

​	Nelder-Mead

​	Powell

Ensure convergence to the global minimum in consecutive independent trials

Sensitive Analysis

