# PlasmaHO: Stochastic Hierarchical Data-Driven Optimization

This repository contains the Python framework and data associated with the research paper:

**"STOCHASTIC HIERARCHICAL DATA-DRIVEN OPTIMIZATION: APPLICATION TO PLASMA-SURFACE KINETICS"**

This framework is designed for modeling, optimizing, and propagating uncertainty in plasma-surface kinetics simulations.

## How to Run

To get started with the optimization pipeline, refer to the following notebook as a primary example:

* **Example Notebook:** [`study_opt_hierarchical/opt_hier_pipeline-1.ipynb`](study_opt_hierarchical/opt_hier_pipeline-1.ipynb)

This notebook demonstrates the core workflow:

1. Initializing the `Simulator` object.
2. Setting up the optimization pipeline.
3. Executing the stochastic hierarchical optimization.

### Modifying Reaction Parameters

To adapt the code to your specific plasma chemistry case, you must modify the reaction parameters. Ensure that the initial conditions and rate coefficients in the simulation setup match your specific experimental or theoretical data.

## Code Structure

The core logic of the framework is located in the `src/` directory. Below is a brief explanation of the key modules:

* **`src/Simulator.py`**: Contains the main `Simulator` class, which acts as the central engine for running plasma simulations.
* **`src/Optimizer.py`**: Defines the loss objective function. This function operates by sequentially running the necessary simulations for every experimental data point provided, aggregating the results to compute the final loss value for the optimizer.
* **`src/rates.py` & `SimulatorRates.py`**: Manages the calculation of reaction rates and coefficients.
* **`src/Uncertainty.py`**: Handles the propagation of uncertainty through the model parameters.
* **`src/SimData.py` & `src/SimParser.py`**: Utilities for parsing output data and managing simulation datasets.

## Reproducing Paper Results

The results presented in the paper were obtained by running the notebooks located in the various study folders.

### Hierarchical Algorithm Implementation
* **`study_opt_hierarchical/`**: **This folder contains the core implementation of the Stochastic Hierarchical Data-Driven Optimization algorithm presented in the paper.** It demonstrates how the optimization is broken down into hierarchical levels to handle complex parameter spaces efficiently.

### Additional Studies
You can explore these directories to reproduce other figures or analyses found in the publication:

* **`study_generalization/`**: Analysis of model generalization capabilities (train vs. test results).
* **`study_opt_local/`**: Comparisons with local optimization approaches (e.g., Levenberg-Marquardt, Powell).
* **`study_opt_model/`**: Bayesian optimization pipelines and model selection studies.
* **`study_validation_model/`**: Validation metrics and parity plots comparing model predictions against experimental data.


## Requirements

* Python 3.x
* NumPy
* SciPy
* Matplotlib
* h5py

---

*If you use this code in your research, please cite the paper mentioned above.*
