import numpy as np
import scipy as sp
import sympy as sy
import torch
import copy
import time

import src.Simulator as sim_system
import src.Optimizer as opt
import src.SimGrad as sim_diff
from pathos.multiprocessing import ProcessPool



###* Input Space Discretization
def lhs_sampling(n_samples, n_dims):
    samples = np.zeros((n_samples, n_dims))
    for j in range(n_dims):
        perm = np.random.permutation(n_samples)
        samples[:, j] = (perm + np.random.rand(n_samples)) / n_samples
        
    return samples
    
def map_to_bounds(samples, lower_bounds, upper_bounds):
    return lower_bounds + (upper_bounds - lower_bounds) * samples
    
    
###* store the results in txt file
def store_results(samples, results, best_result, config):
        
    with open(config['output_file'], 'w') as f:
        f.write("Input Space Discretization:\n")
        f.write(f"Lower Bounds: {config['lower_bounds']}\n")
        f.write(f"Upper Bounds: {config['upper_bounds']}\n")
        f.write(f"Number of Samples: {config['n_samples']}\n")
        f.write("\n")
            
        f.write("Optimization Results:\n")
        ###* best result
        f.write("Best Result:\n")
        f.write(f"Message: {best_result.message}\n")
        f.write(f"Status: {best_result.status}\n")
        f.write(f"Success: {best_result.success}\n")
        f.write(f"Iterations: {best_result.nit}")
        f.write(f"Loss: {best_result.fun}\n")
        f.write(f"Parameters: {best_result.x}\n")
        f.write(f"Iterations: {best_result.nit}\n")
        f.write(f"Function Evaluations: {best_result.nfev}\n")
        f.write(f"Gradient Evaluations: {best_result.njev}\n")
        f.write("\n")
            
        for idx, result in enumerate(results):
            f.write(f"Init Sample {samples[idx]}:\n")
            f.write(f"Message: {result.message}\n")
            f.write(f"Status: {result.status}\n")
            f.write(f"Success: {result.success}\n")
            f.write(f"Iterations: {result.nit}\n")
            f.write(f"Loss: {result.fun}\n")
            f.write(f"Parameters: {result.x}\n")
            f.write(f"Iterations: {result.nit}\n")
            f.write(f"Function Evaluations: {result.nfev}\n")
            f.write(f"Gradient Evaluations: {result.njev}\n")
            f.write("\n")
    
    
###* Optimization
def loss_and_grads(params, opt_object, diff_object):
    loss_val, frac_solutions_arr, rates_arr, _, gammas_predicted_arr = opt_object.objective_function_diff(params)
    grad_val = diff_object.objective_function_grad(params, frac_solutions_arr, rates_arr, gammas_predicted_arr)
    # print("loss_val: ", loss_val, "params: ", params)
    return loss_val, grad_val.numpy()


def run_local_optimization(params_arr, opt_object, diff_object, config):
    lower_bounds = config['lower_bounds']
    upper_bounds = config['upper_bounds']
    
    res = sp.optimize.minimize(
        lambda params: loss_and_grads(params, opt_object, diff_object),
        x0=params_arr,
        jac=True,
        bounds=[(a_i, b_i) for a_i, b_i in zip(lower_bounds, upper_bounds)],
        method='L-BFGS-B')
        
    return res


def pipeline1(opt_object, diff_object, config, num_workers=4):
    
    ##* discretize the input space
    lower_bounds = config['lower_bounds']
    upper_bounds = config['upper_bounds']
    n_samples = config['n_samples']
    
    lhs_samples = lhs_sampling(n_samples, len(lower_bounds))
    samples = map_to_bounds(lhs_samples, lower_bounds, upper_bounds)
    
    results = []
    with ProcessPool(nodes=num_workers) as pool:
        results = pool.map(
            lambda params: run_local_optimization(params, opt_object, diff_object, config),
            samples
        )
        
    ###* choose the best one
    best_loss = np.inf
    best_result = None
    
    for result in results:
        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result
        
    ###* store the results in a file:
    store_results(results, best_result, config)
    
    return best_result, results