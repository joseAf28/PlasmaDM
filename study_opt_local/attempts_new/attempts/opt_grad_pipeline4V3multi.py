import sys, os
PATH = os.path.dirname(os.path.abspath(os.curdir))
if PATH not in sys.path:
    sys.path.insert(0, PATH)

import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from pathos.multiprocessing import ProcessPool
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from functools import partial
import scipy as sp
import numpy as np
import torch
import tqdm
import h5py

import src.Simulator as sim_system
import src.Optimizer as opt
import src.SimGrad as sim_diff

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


###! Create Simulator object and  Functions
reactions_file = "../reactions/reactionsSimpleV1.json"

const_dict = {
        "F0": 1.5e15,           # cm^-2
        "S0": 3e13,             # cm^-2
        
        "R": 0.00831442,        # kJ/mol*K
        "kBoltz": 1.380649e-23, # J/K
}

initial_state_dict = {'O_F': 0.1, 'O2_F':0.1 ,'O_S': 0.1, 'Vdb_S':0.1, 
                    'Odb_S': 0.1, 'CO_F': 0.1, 'CO2_F':0.1, 'CO_S': 0.1, 
                    'COdb_S': 0.0}

###* Functions for the data transformation
def compute_flux(const_dict, exp_dict, specie, molar_mass):
    den = exp_dict.get(specie, 0.0)
    v_th = np.sqrt((8.0 * const_dict['R'] * 1000 * exp_dict['Tnw'])/(molar_mass * np.pi))
    flux = 0.25 * v_th * den * 100
    return flux


def compute_remaining_flux(const_dict, exp_dict, molar_mass): 
    den = exp_dict['N'] - exp_dict['O'] - exp_dict['CO']
    v_th = np.sqrt((8.0 * const_dict['R'] * 1000 * exp_dict['Tnw'])/(molar_mass * np.pi))
    flux = 0.25 * v_th * den * 100
    return flux

####? EavgMB data extracted from the Booth et al. 2019 paper
p_data_exp = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.5]
EavgMB_data = [1.04, 0.91, 0.87, 0.83, 0.77, 0.5, 0.001]
interpolator = sp.interpolate.interp1d(p_data_exp, EavgMB_data, kind='linear', fill_value=0.001, bounds_error=False)


transformations_exp = {
    'Tw':       lambda const_dict, exp_dict: exp_dict['Tw'] + 273.15,
    'fluxO' :   lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict,'O', 0.016),
    'fluxO2' :  lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict,'O2', 0.032),
    'fluxO3' :  lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict,'O3', 0.048),
    'fluxC':    lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict, 'C', 0.012),
    'fluxCO':   lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict, 'CO', 0.028),
    'fluxCO2':  lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict, 'CO2', 0.048),
    'EavgMB':   lambda const_dict, exp_dict: interpolator(exp_dict['pressure']).item(),
    'Ion':      lambda const_dict, exp_dict: 1e14 * exp_dict["current"]
}

output_folder_path = "../Buffer_Data"
exp_data_file = "Experimental_data_CO_Jorge.hdf5"
exp_file = os.path.join(output_folder_path, exp_data_file)

sim = sim_system.Simulator(reactions_file, const_dict, exp_file, initial_state_dict, transformations_exp=transformations_exp)



###* Create optimization and diff objects

lower_bounds = np.array([1e-8, 1e-8, 0.0, 1e-8, 1e-8, 0.0, 1e-5, 1e-5, 1e-5, 1e-5])
upper_bounds = np.array([1e-1, 1e-1, 30.0, 1e-1, 1e-1, 30.0, 1.0, 1.0, 1.0, 1.0])


##! define parameters to optimize
def func_optimization(params, flag='numpy'):
    
    A_d, B_d, E_d, A_D, B_D, E_D, SF_1, SF_2, SF_3, SF_4 = params
    
    A_d = lower_bounds[0] + (upper_bounds[0] - lower_bounds[0]) * A_d
    B_d = lower_bounds[1] + (upper_bounds[1] - lower_bounds[1]) * B_d
    E_d = lower_bounds[2] + (upper_bounds[2] - lower_bounds[2]) * E_d
    A_D = lower_bounds[3] + (upper_bounds[3] - lower_bounds[3]) * A_D
    B_D = lower_bounds[4] + (upper_bounds[4] - lower_bounds[4]) * B_D
    E_D = lower_bounds[5] + (upper_bounds[5] - lower_bounds[5]) * E_D
    SF_1 = lower_bounds[6] + (upper_bounds[6] - lower_bounds[6]) * SF_1
    SF_2 = lower_bounds[7] + (upper_bounds[7] - lower_bounds[7]) * SF_2
    SF_3 = lower_bounds[8] + (upper_bounds[8] - lower_bounds[8]) * SF_3
    SF_4 = lower_bounds[9] + (upper_bounds[9] - lower_bounds[9]) * SF_4
    
    
    if flag=='numpy':
        nu_d_mod = lambda T: 1e15 * (B_d + A_d * np.exp(E_d/(const_dict['R'] * T)))
        nu_D_mod = lambda T: 1e13 * (B_D + A_D * np.exp(E_D/(const_dict['R'] * T)))
    elif flag=='torch':
        nu_d_mod = lambda T: 1e15 * (B_d + A_d * torch.exp(E_d/(const_dict['R'] * T)))
        nu_D_mod = lambda T: 1e13 * (B_D + A_D * torch.exp(E_D/(const_dict['R'] * T)))
    else:
        raise ValueError(f"{flag} does not exist")
    
    dict_mod_vec = [
    {"id": 2, "rate": None, "model_dict": {"nu_d": nu_d_mod}},
    {"id": 10, "rate": None, "model_dict": {"nu_d": nu_d_mod}},
    {"id": 31, "rate": None, "model_dict": {"nu_d": nu_d_mod}},
    
    {"id": 5, "rate": None, "model_dict": {"nu_D": nu_D_mod}},
    {"id": 7, "rate": None, "model_dict": {"nu_D": nu_D_mod}},
    {"id": 8, "rate": None, "model_dict": {"nu_D": nu_D_mod}},
    
    {"id": 34, "rate": None, "model_dict": {"SF": SF_1}},
    {"id": 35, "rate": None, "model_dict": {"SF": SF_2}},
    {"id": 36, "rate": None, "model_dict": {"SF": SF_3}},
    {"id": 37, "rate": None, "model_dict": {"SF": SF_4}},
    ]
    
    return dict_mod_vec

##! define the default parameters
params_default_aux = list((1e-4, 1e-5, 19.0, 1e-4, 1e-5, 19.0, 1e-1, 1e-2, 1e-1, 1e-2))

params_default = []
for idx, param in enumerate(params_default_aux):
    value = (param - lower_bounds[idx])/(upper_bounds[idx] - lower_bounds[idx])
    params_default.append(value)

params_default = tuple(params_default)
print("params_default: ", params_default)
    

def loss_function(exp, teo, flag='numpy'):
    
    func = ((teo-exp)**2)/(exp**2)
    if flag == 'numpy':
        return np.mean(func)
    elif flag == 'torch':
        return torch.mean(func)
    else:
        raise ValueError(f"{flag} does not exist")


optimizer = opt.Optimizer(sim, 
                        lambda params: func_optimization(params, 'numpy'), 
                        lambda exp, teo: loss_function(exp, teo, 'numpy')
                        )

diff = sim_diff.SimDiff(sim, 
                        lambda params: func_optimization(params, 'torch'),
                        params_default=params_default,
                        gamma_exp_data=sim.gamma_exp_data_arr,
                        loss_function=lambda exp, teo: loss_function(exp, teo, 'torch')
                        )


##! Auxiliar functions used for the optimization
def lhs_sampling(n_samples, n_dims):
    samples = np.zeros((n_samples, n_dims))
    for j in range(n_dims):
        perm = np.random.permutation(n_samples)
        samples[:, j] = (perm + np.random.rand(n_samples)) / n_samples
        
    return samples


def map_to_bounds(samples, lower_bounds, upper_bounds):
    return lower_bounds + (upper_bounds - lower_bounds) * samples


def loss_func(params, opt_object):
    loss_val, _, _, _, _ = opt_object.objective_function_diff(params)    
    # print("loss: ", loss_val, "params: ", params)
    return loss_val


def loss_func_B(v, x_base, opt_object, B):
    x_temp = x_base.copy()
    x_temp[B] = v
    loss_val, _, _, _, _ = opt_object.objective_function_diff(x_temp)    
    return loss_val


def loss_and_grads(params, opt_object, diff_object):
    loss_val, frac_solutions_arr, rates_arr, _, gammas_predicted_arr = opt_object.objective_function_diff(params)
    grad_val = diff_object.objective_function_grad(params, frac_solutions_arr, rates_arr, gammas_predicted_arr)
    # print("loss_val: ", loss_val, "grad_val: ", grad_val, "params: ", params)
    return loss_val, grad_val.numpy()


def loss_and_grads_B(v, x_base, B):
    x_temp = x_base.copy()
    x_temp[B] = v
    loss, grad = loss_and_grads(x_temp, optimizer, diff)
    return loss, grad[B]


def loss_and_grads_sampler(params, opt_object, diff_object):
    loss_val, frac_solutions_arr, rates_arr, _, gammas_predicted_arr = opt_object.objective_function_diff(params)
    grad_val = diff_object.objective_function_grad(params, frac_solutions_arr, rates_arr, gammas_predicted_arr)
    return grad_val.numpy()


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


def run_local_optimization_B(params, opt_object, diff_object):
    params, x_base, B, lower_bounds, upper_bounds = params_aug
    
    res = sp.optimize.minimize(
        lambda params: loss_and_grads_B(params, x_base, B),
        x0=params,
        jac=True,
        bounds=[(a_i, b_i) for a_i, b_i in zip(lower_bounds[B], upper_bounds[B])],
        method='L-BFGS-B')
    return res



def gradient_similarity(Sigma, eps=1e-8):
    # Normalized absolute covariance → [0,1]
    diag = np.sqrt(np.diag(Sigma) + eps)
    W = np.abs(Sigma) / (diag[:,None] * diag[None,:])
    np.fill_diagonal(W, 0.0)
    return W  # n×n matrix of edge weights


def spectral_blocks(W, k):
    ## Build Degree & Normalized Laplacian
    d = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-12))
    L_sym = np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt
    ## Smallest k eigenvectors of L_sym
    vals, vecs = eigsh(L_sym, k=k, which='SM')
    U = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ## k-means on rows of U
    labels = KMeans(n_clusters=k, random_state=seed).fit_predict(U)
    blocks = [np.where(labels == i)[0].tolist() for i in range(k)]
    return blocks


###! run optimization pipeline
if __name__ == '__main__':
    
    lower_bounds = np.array([0.0]*10)
    upper_bounds = np.array([1.0]*10)
    output_file = "results4V3multi.txt"
    
    num_workers = 8
    de_max_iter = 20
    top_k = 1
    n_samples = 200
    n_blocks = 3
    max_cycles = 10
    
    config = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "n_samples": n_samples,
        "output_file": output_file
    }
    
    run_grads = False    
    pool = ProcessPool(nodes=num_workers)
    
    if run_grads:
        ###* Input Space Discretization
        samples = lhs_sampling(n_samples, len(lower_bounds))
        samples = map_to_bounds(samples, lower_bounds, upper_bounds)
        
        ###* Run the local minimization procedure
        results = list(tqdm.tqdm(
                pool.imap(
                        lambda params: loss_and_grads_sampler(params, optimizer, diff),
                        samples
                ),
                total=len(samples),
                desc="Optimizing"
            )
        )
        grads_arr = np.array(results)
        with h5py.File("Grads.hdf5", 'w') as f:
            f.create_dataset("grads", data=grads_arr)
    else:
        
        with h5py.File("Grads.hdf5", 'r') as f:
            grads_arr = f['grads'][:]
    
    
    ###* Grad Covariance Matrix
    grads_mean = grads_arr.mean(axis=0, keepdims=True)
    G = grads_arr - grads_mean
    Sigma = (G.T @ G)/ n_samples
    W = gradient_similarity(Sigma)
    blocks = spectral_blocks(W, n_blocks)
    print("blocks: ", blocks)
    
    params_default_arr = np.array(params_default)
    x = params_default_arr.copy()
    tol = 1e-5
    for cycle in range(max_cycles):
        x_prev = x.copy()
        loss_prev = loss_func(x_prev, optimizer)
        
        for B in blocks:
            
            ###* DE optimization - global search
            func = partial(loss_func_B, x_base=x, opt_object=optimizer, B=B)
            de_result = sp.optimize.differential_evolution(
                    func,
                    bounds=np.vstack((lower_bounds[B], upper_bounds[B])).T,
                    x0=x[B],
                    strategy='best1bin',
                    maxiter=de_max_iter,
                    popsize=15,
                    polish=False,
                    disp=True,          
                    workers=pool.map,      
                    updating="deferred"   
            )
            
            ###* Run the local minimization procedure based on the top-k elements 
            sorted_indices = np.argsort(de_result.population_energies)
            top_indices = sorted_indices[:top_k]
            population_top_k = de_result.population[top_indices]
            loss_top_k = de_result.population_energies[top_indices]
            
            pbar = tqdm.tqdm(total=top_k, desc="Local Optimization")
            results_top_k = []
            for params in population_top_k:
                params_aug = (params, x, B, lower_bounds, upper_bounds)
                results = run_local_optimization_B(params_aug, optimizer, diff)
                results_top_k.append(results)
                pbar.update(1)
            pbar.close()
            
            ###* choose the best one and update x
            best_loss = np.inf
            best_result = None
                
            for result in results_top_k:
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result
                
            x[B] = best_result.x
            loss = loss_func(x, optimizer)
            
        if np.linalg.norm(x - x_prev) < tol * np.linalg.norm(x_prev):
            break
        
    
    pool.close()
    pool.join()
    
    params_best = x
    loss_best = loss(params_best, optimizer)
    
    print("params_best: ", params_best)
    print("loss_best: ", loss_best)
    
    config['params_before_refinement'] = list(params_best)
    config['blocks'] = blocks
    
    ###* run one step local optimization on the full problem
    result_refined = run_local_optimization(params_best, optimizer, diff, config)
    
    print("result_refined: ", result_refined)
    
    ###* store results
    grad_based.store_results_2(result_refined, config, output_file)