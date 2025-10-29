import sys, os
PATH = os.path.dirname(os.path.abspath(os.curdir))
if PATH not in sys.path:
    sys.path.insert(0, PATH)

import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from pathos.multiprocessing import ProcessPool
import src.Simulator as sim_system
import src.Optimizer as opt
import src.SimGrad as sim_diff
import scipy as sp
import numpy as np
import torch
import tqdm
import inspect
import h5py


###* Create Simulator object
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


####! bounds observables
lower_bounds = np.array([1e-8, 1e-8, 0.0, \
                    1e-5, 1e-5, 1e-5, 1e-5, 1e-5, \
                    1e-5, 1e-5, 1e-5, 1e-5, 1e-5
                    ])

upper_bounds = np.array([5e-1, 1e-2, 30.0, \
                    1.0, 1.0, 1.0, 1.0, 1.0, \
                    1.0, 1.0, 1.0, 1.0, 1.0
                    ])


##! define the default parameters
params_default = []
params_default_aux = list((0.01634, 1.67e-4, 19.75, \
                1.0, 1.0, 1e-2, 1e-1, 1e-1, \
                1e-2, 1e-1, 1e-1, 1e-1, 1e-1
                ))
for idx, param in enumerate(params_default_aux):
    value = (param - lower_bounds[idx])/(upper_bounds[idx] - lower_bounds[idx])
    params_default.append(value)
params_default = tuple(params_default)


###! optimization function
def func_optimization(params, flag='numpy'):
    
    ##! normalize variables
    params = list(params)
    for idx, param in enumerate(params):
        params[idx] = lower_bounds[idx] + (upper_bounds[idx] - lower_bounds[idx]) * param
    
    A_d, B_d, E_d, SF_1, SF_2, SF_3, SF_4, SF_5, SF_6, SF_7, SF_8, SF_9, SF_10 = params
    
    
    if flag=='numpy':
        nu_d_mod = lambda T: 1e15 * (A_d + B_d * np.exp(E_d/(const_dict['R'] * T)))
    elif flag=='torch':
        nu_d_mod = lambda T: 1e15 * (A_d + B_d * torch.exp(E_d/(const_dict['R'] * T)))
    else:
        raise ValueError(f"{flag} does not exist")
    
    dict_mod_vec = [
    {"id": 2, "rate": None, "model_dict": {"nu_d": nu_d_mod}},
    {"id": 10, "rate": None, "model_dict": {"nu_d": nu_d_mod}},
    {"id": 31, "rate": None, "model_dict": {"SF": SF_2, "nu_d": nu_d_mod}},
    
    {"id": 30, "rate": None, "model_dict": {"SF": SF_1}},
    {"id": 32, "rate": None, "model_dict": {"SF": SF_3}},
    {"id": 33, "rate": None, "model_dict": {"SF": SF_4}},
    {"id": 34, "rate": None, "model_dict": {"SF": SF_5}},
    
    {"id": 35, "rate": None, "model_dict": {"SF": SF_6}},
    {"id": 36, "rate": None, "model_dict": {"SF": SF_7}},
    {"id": 37, "rate": None, "model_dict": {"SF": SF_8}},
    {"id": 38, "rate": None, "model_dict": {"SF": SF_9}},
    {"id": 39, "rate": None, "model_dict": {"SF": SF_10}},
    ]
    
    return dict_mod_vec


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


###! function used for stiff/sloppy optimization

def grads_wise(params, opt_object, diff_object):
    _, frac_solutions_arr, rates_arr, _, gammas_predicted_arr = (opt_object.objective_function_diff(params))
    grad_val = diff_object.objective_function_grad_element_wise(params, frac_solutions_arr, rates_arr, gammas_predicted_arr)
    return grad_val.detach()


def create_subspaces(params, percent_info, opt_object, diff_object, eps = 1e-8, reg = 1e-6):

    grad_errors = - grads_wise(params, opt_object, diff_object)
    gamma_exp = torch.tensor(
        sim.gamma_exp_data_arr, dtype=grad_errors.dtype
    ).reshape(-1, 1)
    grad_errors = grad_errors / (gamma_exp + eps)
    norms = grad_errors.norm(dim=1, keepdim=True) + eps
    G = grad_errors / norms                 
    F  = G.T @ G                            

    F_reg = F + reg * torch.eye(F.size(0), dtype=F.dtype)
    eigvals, eigvecs = torch.linalg.eigh(F_reg) # ascending
    
    idx = torch.argsort(eigvals, descending=True)
    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]
    
    total_mass = eigvals_sorted.sum()
    cumulative  = torch.cumsum(eigvals_sorted, dim=0)
    k = int((cumulative < percent_info * total_mass).sum().item()) + 1
    
    Vs = eigvecs_sorted[:, :k]
    Vl = eigvecs_sorted[:, k:]
    return {
        'num_components': k,
        'Vs': Vs,
        'Vl': Vl,
        'eigvals_sorted': eigvals_sorted,
        'eigvecs_sorted': eigvecs_sorted
    }



def loss_stiff_subspace(phi, opt_object, diff_object, config):
    num_components = config['num_components']
    Vs = config['Vs']
    Vl = config['Vl']
    phi0 = config['phi0']
    
    params_aux = np.dot(Vs, phi).reshape(-1) + np.dot(Vl, phi0[num_components:]).reshape(-1)
    params = tuple(np.abs(params_aux).reshape(-1))    
    
    loss_val, frac_solutions_arr, rates_arr, _, gammas_predicted_arr = opt_object.objective_function_diff(params)
    
    print("loss stiff: ", loss_val, "phi: ", phi)
    
    return loss_val



def loss_sloppy_subspace(phi, opt_object, diff_object, config):
    num_components = config['num_components']
    Vs = config['Vs']
    Vl = config['Vl']
    phi0 = config['phi0']
    
    params_aux = np.dot(Vs, phi0[:num_components]).reshape(-1) + np.dot(Vl, phi).reshape(-1)
    params = tuple(np.abs(params_aux).reshape(-1))    
    loss_val, frac_solutions_arr, rates_arr, _, gammas_predicted_arr = opt_object.objective_function_diff(params)
    
    print("loss sloppy: ", loss_val, "params_real: ", params)
    
    return loss_val



####! Optimization procedure
output_file = "results_stiff_sloppy_test.txt"
percent_info = 0.90

##* init space
config_dict = create_subspaces(params_default, percent_info, optimizer, diff)
config_dict['phi0'] = np.linalg.solve(config_dict['eigvecs_sorted'], np.array(params_default))





##* stiff optimization
res_stiff = sp.optimize.minimize(
        lambda params: loss_stiff_subspace(params, optimizer, diff, config_dict),
        x0=config_dict['phi0'][:config_dict['num_components']],
        tol = 5e-5,
        method='Powell'
        )

num_components = config_dict['num_components']
config_dict['phi0'][:num_components] = res_stiff.x
print("stiff optimization done")



###* sloppy optimization
res_sloppy = sp.optimize.minimize(
        lambda params: loss_sloppy_subspace(params, optimizer, diff, config_dict),
        x0 = config_dict['phi0'][config_dict['num_components']:],
        method='Nelder-Mead',
        tol=5e-5
        )

config_dict['phi0'][num_components:] = res_sloppy.x 
phiVs = config_dict['phi0'][:num_components]
phiVl = config_dict['phi0'][num_components:]
params = np.abs(np.dot(config_dict['Vs'], phiVs).reshape(-1) + np.dot(config_dict['Vl'], phiVl).reshape(-1))
params_inversed = lower_bounds + params * (upper_bounds - lower_bounds)


###* store results data
with open(output_file, 'w') as f:
    f.write("Results Stiff-Sloppy optimization\n")
    f.write("\n")
    f.write("params_optimized: ")
    f.write(f"{params_inversed}\n")
    f.write("loss: ")
    f.write(f"{res_sloppy.fun}\n")
    f.write("\n")
    
    for key, vals in config_dict.items():
        f.write(f"{key}: {vals}\n")
        
    f.write("\n")
    f.write("Stiff Optimization: ")
    f.write(f"{res_stiff}\n")
    f.write("\n")
    f.write("Sloppy Optimization: ")
    f.write(f"{res_sloppy}\n")
    f.write("\n")
    
    src = inspect.getsource(func_optimization)
    f.write(f"{src}\n")