import sys, os
PATH = os.path.dirname(os.path.abspath(os.curdir))
if PATH not in sys.path:
    sys.path.insert(0, PATH)

import logging

logging.basicConfig(level=logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import multiprocessing as mp
from pathos.multiprocessing import ProcessPool
import src.Simulator as sim_system
import src.Optimizer as opt
import scipy as sp
import numpy as np
import torch
import tqdm
import inspect
import h5py


###* Create Simulator object
reactions_file = "../reactions/reactionsCompleteV2.json"

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
exp_data_file = "Experimental_data_CO_O_merged.hdf5"
exp_file = os.path.join(output_folder_path, exp_data_file)

sim = sim_system.Simulator(reactions_file, const_dict, exp_file, initial_state_dict, transformations_exp=transformations_exp)

####! bounds observables
lower_bounds = np.array([1e-8, 1e-8, 0.0, \
                    1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 
                    1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, \
                    1.0, 2000
                    ])

upper_bounds = np.array([5e-1, 1e-2, 30.0, \
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
                    5.0, 5000
                    ])


##! define the default parameters
params_default_init = np.array([
                0.02634, 7.67e-4, 10.75, \
                1.0, 1.0, 1e-2, 1e-1, 1e-1, 1e-2, 1e-1, 1e-1, 1e-1, 1e-1, \
                1e-2, 1.0, 1.0, 1.0, 1e-1, 1e-1, 1.0, 1.0, 1.0, 1e-1, 1e-1, 1e-2, 1e-1, 1e-1, \
                3.4, 3000
                ])


params_default_norm = (params_default_init - lower_bounds) * np.reciprocal(upper_bounds - lower_bounds)

###! optimization function
def func_optimization(params_input, flag='numpy'):
    
    ##! normalize variables
    params = [0] * len(params_input)
    for idx, param in enumerate(params_input):
        params[idx] = lower_bounds[idx] + (upper_bounds[idx] - lower_bounds[idx]) * param
    
    A_d, B_d, E_d = params[0:3] # 3
    SF_30, SF_31, SF_32, SF_33, SF_34, SF_35, SF_36, SF_37, SF_38, SF_39 = params[3:13] # 10 
    SF_49, SF_50, SF_51, SF_52, SF_53, SF_54, SF_55, SF_56, SF_57, SF_58, SF_59, SF_60, SF_61, SF_62 = params[13:27] # 14
    Emin, Ealpha = params[27:] # 2
    
    if flag=='numpy':
        nu_d_mod = lambda T: 1e15 * (A_d + B_d * np.exp(E_d/(const_dict['R'] * T)))
    elif flag=='torch':
        nu_d_mod = lambda T: 1e15 * (A_d + B_d * torch.exp(E_d/(const_dict['R'] * T)))
    else:
        raise ValueError(f"{flag} does not exist")
    
    dict_mod_vec = [
    {"id": 2, "rate": None, "model_dict": {"nu_d": nu_d_mod}},
    {"id": 10, "rate": None, "model_dict": {"nu_d": nu_d_mod}},
    {"id": 16, "rate": None, "model_dict": {"Emin": Emin}},
    {"id": 18, "rate": None, "model_dict": {"Emin": Emin}},
    
    {"id": 31, "rate": None, "model_dict": {"SF": SF_31, "nu_d": nu_d_mod}},
    
    {"id": 30, "rate": None, "model_dict": {"SF": SF_30}},
    {"id": 32, "rate": None, "model_dict": {"SF": SF_32}},
    {"id": 33, "rate": None, "model_dict": {"SF": SF_33}},
    {"id": 34, "rate": None, "model_dict": {"SF": SF_34}},
    
    {"id": 35, "rate": None, "model_dict": {"SF": SF_35}},
    {"id": 36, "rate": None, "model_dict": {"SF": SF_36}},
    {"id": 37, "rate": None, "model_dict": {"SF": SF_37}},
    {"id": 38, "rate": None, "model_dict": {"SF": SF_38}},
    {"id": 39, "rate": None, "model_dict": {"SF": SF_39}},
    
    {"id": 44, "rate": None, "model_dict": {"Emin": Emin}},
    
    {"id": 49, "rate": None, "model_dict": {"SF": SF_49}},
    {"id": 50, "rate": None, "model_dict": {"SF": SF_50, "Ealpha": Ealpha}},
    {"id": 51, "rate": None, "model_dict": {"SF": SF_51, "Ealpha": Ealpha}},
    {"id": 52, "rate": None, "model_dict": {"SF": SF_52, "Ealpha": Ealpha}},
    {"id": 53, "rate": None, "model_dict": {"SF": SF_53, "Ealpha": Ealpha}},
    {"id": 54, "rate": None, "model_dict": {"SF": SF_54, "Ealpha": Ealpha}},
    {"id": 55, "rate": None, "model_dict": {"SF": SF_55, "Ealpha": Ealpha}},
    {"id": 56, "rate": None, "model_dict": {"SF": SF_56, "Ealpha": Ealpha}},
    {"id": 57, "rate": None, "model_dict": {"SF": SF_57, "Ealpha": Ealpha}},
    {"id": 58, "rate": None, "model_dict": {"SF": SF_58, "Ealpha": Ealpha}},
    {"id": 59, "rate": None, "model_dict": {"SF": SF_59, "Ealpha": Ealpha}},
    {"id": 60, "rate": None, "model_dict": {"SF": SF_60}},
    {"id": 61, "rate": None, "model_dict": {"SF": SF_61}},
    {"id": 62, "rate": None, "model_dict": {"SF": SF_62}}
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



##! functions to optimize 
def lhs_sampling(n_samples, n_dims):
    samples = np.zeros((n_samples, n_dims))
    for j in range(n_dims):
        perm = np.random.permutation(n_samples)
        samples[:, j] = (perm + np.random.rand(n_samples)) / n_samples
        
    return samples



lower_bounds = np.array([1e-8, 1e-8, 0.0, \
                    1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 
                    1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, \
                    1.0, 2000
                    ])

upper_bounds = np.array([5e-1, 1e-2, 30.0, \
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
                    5.0, 5000
                    ])


params_default_init = np.array([
                0.02634, 7.67e-4, 10.75, \
                1.0, 1.0, 1e-2, 1e-1, 1e-1, 1e-2, 1e-1, 1e-1, 1e-1, 1e-1, \
                1e-2, 1.0, 1.0, 1.0, 1e-1, 1e-1, 1.0, 1.0, 1.0, 1e-1, 1e-1, 1e-2, 1e-1, 1e-1, \
                3.4, 3000
                ])


params_default_norm = (params_default_init - lower_bounds) * np.reciprocal(upper_bounds - lower_bounds)


###* global variables to keep track of the evolution measures

iters_calls = 0
best_loss = float('inf')
history = {'iters':[], 'best_loss':[]}


def loss(params, opt_object):
    
    global iters_calls
    global best_loss
    global history
    
    loss_val, frac_solutions_arr, rates_arr, _, gammas_predicted_arr = opt_object.objective_function_diff(params)
    best_loss = loss_val if loss_val < best_loss else best_loss
    
    
    if iters_calls % 10 == 0:
        history['iters'].append(iters_calls)
        history['best_loss'].append(best_loss)
        print("best_loss: ", best_loss, "iters: ", iters_calls)
    
    iters_calls += 1
    
    return loss_val



def run_local_optimization(params_arr, opt_object):
    lower_bounds = [0] * len(params_arr)
    upper_bounds = [1.0] * len(params_arr)
    
    res = sp.optimize.minimize(
        lambda params: loss(params, opt_object),
        x0=params_arr,
        jac=False,
        bounds=[(a_i, b_i) for a_i, b_i in zip(lower_bounds, upper_bounds)],
        method='Powell',
        options={
                'maxfev': 2500, 
        }
        
        )
    
    return res


###! run optimization pipeline
if __name__ == '__main__':
    
    output_file = f"results_opt_nograd_All.h5"
    result = run_local_optimization(params_default_norm, optimizer)
    
    with h5py.File(output_file, "w") as f:
        f.create_dataset("best_loss", data=history['best_loss'])
        f.create_dataset("params", data=result.x)
        f.create_dataset("iters", data=history['iters'])