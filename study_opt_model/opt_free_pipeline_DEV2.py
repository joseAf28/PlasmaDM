
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
                    1e-5, 1e-5, 1e-5, 1e-5, 1e-5, \
                    1e-5, 1e-5, 1e-5, 1e-5, 1e-5
                    ])

upper_bounds = np.array([5e-1, 1e-2, 30.0, \
                    1.0, 1.0, 1.0, 1.0, 1.0, \
                    1.0, 1.0, 1.0, 1.0, 1.0
                    ])


##! define the default parameters
params_default_init = np.array([
                0.02634, 7.67e-4, 10.75, \
                1.0, 1.0, 1e-2, 1e-1, 1e-1, \
                1e-2, 1e-1, 1e-1, 1e-1, 1e-1
                ])

params_default_norm = (params_default_init - lower_bounds) * np.reciprocal(upper_bounds - lower_bounds)

###! optimization function
def func_optimization(params_input, flag='numpy'):
    
    ##! normalize variables
    params = [0] * len(params_input)
    for idx, param in enumerate(params_input):
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
                        params_default=torch.tensor(params_default_norm),
                        gamma_exp_data=sim.gamma_exp_data_arr,
                        loss_function=lambda exp, teo: loss_function(exp, teo, 'torch')
                        )



#### tracking the multiprocessing procedure
history = {
    'generation': [],
    'best_loss': [],
    'nfev': []
}


def init_worker(counter):
    global nfev
    nfev = counter                 


def counted_obj(x):
    with nfev.get_lock():
        nfev.value += 1
        
    return optimizer.objective_function(x)


def store_callback(xk, convergence):
    
    global history
    loss = optimizer.objective_function(xk)
    history['generation'].append(len(history['generation'])+1)
    history['best_loss'].append(loss)
    history['nfev'].append(nfev.value)
    
    print("best_loss: ", loss, "nfev: ", nfev.value)




if __name__ == '__main__':

    ##* shared value
    nfev = mp.Value('i', 0)
    
    de_max_generations = 90
    de_polish = True
    de_pop_size = 15
    num_workers = 8
    filename_results = f"results_DE_{de_max_generations}_merged_All.h5"
    
    
    optimizer = opt.Optimizer(sim, func_optimization, loss_function)
    
    lower_bounds = [0.0] * len(lower_bounds)
    upper_bounds = [1.0] * len(upper_bounds)
    global_bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]
    
    pool = ProcessPool(
    nodes=num_workers,              
    initializer=init_worker,
    initargs=(nfev,)
    )
    
    de_result = sp.optimize.differential_evolution(
                func=counted_obj,
                x0=params_default_norm,
                bounds=global_bounds,
                strategy='best1bin',
                maxiter=de_max_generations,
                popsize=de_pop_size,
                tol=0.01,
                recombination=0.7,
                polish=de_polish,
                disp=True,          
                workers=pool.map,   
                updating="deferred", 
                callback=store_callback
    )
    
    pool.close()
    pool.join()
    
    
    print("de_resul:", de_result)
    
    history['generation'].append(de_result.nit)
    history['nfev'].append(de_result.nfev)
    history['best_loss'].append(de_result.fun)
    
    history['generation'] = np.array(history['generation'])
    history['nfev'] = np.array(history['nfev'])
    history['best_loss'] = np.array(history['best_loss'])
    
    
    with h5py.File(filename_results, "w") as f:
        
        f.create_dataset("generation", data=history['generation'])
        f.create_dataset("best_loss", data=history['best_loss'])
        f.create_dataset("nfev", data=history['nfev'])
        f.create_dataset("best_params", data=de_result.x)
    
    
    print("Generations:", history['generation'])
    print("Best losses:", history['best_loss'])
    print("Function evaluations:", history['nfev'])


# differential_evolution step 17: f(x)= 0.2491924582204291
# best_loss:  0.2491924582204291 nfev:  3510
# differential_evolution step 18: f(x)= 0.2491924582204291
# best_loss:  0.2491924582204291 nfev:  3705
# differential_evolution step 19: f(x)= 0.2491924582204291
# best_loss:  0.2491924582204291 nfev:  3900
# differential_evolution step 20: f(x)= 0.2491924582204291
# best_loss:  0.2491924582204291 nfev:  4095