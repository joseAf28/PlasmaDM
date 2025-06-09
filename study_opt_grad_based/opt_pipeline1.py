import sys, os
PATH = os.path.dirname(os.path.abspath(os.curdir))
if PATH not in sys.path:
    sys.path.insert(0, PATH)

from pathos.multiprocessing import ProcessPool
import src.Simulator as sim_system
import src.Optimizer as opt
import src.SimGrad as sim_diff
import src.Grad_based_methods as grad_based
import scipy as sp
import numpy as np
import torch
import tqdm


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



###* Create optimization and diff objects

##! define parameters to optimize
def func_optimization(params, flag='numpy'):
    
    A_d, B_d, E_d, A_D, B_D, E_D, SF_1, SF_2, SF_3, SF_4 = params
    
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
params_default = (1e-4, 1e-5, 19.0, 1e-4, 1e-5, 19.0, 1e-1, 1e-2, 1e-1, 1e-2)


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



###! run optimization pipeline
if __name__ == '__main__':
    
    lower_bounds = np.array([1e-8, 1e-8, 0.0, 1e-8, 1e-8, 0.0, 1e-5, 1e-5, 1e-5, 1e-5])
    upper_bounds = np.array([1e-1, 1e-1, 30.0, 1e-1, 1e-1, 30.0, 1.0, 1.0, 1.0, 1.0])
    n_samples = 16
    output_file = "results2.txt"
    config = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "n_samples": n_samples,
        "output_file": output_file
    }
    num_workers = 8
    
    ###* Input Space Discretization
    lhs_samples = grad_based.lhs_sampling(n_samples, len(lower_bounds))
    samples = grad_based.map_to_bounds(lhs_samples, lower_bounds, upper_bounds)
    
    
    ###* Run the local minimization procedure
    with ProcessPool(nodes=num_workers) as pool:
        results = list(tqdm.tqdm(
                    pool.imap(
                        lambda params: grad_based.run_local_optimization(params, optimizer, diff, config),
                        samples
                ),
                total=len(samples),
                desc="Optimizing"
            )
        )
    
    
    ###* choose the best one
    best_loss = np.inf
    best_result = None
    
    for result in results:
        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result
    
    
    grad_based.store_results(samples, results, best_result, config)
    print("best_result: ",  best_result)
    print("results_vec: ", results)