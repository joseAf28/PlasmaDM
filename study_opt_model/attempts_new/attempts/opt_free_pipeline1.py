import sys, os
path = os.path.dirname(os.path.abspath(os.curdir))
if path not in sys.path:
    sys.path.insert(0, path)


import numpy as np
import scipy as sp
import h5py

import src.SimData as sim_data
import src.Simulator as sim_system
import src.Optimizer as opt
import src.Model_free_methods as free_based


###* Input files and function

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


##! define the optimization function
def func_optimization(params):
    
    SF_1, SF_2, SF_3, SF_4, SF_5, SF_6, SF_7, SF_8 = params
    
    A = 2.777e-02
    B = 7.792e-04
    E = 1.627e+01
    
    nu_d_mod = lambda T: 1e15 * (A + B * np.exp(E/(const_dict['R'] * T)))
    
    dict_mod_vec = [
    {"id": 2, "rate": None, "model_dict": {"nu_d": nu_d_mod}},
    {"id": 10, "rate": None, "model_dict": {"nu_d": nu_d_mod}},
    {"id": 31, "rate": None, "model_dict": {"nu_d": nu_d_mod}},
    
    {"id": 32, "rate": None, "model_dict": {"SF": SF_1}},
    {"id": 35, "rate": None, "model_dict": {"SF": SF_2}},
    
    {"id": 33, "rate": None, "model_dict": {"SF": SF_3}},
    {"id": 34, "rate": None, "model_dict": {"SF": SF_4}},
    
    {"id": 36, "rate": None, "model_dict": {"SF": SF_5}},
    {"id": 37, "rate": None, "model_dict": {"SF": SF_6}},
    
    {"id": 38, "rate": None, "model_dict": {"SF": SF_7}},
    {"id": 39, "rate": None, "model_dict": {"SF": SF_8}},
    ]
    
    return dict_mod_vec

loss_function = lambda exp, teo: np.mean((np.reciprocal(exp)*(exp-teo))**2)

exp_file = os.path.join(output_folder_path, exp_data_file)
sim = sim_system.Simulator(reactions_file, const_dict, exp_file, initial_state_dict, transformations_exp=transformations_exp)


if __name__ == '__main__':
    optimizer = opt.Optimizer(sim, func_optimization, loss_function)

    global_bounds = [
                    (1e-3, 5e-2), (5e-2, 0.5), (1e-3, 9e-2), 
                    (1e-3, 9e-2), (6e-2, 6e-1), (9e-3, 9e-2),
                    (5e-3, 9e-2), (5e-2, 0.3)
                    ]

    config = {
        "bounds": global_bounds,
        "de_maxiter": 3,
        "de_num_iterations": 1,
    }
    output_folder_path = "results.txt"
    
    best_local, best_local_loss = free_based.pipeline1(optimizer, config)
    free_based.store_results(best_local, best_local_loss, output_folder_path)