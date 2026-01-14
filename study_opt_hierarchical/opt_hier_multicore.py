
import sys, os
PATH = os.path.dirname(os.path.abspath(os.curdir))
if PATH not in sys.path:
    sys.path.insert(0, PATH)

import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import numpy as np
import scipy as sp
import torch
import h5py
from scipy.optimize import least_squares

import src.Optimizer as opt
import src.simulation_setup as setup
import hierarchical_algorithm as ha

from loky import get_reusable_executor



def run_hier_set(buffer_train, filename_results):
    
    const_dict, sim = setup.create_common_simulator(PATH, data_file=buffer_train)
    
    print("optimization with: ", buffer_train, filename_results)
    
    # 2. Define Parameters and Bounds
    lower_bounds_dict = {
        'A_d': 1e-8, 'B_d': 1e-8, 'E_d': 0.0, 
        'SF_30': 1e-5, 'SF_31': 1e-5, 'SF_32': 1e-5, 'SF_33': 1e-5, 'SF_34': 1e-5, 'SF_35': 1e-5, 'SF_36': 1e-5, 'SF_37': 1e-5, 'SF_38': 1e-5, 'SF_39': 1e-5,
        'SF_49': 1e-5, 'SF_50': 1e-5, 'SF_51': 1e-5, 'SF_52': 1e-5, 'SF_53': 1e-5, 'SF_54': 1e-5, 'SF_55': 1e-5, 'SF_56': 1e-5, 'SF_57': 1e-5, 'SF_58': 1e-5, 'SF_59': 1e-5, 'SF_60': 1e-5, 'SF_61': 1e-5, 'SF_62': 1e-5,
        'Emin': 1.0, 'Ealpha': 2000
    }

    upper_bounds_dict = {
        'A_d': 5e-1, 'B_d': 1e-2, 'E_d': 30.0, 
        'SF_30': 1.0, 'SF_31': 1.0, 'SF_32': 1.0,  'SF_33': 1.0, 'SF_34': 1.0, 'SF_35': 1.0, 'SF_36': 1.0, 'SF_37': 1.0, 'SF_38': 1.0, 'SF_39': 1.0,
        'SF_49': 1.0, 'SF_50': 1.0, 'SF_51': 1.0, 'SF_52': 1.0, 'SF_53': 1.0, 'SF_54': 1.0, 'SF_55': 1.0, 'SF_56': 1.0, 'SF_57': 1.0, 'SF_58': 1.0, 'SF_59': 1.0, 'SF_60': 1.0, 'SF_61': 1.0, 'SF_62': 1.0,
        'Emin': 5.0, 'Ealpha': 5000
    }

    params_default_dict = {
        'A_d': 0.02634, 'B_d': 7.67e-4, 'E_d': 10.75, 
        'SF_30': 1.0, 'SF_31': 1.0, 'SF_32': 1e-2,  'SF_33': 1e-1, 'SF_34': 1e-1, 'SF_35': 1e-2, 'SF_36': 1e-1, 'SF_37': 1e-1, 'SF_38': 1e-1, 'SF_39': 1e-1,
        'SF_49': 1e-2, 'SF_50': 1.0, 'SF_51': 1.0, 'SF_52': 1.0, 'SF_53': 1e-1, 'SF_54': 1e-1, 'SF_55': 1.0, 'SF_56': 1.0, 'SF_57': 1.0, 'SF_58': 1e-1, 'SF_59': 1e-1, 'SF_60': 1e-2, 'SF_61': 1e-1, 'SF_62': 1e-1,
        'Emin': 3.4, 'Ealpha': 3000
    }

    lower_bounds = np.array(list(lower_bounds_dict.values()))
    upper_bounds = np.array(list(upper_bounds_dict.values()))
    params_default_init = np.array(list(params_default_dict.values()))
    params_default_norm = (params_default_init - lower_bounds) * np.reciprocal(upper_bounds - lower_bounds)


    def func_optimization(params_input, flag='numpy'):
        
        ##! normalize variables
        params = [0] * len(params_input)
        for idx, param in enumerate(params_input):
            params[idx] = lower_bounds[idx] + (upper_bounds[idx] - lower_bounds[idx]) * param
        
        A_d, B_d, E_d = params[0:3]
        SF_30, SF_31, SF_32, SF_33, SF_34, SF_35, SF_36, SF_37, SF_38, SF_39 = params[3:13]
        SF_49, SF_50, SF_51, SF_52, SF_53, SF_54, SF_55, SF_56, SF_57, SF_58, SF_59, SF_60, SF_61, SF_62 = params[13:27]
        Emin, Ealpha = params[27:]
        
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


    # 4. Instantiate and Run Optimizer
    optimizer = opt.Optimizer(sim, 
                            lambda params: func_optimization(params, 'numpy'), 
                            lambda exp, teo: loss_function(exp, teo, 'numpy')
                            )
    
    max_iters_tot = 1300

    pipeline = [
        ('decompose_exact', {'percent_info': 0.90, 'tau': 1e-4}),
        ('optimize_stiff', {'max_iter': 70}),
        ('realign_sloppy', {}),
        ('optimize_sloppy', {'max_iter': 130}),
        ('check_convergence', {'tol_s': 1e-4})
    ]

    opt_hier = ha.HierarchicalOptimizer(optimizer, params_default_norm, pipeline=pipeline, print_flag=False)
    opt_hier.run(max_cycles=5, max_iter=max_iters_tot)
    
    
    with h5py.File(filename_results, "w") as f:
        
        f.create_dataset("best_loss", data=opt_hier.history['best_loss'])
        f.create_dataset("iters", data=opt_hier.history['iters'])
        f.create_dataset("best_params", data=opt_hier.history['best_params'])
        
        f.create_dataset("best_params_end", data=opt_hier.phi)



if __name__ == "__main__":
    
    
    ###! files that store the results
    
    buffer_vec = [
        "Experimental_data_CO_O_merged_train.hdf5",
        "Experimental_data_CO_O_merged_train1.hdf5",
        "Experimental_data_CO_O_merged_train2.hdf5",
        "Experimental_data_CO_O_merged_train3.hdf5",
        "Experimental_data_CO_O_merged_train4.hdf5",
    ]
    
    write_vec = [
        "results/results_hier0_validation0.h5",
        "results/results_hier0_validation1.h5",
        "results/results_hier0_validation2.h5",
        "results/results_hier0_validation3.h5",
        "results/results_hier0_validation4.h5",
    ]
    
    workers = len(buffer_vec)
    
    executor = get_reusable_executor(max_workers=workers, timeout=2)
    results = executor.map(run_hier_set, buffer_vec, write_vec)