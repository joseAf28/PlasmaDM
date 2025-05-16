import numpy as np
import scipy as sp
import os
from pathos.multiprocessing import ProcessPool

import SimulatorExp as sim_system 


class ErrorPropagator():
    
    def __init__(self, simulator_object, func_uncertainty_dict, transformations_exp, print_flag=True, seed=42):
        
        np.random.seed(seed)
        
        self.simulator = simulator_object
        self.exp_data_arr = self.simulator.exp_data_arr
        self.transformations_exp = transformations_exp
        self.const_dict = self.simulator.const_dict
        
        self.func_uncertainty_dict = func_uncertainty_dict
        
        self.gamma_exp_data = self.simulator.gamma_exp_data_arr
        
        self.functional_calls = 0
        self.print_flag = print_flag
    
    
    def update_reaction_list(self, reactions_list, dict_new_vec):
        ###* update the reaction list
        id2index_model = {reaction['id']: idx for idx, reaction in enumerate(reactions_list)}
        
        for dict_new in dict_new_vec:
            mod_update = dict_new["model_dict"]
            
            if dict_new["id"] is not None:
                idx = id2index_model[dict_new["id"]]
                reactions_list[idx]["model_dict"].update(mod_update)
                
            else:
                
                if dict_new["rate"] is not None:
                    idx2update = [idx for idx, reaction in enumerate(reactions_list) if reaction["rate"] == dict_new["rate"]]
                    for idx in idx2update:
                        reactions_list[idx]["model_dict"].update(mod_update)
                else:
                    
                    if len(mod_update) == 1:
                        param_update = next(iter(mod_update))
                    else:
                        raise ValueError(f"Dictionaire 'model_dict' has multiple keys: {dict_new}")
                    
                    idx2update = []
                    for idx, reaction in enumerate(reactions_list):
                        reaction_model_dict = reactions_list[idx]['model_dict']
                        if param_update in reaction_model_dict.keys():
                            idx2update.append(idx)
                            
                    for idx in idx2update:
                        reactions_list[idx]["model_dict"].update(mod_update)
                        
        return reactions_list
    
    
    def truncated_gaussian(self, mean=0.0, error=0.1, min_value=0.0, max_value=np.inf, min_sigma=1.0):
        
        sigma = max(error*mean, min_sigma)
        lower, upper = min_value, max_value
        a, b = (lower - mean) / sigma, (upper - mean) / sigma
        
        distribution = sp.stats.truncnorm(a, b, loc=mean, scale=sigma)
        return distribution
    
    
    def solve_simulations_model(self, params, exp_data_arr, reactions_list):
        
        dict_new_vec = self.func_uncertainty_dict(params)
        reactions_list = self.update_reaction_list(reactions_list, dict_new_vec)
        
        rates_calculation_arr = self.simulator.compute_rates(exp_data_arr, reactions_list)
        _, gammas_results_arr = self.simulator.solve_system(exp_data_arr, rates_calculation_arr, solver="fixed_point")
        gammas_sum_arr = np.array([sum(gamma_dict.values()) for gamma_dict in gammas_results_arr])
        return gammas_sum_arr
    
    
    def solve_simulation_exp(self, params, exp_data_arr, reactions_list, exp_defined_list):
        
        n_samples = exp_data_arr.shape[0]
        for j in range(n_samples):
            for k, vec in enumerate(exp_defined_list):
                key = vec[0]
                exp_data_arr[j][key] = params[j, k]
                
                for key in self.transformations_exp.keys():
                    exp_data_arr[j][key] = self.transformations_exp[key](self.const_dict, exp_data_arr[j])
        
        
        ###* compute the new rates and solve the system of equations
        rates_calculation_arr = self.simulator.compute_rates(exp_data_arr, reactions_list)
        _, gammas_results_arr = self.simulator.solve_system(exp_data_arr, rates_calculation_arr, solver="fixed_point")
        gammas_sum = np.array([sum(gamma_dict.values()) for gamma_dict in gammas_results_arr])
        return gammas_sum
    
    
    def error_propagation_model(self, model_uncert_list=None, N=20, nb_workers=4):
        
        exp_data_arr = self.exp_data_arr.copy()
        reactions_list = self.simulator.output_parser['reactions_list'].copy()
        
        if model_uncert_list is not None and self.func_uncertainty_dict is not None:
            params_mod_arr = np.zeros((N, len(model_uncert_list)), dtype=float)
            
            for idx, line in enumerate(model_uncert_list):
                val = line[0]
                error = line[1]
                
                dist = self.truncated_gaussian(mean=val, error=error)
                vec = dist.rvs(N)
                params_mod_arr[:, idx] = vec
        else:
            return ValueError("Add model_uncert_dict or func_uncertainty_dict")
        
        pool = ProcessPool(nodes=nb_workers)
        result = pool.map(lambda params:self.solve_simulations_model(params, exp_data_arr, reactions_list), params_mod_arr)
        pool.close()
        
        gammas_arr = np.array(result)
        return params_mod_arr, gammas_arr
    
    
    def error_propagation_exp(self, exp_uncert_list=None, N=1_00, nb_workers=4):
        
        exp_data_arr = self.exp_data_arr.copy()
        n_samples = exp_data_arr.shape[0]
        
        reactions_list = self.simulator.output_parser['reactions_list'].copy()
        
        if exp_uncert_list is not None:
            params_exp_arr = np.zeros((N, n_samples, len(exp_uncert_list)), dtype=float)
            
            for i in range(n_samples):
                for j in range(len(exp_uncert_list)):
                    name = exp_uncert_list[j][0]
                    error = exp_uncert_list[j][1]
                    
                    mean = exp_data_arr[i][name]
                    dist = self.truncated_gaussian(mean=mean, error=error)
                    vec = dist.rvs(N)
                    
                    params_exp_arr[:, i, j] = vec
        else:
            return ValueError("Add model_uncert_dict or func_uncertainty_dict")
        
        
        pool = ProcessPool(nodes=nb_workers)
        result = pool.map(lambda params:self.solve_simulation_exp(params, exp_data_arr, reactions_list, exp_uncert_list), params_exp_arr)
        pool.close()
        
        result = np.array(result)
        gammas_arr = result[:, 2]
        
        return params_exp_arr, gammas_arr