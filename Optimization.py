import numpy as np
import scipy as sp
import h5py
import os
from pathos.multiprocessing import ProcessPool


class Optimizer():
    
    def __init__(self, simulator_object, func_new_model_dict, loss_function, print_flag=True):
        
        self.simulator = simulator_object
        self.func_new_model_dict = func_new_model_dict
        self.loss_function = loss_function
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
    
    
    def solve_simulations(self, params, solver="fixed_point"):
        
        ###* define the dictionaries with the params to update
        dict_new_vec = self.func_new_model_dict(params)
        reactions_list = self.update_reaction_list(self.simulator.output_parser['reactions_list'], dict_new_vec)
        
        ###* compute the new rates and solve the system of equations
        rates_calculation_arr = self.simulator.compute_rates(reactions_list)
        frac_solutions_arr, gammas_results_arr = self.simulator.solve_system(rates_calculation_arr, solver=solver)
        gammas_sum_arr = np.array([sum(gamma_dict.values()) for gamma_dict in gammas_results_arr])
        
        return frac_solutions_arr, gammas_results_arr, gammas_sum_arr
        
        
    def functional_loss(self, params, reaction_list):
        
        ###* define the dictionaries with the new params and update the reaction list
        dict_new_vec = self.func_new_model_dict(params)
        # reactions_list = self.update_reaction_list(self.simulator.output_parser['reactions_list'], dict_new_vec)
        reactions_list = self.update_reaction_list(reaction_list, dict_new_vec)
        
        ###* compute the new rates and solve the system of equations
        rates_calculation_arr = self.simulator.compute_rates(reactions_list)
        frac_solutions_arr, gammas_result_arr = self.simulator.solve_system(rates_calculation_arr, solver="fixed_point")
        
        ###* compute the loss
        gammas_sum_arr = np.array([sum(gamma_dict.values()) for gamma_dict in gammas_result_arr])
        loss = self.loss_function(self.gamma_exp_data, gammas_sum_arr)

        self.functional_calls += 1
        if self.print_flag:
            print("loss: ", loss, "rate r1: ", rates_calculation_arr[0, 0], "rate r2: ", rates_calculation_arr[0, 1], "rate r8: ", rates_calculation_arr[0, 7])
        
        return loss
    
    
    def hybrid_search(self, config, nb_workers=2, print_flag=False):
        
        ###* give a fixed argument for each functional loss call
        reaction_list = self.simulator.output_parser['reactions_list'].copy()
        
        bounds = config["bounds"]
        nb_calls = config["nb_calls"]
        de_maxiter = config["de_maxiter"]
        local_attempts = config["local_attempts"]
        epsilon_local = config["epsilon_local"]
        top_k = config["top_k"]
        
        candidates = []
        
        pool = ProcessPool(nodes=nb_workers)
        for _ in range(nb_calls):
            result = sp.optimize.differential_evolution(self.functional_loss, 
                                                        bounds=bounds,
                                                        args=(reaction_list,),
                                                        polish=True,
                                                        disp=True,
                                                        workers=pool.map,
                                                        updating="deferred",
                                                        maxiter=de_maxiter
                                                        )
        pool.close()
        
        candidates_losses = [self.functional_loss(candidate, reaction_list) for candidate in candidates]
        best_candidate = candidates[np.argmin(candidates_losses)]
        
        if print_flag:
            print("Candidates: ", candidates)
            print("Candidates Losses: ", candidates_losses)
            
            print("Best candidate Global Search: ", best_candidate)
            print()
        
        bounds_array = np.array(bounds)
        k_local = top_k if top_k < nb_de_calls else nb_calls
        
        ### chose the k best candidates
        idx_best_candidates = np.argsort(candidates_losses)[:k_local]
        best_candidates = np.array([candidates[i] for i in idx_best_candidates])
        best_candidates_losses = np.array([candidates_losses[i] for i in idx_best_candidates])
        
        best_local = best_candidates[np.argmin(best_candidates_losses)]
        best_local_loss = np.min(best_candidates_losses)
        
        if print_flag:
            print("best_local_loss: ", best_candidates_losses)
            print("best_local: ", best_candidates)
        
        for best_candidate, best_candidate_loss in zip(best_candidates, best_candidates_losses):
        
            for attempt in range(local_attempts):
                
                perturbation = np.random.uniform(-epsilon_local, epsilon_local, len(best_candidate))
                x0 = best_local + perturbation
                
                ### check bounds
                x0 = np.maximum(x0, bounds_array[:, 0]) 
                x0 = np.minimum(x0, bounds_array[:, 1])
                
                # local_result = sp.optimize.minimize(self.functional_loss, x0, method="Nelder-Mead", bounds=bounds)
                local_result = sp.optimize.minimize(self.functional_loss, x0, 
                                                    args=(reaction_list,), 
                                                    method="L-BFGS-B", 
                                                    bounds=bounds)
                
                if print_flag:
                    print("Local Search Attempt: ", attempt, "Loss: ", local_result.fun, "Params: ", local_result.x)
                    print()
                
                if local_result.fun < best_local_loss:
                    best_local_loss = local_result.fun
                    best_local = local_result.x
        
        # ### trim buffer
        # self.buffer = self.buffer[:self.nb_calls, :]
        
        return best_local, best_local_loss