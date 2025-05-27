import numpy as np
import scipy as sp
import os
import logging
import copy
from pathos.multiprocessing import ProcessPool
from typing import List, Dict, Callable, Any, Tuple, Optional


class Optimizer():
    """
    Optimizes parameters of the simulation model using a hybrid global-local search strategy.
    It interacts with a Simulator object to run simulations and evaluate a loss function.
    """
    
    def __init__(self,
                simulator_object: Any, 
                func_new_model_dict: Callable[[np.ndarray], List[Dict[str, Any]]],
                loss_function: Callable[[np.ndarray, np.ndarray], float],
                use_logging: bool = True):
        
        self.simulator = simulator_object
        self.func_new_model_dict: Callable[[np.ndarray], List[Dict[str, Any]]] = func_new_model_dict
        self.loss_function: Callable[[np.ndarray, np.ndarray], float] = loss_function
        
        # Ensure gamma_exp_data_arr exists and is not None
        if self.simulator.gamma_exp_data_arr is None:
            logging.warning("Experimental gamma data (gamma_exp_data_arr) is None in the simulator object.")
            self.gamma_exp_data: np.ndarray = np.array([])
        else:
            self.gamma_exp_data: np.ndarray = self.simulator.gamma_exp_data_arr
        
        self.exp_data_arr: np.ndarray = self.simulator.exp_data_arr # Array of experiment dicts
        
        self.use_logging: bool = use_logging
        
        # Store an initial deep copy of the reaction list from the simulator and
        # it serves as the base template for modifications.
        if 'reactions_list' not in self.simulator.output_parser:
            raise ValueError("Simulator's output_parser does not contain 'reactions_list'.")
        self._base_reactions_list: List[Dict[str, Any]] = copy.deepcopy(self.simulator.output_parser['reactions_list'])
    
    
    
    def _update_reactions_list_for_params(self,
                                        base_reactions: List[Dict[str, Any]],
                                        param_update_instructions: List[Dict[str, Any]]
                                        ) -> List[Dict[str, Any]]:
        updated_reactions_list = copy.deepcopy(base_reactions) # Work on a fresh deep copy
        id_to_index_map = {reaction['id']: idx for idx, reaction in enumerate(updated_reactions_list)}
        
        for instruction in param_update_instructions:
            target_reaction_id = instruction.get("id")
            target_rate_type = instruction.get("rate")
            model_dict_updates = instruction.get("model_dict")
            
            if not isinstance(model_dict_updates, dict):
                logging.warning(f"Skipping instruction due to invalid 'model_dict': {instruction}")
                continue
            
            indices_to_update: List[int] = []
            
            if target_reaction_id is not None:
                if target_reaction_id in id_to_index_map:
                    indices_to_update.append(id_to_index_map[target_reaction_id])
                else:
                    logging.warning(f"Reaction ID '{target_reaction_id}' not found for update. Instruction: {instruction}")
                    continue
            elif target_rate_type is not None:
                indices_to_update = [idx for idx, reaction in enumerate(updated_reactions_list)
                                    if reaction.get("rate") == target_rate_type]
                if not indices_to_update:
                    logging.warning(f"No reactions found with rate type '{target_rate_type}'. Instruction: {instruction}")
                    continue
            else:
                # Fallback: update based on parameter key presence (original logic, can be risky)
                if len(model_dict_updates) == 1:
                    param_key_to_match = next(iter(model_dict_updates))
                    indices_to_update = [
                        idx for idx, reaction in enumerate(updated_reactions_list)
                        if param_key_to_match in reaction.get('model_dict', {})
                    ]
                    if not indices_to_update:
                        logging.warning(f"No reactions found containing parameter key '{param_key_to_match}' in their model_dict. Instruction: {instruction}")
                        continue
                else:
                    logging.error(f"Ambiguous update: 'id' and 'rate' are None, and 'model_dict' has multiple keys. Instruction: {instruction}")
                    raise ValueError(f"Ambiguous update instruction: {instruction}")
            
            for idx in indices_to_update:
                if 'model_dict' not in updated_reactions_list[idx] or updated_reactions_list[idx]['model_dict'] is None:
                    updated_reactions_list[idx]['model_dict'] = {} # Ensure model_dict exists
                    
                updated_reactions_list[idx]['model_dict'].update(model_dict_updates)
        
        return updated_reactions_list
    
    
    
    def _run_simulation_with_updated_reactions(self,
                                            exp_data_arr: np.ndarray,
                                            current_reactions_list: List[Dict[str, Any]],
                                            solver_type: str = "fixed_point"
                                            ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        
        try:
            
            frac_solutions_arr, gammas_results_arr = self.simulator.solve_all_conditions(self.exp_data_arr, current_reactions_list, solver_type=solver_type)
            
            if gammas_results_arr is None or any(g is None for g in gammas_results_arr):
                logging.warning("Solver returned None or partial None for gamma_results_arr.")
                processed_gammas = []
                for g_dict in gammas_results_arr if gammas_results_arr is not None else []:
                    if g_dict is None:
                        processed_gammas.append({})
                    else:
                        processed_gammas.append(g_dict)
                gammas_results_arr_processed = np.array(processed_gammas, dtype=object)
            else:
                gammas_results_arr_processed = gammas_results_arr
                
            gammas_sum_list = []
            for gamma_dict in gammas_results_arr_processed:
                if isinstance(gamma_dict, dict) and gamma_dict:
                    valid_values = [v for v in gamma_dict.values() if isinstance(v, (int, float)) and not np.isnan(v)]
                    gammas_sum_list.append(sum(valid_values) if valid_values else np.nan)
                else: # Empty dict or not a dict
                    gammas_sum_list.append(np.nan)
            gammas_sum_arr = np.array(gammas_sum_list)
            
            return frac_solutions_arr, gammas_results_arr_processed, gammas_sum_arr
        except Exception as e:
            logging.error(f"Error during simulation run for optimization: {e}", exc_info=True)
            return None, None, None
    
    
    
    def solve_simulations_updated(self, params: np.ndarray) ->  Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        
        param_update_instructions = self.func_new_model_dict(params)
        current_reactions_list = self._update_reactions_list_for_params(self._base_reactions_list, param_update_instructions)
        
        frac_solutions_arr, gammas_results_arr, gammas_sum_arr = self._run_simulation_with_updated_reactions(self.exp_data_arr, current_reactions_list, solver_type="fixed_point")
        
        return frac_solutions_arr, gammas_results_arr, gammas_sum_arr, self.gamma_exp_data
    
    
    
    def objective_function(self, params: np.ndarray) -> float:
        
        param_update_instructions = self.func_new_model_dict(params)
        current_reactions_list = self._update_reactions_list_for_params(self._base_reactions_list, param_update_instructions)
        
        _, _, gammas_simulated_sum = self._run_simulation_with_updated_reactions(self.exp_data_arr, current_reactions_list, solver_type="fixed_point")
        
        if gammas_simulated_sum is None or np.all(np.isnan(gammas_simulated_sum)):
            logging.warning(f"Call #{current_call_number}: Simulation failed or returned all NaN gammas. Loss set to infinity.")
            return np.inf # Or a very large number if np.inf is problematic for optimizer
        
        if len(self.gamma_exp_data) != len(gammas_simulated_sum):
            logging.error(f"Call #{current_call_number}: Mismatch in length between experimental gamma ({len(self.gamma_exp_data)}) "
                        f"and simulated gamma sum ({len(gammas_simulated_sum)}). Returning Inf loss.")
            return np.inf
        
        # Handle NaNs before passing to loss function if it's sensitive
        valid_indices = ~np.isnan(gammas_simulated_sum) & ~np.isnan(self.gamma_exp_data)
        if not np.any(valid_indices):
            logging.warning(f"Call #{current_call_number}: No valid (non-NaN) gamma pairs for loss calculation. Returning Inf loss.")
            return np.inf
        
        loss = self.loss_function(self.gamma_exp_data[valid_indices], gammas_simulated_sum[valid_indices])
        
        if self.use_logging:
            log_msg = f"Loss = {loss:.4e}"
            logging.info(log_msg)
        
        return loss
    
    
    
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
    
    
    
    def hybrid_optimization_search(self, config: Dict[str, Any], num_workers: int = 9, local_refinement: bool = False) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Performs a hybrid optimization: global search with Differential Evolution,
        followed by local search refinement on the best candidates.
        """
        bounds = config.get("bounds")
        de_num_iterations = config.get("de_num_iterations", 5) # How many separate DE runs
        de_max_generations = config.get("de_max_generations", 100)
        de_polish = config.get("de_polish", False) # DE's internal polishing
        de_pop_size = config.get("de_population_size", 15) # Typical DE parameter

        local_attempts_per_candidate = config.get("local_search_attempts_per_candidate", 3)
        perturbation_factor = config.get("local_search_perturbation_factor", 0.05) # Relative to bounds width
        top_k_candidates_for_local = config.get("local_search_top_k_candidates", 3)

        if bounds is None:
            logging.error("Bounds must be provided in the configuration.")
            return None, None

        bounds_array = np.array(bounds)
        if bounds_array.ndim != 2 or bounds_array.shape[1] != 2:
            logging.error("Bounds must be a list of (min, max) tuples.")
            return None, None

        global_search_candidates_params: List[np.ndarray] = []
        global_search_candidates_loss: List[float] = []
        
        self.total_calls = 0
        self.total_itts = 0

        ####* Global Search: Differential Evolution 
        if self.use_logging:
            logging.info(f"Starting Differential Evolution with {de_num_iterations} iteration(s), "
                        f"{de_max_generations} generations each, using {num_workers} worker(s).")
        
        pool = ProcessPool(nodes=num_workers)
        try:
            for i in range(de_num_iterations):
                if self.use_logging:
                    logging.info(f"DE Iteration {i+1}/{de_num_iterations}...")
                de_result = sp.optimize.differential_evolution(
                    func=self.objective_function,
                    bounds=bounds,
                    strategy='best1bin',
                    maxiter=de_max_generations,
                    popsize=de_pop_size,
                    tol=0.01,
                    recombination=0.7,
                    polish=de_polish,
                    disp=True,         # Quieter output, rely on logging
                    workers=pool.map,   # Parallel evaluation of population
                    updating="deferred" # Good for parallel,
                )
                
                global_search_candidates_params.append(de_result.x)
                global_search_candidates_loss.append(de_result.fun)
                
                if de_result.success:
                    logging.warning(f"DE Iteration {i+1} finished. Best Loss: {de_result.fun:.4e}, Params: {de_result.x}")
                else:
                    logging.warning(f"DE Iteration {i+1} did not converge successfully (Message: {de_result.message}).")
                    
                self.total_calls += de_result.nfev
                self.total_itts += de_result.nit
                
                print(f"DE Iteration {i+1} | {de_result.success}")
                print(f"DE resul: {de_result.x} | {de_result.fun}")
                print(f"DE: {de_result}")
                
                print("calls: ", self.total_calls)
                print("itts: ", self.total_itts)
                
        finally:
            pool.close()
            pool.join()
            
        if not global_search_candidates_params:
            logging.error("Differential Evolution failed to produce any candidates.")
            return None, None
        
        # Sort candidates from global search
        sorted_indices = np.argsort(global_search_candidates_loss)
        best_global_candidates_params = [global_search_candidates_params[i] for i in sorted_indices]
        best_global_candidates_loss = [global_search_candidates_loss[i] for i in sorted_indices]
        
        if not local_refinement:
            return best_global_candidates_params[0], best_global_candidates_loss[0]
        
        else:
            if self.use_logging:
                logging.info("Global search (DE) candidates (top 5 or all):")
                for k in range(min(5, len(best_global_candidates_params))):
                    logging.info(f"  Rank {k+1}: Loss={best_global_candidates_loss[k]:.4e}, Params={best_global_candidates_params[k]}")
                    
            # Initialize overall best with the best from DE
            overall_best_params = np.copy(best_global_candidates_params[0])
            overall_best_loss = best_global_candidates_loss[0]
            
            ###* Local Search Refinement
            num_to_refine = min(top_k_candidates_for_local, len(best_global_candidates_params))
            if self.use_logging:
                logging.info(f"\nStarting local search refinement for the top {num_to_refine} DE candidate(s)...")
                
            for k_candidate_idx in range(num_to_refine):
                current_best_params_for_local = np.copy(best_global_candidates_params[k_candidate_idx])
                current_best_loss_for_local = best_global_candidates_loss[k_candidate_idx]
                
                if self.use_logging:
                    logging.info(f"Refining DE Candidate {k_candidate_idx+1} (Loss: {current_best_loss_for_local:.4e})")
                    
                for attempt in range(local_attempts_per_candidate):
                    # Perturb the starting point for local search
                    param_ranges = bounds_array[:, 1] - bounds_array[:, 0]
                    perturbation = np.random.uniform(-perturbation_factor * param_ranges, perturbation_factor * param_ranges)
                    x0_local = current_best_params_for_local + perturbation
                    x0_local = np.clip(x0_local, bounds_array[:, 0], bounds_array[:, 1]) # Ensure within bounds
                    
                    try:
                        local_result = sp.optimize.minimize(
                            fun=self.objective_function,
                            x0=x0_local,
                            method="L-BFGS-B", # Good for bound-constrained problems
                            bounds=bounds,
                            options={'maxiter': 50, 'ftol': 1e-7, 'gtol': 1e-6} 
                        )
                        if self.use_logging:
                            logging.info(f"  Local Search (L-BFGS-B) from DE candidate {k_candidate_idx+1}, attempt {attempt+1}: "
                                        f"Loss={local_result.fun:.4e}, Success={local_result.success}, Message='{local_result.message}'")
                            
                        if local_result.success and local_result.fun < current_best_loss_for_local:
                            current_best_loss_for_local = local_result.fun
                            current_best_params_for_local = local_result.x
                            # Update overall best if this local search found a better solution
                            if current_best_loss_for_local < overall_best_loss:
                                overall_best_loss = current_best_loss_for_local
                                overall_best_params = np.copy(current_best_params_for_local)
                                logging.info(f"    New overall best found! Loss: {overall_best_loss:.4e}, Params: {overall_best_params}")
                                
                    except Exception as e:
                        logging.error(f"  Local search attempt failed with exception: {e}", exc_info=True)
                
                if self.use_logging:
                    logging.info(f"  Finished refinement for DE Candidate {k_candidate_idx+1}. Best loss for this branch: {current_best_loss_for_local:.4e}")
                    
            if self.use_logging:
                logging.info(f"\nHybrid optimization finished.")
                logging.info(f"Overall Best Loss: {overall_best_loss:.4e}")
                logging.info(f"Overall Best Parameters: {overall_best_params}")
                logging.info(f"Total objective function calls: {self.functional_calls}")
                
            return overall_best_params, overall_best_loss
        
        
        
    def hybrid_optimization_searchV2(self, config: Dict[str, Any], num_workers: int = 9, local_refinement: bool = False) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Performs a hybrid optimization: global search with Differential Evolution,
        followed by local search refinement on the best candidates.
        """
        bounds = config.get("bounds")
        de_num_iterations = config.get("de_num_iterations", 1) # How many separate DE runs
        de_pop_size = config.get("de_population_size", 15) # Typical DE parameter
        
        top_k = 30
        
        if bounds is None:
            logging.error("Bounds must be provided in the configuration.")
            return None, None

        bounds_array = np.array(bounds)
        if bounds_array.ndim != 2 or bounds_array.shape[1] != 2:
            logging.error("Bounds must be a list of (min, max) tuples.")
            return None, None

        global_search_candidates_params: List[np.ndarray] = []
        global_search_candidates_loss: List[float] = []
        
        self.total_calls = 0
        self.total_itts = 0

        ####* Global Search: Differential Evolution 
        if self.use_logging:
            logging.info(f"Starting Differential Evolution with {de_num_iterations} iteration(s), "
                        ", using {num_workers} worker(s).")
        
        pool = ProcessPool(nodes=num_workers)
        
        result_global = sp.optimize.differential_evolution(
                    func=self.objective_function,
                    bounds=bounds,
                    strategy='randtobest1bin',
                    maxiter=105,
                    popsize=de_pop_size,
                    tol=0.01,
                    recombination=0.7,
                    polish=False,
                    disp=True,         
                    workers=pool.map,   
                    updating="deferred"
        )
        population = result_global.population
        population_energies = result_global.population_energies
        
        
        ###* Local refinement
        top_k_mod = int(0.3*population.shape[0]) if int(0.1*population.shape[0]) < top_k else top_k
        print("top_k_mod: ", top_k_mod)
        best_idx = np.argsort(population_energies)[:top_k_mod]
        best_population = population[best_idx]
        best_energies = population_energies[best_idx]
        refined = []
        
        
        local_steps = 25
        def _refine(x0):
            res = sp.optimize.minimize(self.objective_function, x0,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={
                        'maxiter': local_steps,
                        'maxfev': local_steps,
                        'xatol': 1e-8,
                        'fatol': 1e-8
                    }
            )
            print("res: ", res.fun, res.x, x0)
            return res.x
        
        
        refined = pool.map(_refine, best_population)
        
        result_reseed = sp.optimize.differential_evolution(self.objective_function, bounds,
                    strategy='best1bin', init=refined, polish=False, 
                    disp=True, maxiter=60,
                    workers=pool.map,   # Parallel evaluation of population
                    updating="deferred" # Good for parallel,
            )
        
        pool.close()
        
        return result_global, result_reseed
