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
            
            frac_solutions_arr, rate_constants_arr, gammas_results_arr = self.simulator.solve_all_conditions(self.exp_data_arr, current_reactions_list, solver_type=solver_type)
            
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
            
            return frac_solutions_arr, rate_constants_arr, gammas_results_arr_processed, gammas_sum_arr
        except Exception as e:
            logging.error(f"Error during simulation run for optimization: {e}", exc_info=True)
            return None, None, None, None
    
    
    def solve_simulations_updated(self, params: np.ndarray) ->  Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        
        param_update_instructions = self.func_new_model_dict(params)
        current_reactions_list = self._update_reactions_list_for_params(self._base_reactions_list, param_update_instructions)
        
        frac_solutions_arr, _, gammas_results_arr, gammas_sum_arr = self._run_simulation_with_updated_reactions(self.exp_data_arr, current_reactions_list, solver_type="fixed_point")
        
        return frac_solutions_arr, gammas_results_arr, gammas_sum_arr, self.gamma_exp_data
    
    
    
    def objective_function(self, params: np.ndarray) -> float:
        
        param_update_instructions = self.func_new_model_dict(params)
        current_reactions_list = self._update_reactions_list_for_params(self._base_reactions_list, param_update_instructions)
        
        _, _, _, gammas_simulated_sum = self._run_simulation_with_updated_reactions(self.exp_data_arr, current_reactions_list, solver_type="fixed_point")
        
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
    
    
    
    def objective_function_diff(self, params: np.ndarray) -> float:
        
        param_update_instructions = self.func_new_model_dict(params)
        current_reactions_list = self._update_reactions_list_for_params(self._base_reactions_list, param_update_instructions)
        
        frac_solutions_arr, rate_constants_arr, gammas_results_arr, gammas_simulated_sum = self._run_simulation_with_updated_reactions(self.exp_data_arr, current_reactions_list, solver_type="fixed_point")
        
        if gammas_simulated_sum is None or np.all(np.isnan(gammas_simulated_sum)):
            logging.warning(f"Call #{current_call_number}: Simulation failed or returned all NaN gammas. Loss set to infinity.")
            return np.inf # Or a very large number if np.inf is problematic for optimizer
        
        if len(self.gamma_exp_data) != len(gammas_simulated_sum):
            logging.error(f"Call #{current_call_number}: Mismatch in length between experimental gamma ({len(self.gamma_exp_data)}) "
                        f"and simulated gamma sum ({len(gammas_simulated_sum)}). Returning Inf loss.")
            return np.inf
        
        
        valid_indices = ~np.isnan(gammas_simulated_sum) & ~np.isnan(self.gamma_exp_data)
        if not np.any(valid_indices):
            logging.warning(f"No valid (non-NaN) gamma pairs for loss calculation. Returning Inf loss.")
            return np.inf
        
        loss = self.loss_function(self.gamma_exp_data[valid_indices], gammas_simulated_sum[valid_indices])
        
        if self.use_logging:
            log_msg = f"Loss = {loss:.4e}"
            logging.info(log_msg)
        
        return loss, frac_solutions_arr, rate_constants_arr, gammas_results_arr, gammas_simulated_sum



    def objective_function_diff_full(self, params: np.ndarray) -> float:
        
        param_update_instructions = self.func_new_model_dict(params)
        current_reactions_list = self._update_reactions_list_for_params(self._base_reactions_list, param_update_instructions)
        
        frac_solutions_arr, rate_constants_arr, gammas_results_arr, gammas_simulated_sum = self._run_simulation_with_updated_reactions(self.exp_data_arr, current_reactions_list, solver_type="fixed_point")
        
        if gammas_simulated_sum is None or np.all(np.isnan(gammas_simulated_sum)):
            logging.warning(f"Call #{current_call_number}: Simulation failed or returned all NaN gammas. Loss set to infinity.")
            return np.inf # Or a very large number if np.inf is problematic for optimizer
        
        if len(self.gamma_exp_data) != len(gammas_simulated_sum):
            logging.error(f"Call #{current_call_number}: Mismatch in length between experimental gamma ({len(self.gamma_exp_data)}) "
                        f"and simulated gamma sum ({len(gammas_simulated_sum)}). Returning Inf loss.")
            return np.inf
        
        
        valid_indices = ~np.isnan(gammas_simulated_sum) & ~np.isnan(self.gamma_exp_data)
        if not np.any(valid_indices):
            logging.warning(f"No valid (non-NaN) gamma pairs for loss calculation. Returning Inf loss.")
            return np.inf
        
        residuals = (gammas_simulated_sum[valid_indices] - self.gamma_exp_data[valid_indices]) / self.gamma_exp_data[valid_indices]
        loss = np.mean(np.pow(residuals,2))
        
        # loss = self.loss_function(self.gamma_exp_data[valid_indices], gammas_simulated_sum[valid_indices])
        
        if self.use_logging:
            log_msg = f"Loss = {loss:.4e}"
            logging.info(log_msg)
        
        return loss, residuals, frac_solutions_arr, rate_constants_arr, gammas_results_arr, gammas_simulated_sum
