import numpy as np
import scipy as sp
import os
import logging 
import copy
from pathos.multiprocessing import ProcessPool
from typing import List, Dict, Callable, Any, Tuple, Optional, Union


class ErrorPropagator():
    """
    Propagates uncertainties from model parameters or experimental conditions
    through the simulation to assess their impact on output gamma values
    """
    
    def __init__(self,
                simulator_object: Any,
                func_uncertainty_model_dict: Optional[Callable[[np.ndarray], List[Dict[str, Any]]]] = None,
                transformations_exp: Optional[Dict[str, Callable]] = None,
                use_logging: bool = True,
                seed: int = 42):

        np.random.seed(seed)
        self.simulator = simulator_object

        if not isinstance(self.simulator.exp_data_arr, np.ndarray) or \
            self.simulator.exp_data_arr.dtype != object:
            logging.warning("self.simulator.exp_data_arr is not in the expected format (numpy array of dicts). "
                            "Experimental error propagation might behave unexpectedly.")
        self._base_exp_data_arr_template: np.ndarray = copy.deepcopy(self.simulator.exp_data_arr)

        self.transformations_exp: Optional[Dict[str, Callable]] = transformations_exp
        self.const_dict: Dict[str, float] = self.simulator.const_dict

        self.func_uncertainty_model_dict: Optional[Callable[[np.ndarray], List[Dict[str, Any]]]] = func_uncertainty_model_dict
        
        self.use_logging: bool = use_logging

        ###* Store an initial deep copy of the reaction list from the simulator
        if 'reactions_list' not in self.simulator.output_parser:
            raise ValueError("Simulator's output_parser does not contain 'reactions_list'.")
        self._base_reactions_list: List[Dict[str, Any]] = copy.deepcopy(self.simulator.output_parser['reactions_list'])



    def _update_reactions_list_for_model_params(self, base_reactions: List[Dict[str, Any]],
                                                param_update_instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ###* Applies model parameter updates to a deep copy of a base reactions list.
        ###* This is similar to the method in the Optimizer class)
        
        updated_reactions_list = copy.deepcopy(base_reactions)
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
                if len(model_dict_updates) == 1:
                    param_key_to_match = next(iter(model_dict_updates))
                    indices_to_update = [idx for idx, reaction in enumerate(updated_reactions_list)
                                        if param_key_to_match in reaction.get('model_dict', {})]
                    if not indices_to_update:
                        logging.warning(f"No reactions found with parameter key '{param_key_to_match}'. Instruction: {instruction}")
                        continue
                else:
                    raise ValueError(f"Ambiguous update instruction (id and rate are None, model_dict has multiple keys): {instruction}")
                
            for idx in indices_to_update:
                if 'model_dict' not in updated_reactions_list[idx] or updated_reactions_list[idx]['model_dict'] is None:
                    updated_reactions_list[idx]['model_dict'] = {}
                updated_reactions_list[idx]['model_dict'].update(model_dict_updates)
        return updated_reactions_list



    def _generate_truncated_gaussian_sample(self,
                                            mean: float = 0.0,
                                            sigma: float = 0.1, # Error as fraction of mean
                                            min_value: float = 0.0,
                                            max_value: float = np.inf,
                                            min_absolute_sigma: float = 1e-8,
                                            size: Optional[int] = None
                                            ) -> Union[float, np.ndarray]:
        
        if sigma <= 0:
            sigma = min_absolute_sigma
            logging.warning(f"Calculated sigma was <=0 for mean {mean}, error {relative_error}. Using min_absolute_sigma {min_absolute_sigma}.")
        
        sigma = max(sigma, min_absolute_sigma)
        
        # Parameters for truncnorm: a, b are (min_val - loc) / scale, (max_val - loc) / scale
        a, b = (min_value - mean) / sigma, (max_value - mean) / sigma
        distribution = sp.stats.truncnorm(a, b, loc=mean, scale=sigma)
        return distribution.rvs(size=size)



    def _run_simulation_for_one_sample_model(self,
                                            model_params_sample: np.ndarray,
                                            base_reactions_list_template: List[Dict[str, Any]],
                                            exp_data_for_sim: np.ndarray
                                            ) -> Optional[np.ndarray]:
        try:
            param_update_instructions = self.func_uncertainty_model_dict(model_params_sample)
            current_reactions_list = self._update_reactions_list_for_model_params(base_reactions_list_template, param_update_instructions)
            
            
            print("One Model: ", current_reactions_list)
            print("One Exp: ", exp_data_for_sim)
            
            _, gammas_results_arr = self.simulator.solve_all_conditions(exp_data_for_sim, current_reactions_list, solver_type="fixed_point")
            
            if gammas_results_arr is None: return None
            
            gammas_sum_list = []
            for gamma_dict in gammas_results_arr:
                if isinstance(gamma_dict, dict) and gamma_dict:
                    valid_values = [v for v in gamma_dict.values() if isinstance(v, (int, float)) and not np.isnan(v)]
                    gammas_sum_list.append(sum(valid_values) if valid_values else np.nan)
                else:
                    gammas_sum_list.append(np.nan)
            return np.array(gammas_sum_list)
        except Exception as e:
            logging.error(f"Error in _run_simulation_for_one_sample_model: {e}", exc_info=True)
            num_experiments = len(exp_data_for_sim)
            return np.full(num_experiments, np.nan) # Return NaNs of correct shape on failure



    def _run_simulation_for_one_sample_exp(self, exp_params_sample_all_exps: np.ndarray, base_exp_data_template: np.ndarray,
                                    reactions_list_for_sim: List[Dict[str, Any]], exp_param_names_to_vary: List[str]) -> Optional[np.ndarray]:
        try:
            ###* Create a deep copy of the experimental data template for this specific MC sample
            current_exp_data_arr = copy.deepcopy(base_exp_data_template)
            num_experiments = len(current_exp_data_arr)

            if exp_params_sample_all_exps.shape[0] != num_experiments or exp_params_sample_all_exps.shape[1] != len(exp_param_names_to_vary):
                logging.error("Shape mismatch for exp_params_sample_all_exps.")
                return np.full(num_experiments, np.nan)
            
            for i_exp in range(num_experiments): # For each experiment in this MC sample
                for k_param_idx, param_name in enumerate(exp_param_names_to_vary):
                    current_exp_data_arr[i_exp][param_name] = exp_params_sample_all_exps[i_exp, k_param_idx]
                    
                ###* Apply transformations after all params for this experiment are updated
                if self.transformations_exp:
                    for trans_key, trans_func in self.transformations_exp.items():
                        try:
                            current_exp_data_arr[i_exp][trans_key] = trans_func(self.const_dict, current_exp_data_arr[i_exp])
                        except Exception as e:
                            logging.error(f"Error applying transformation '{trans_key}' for exp {i_exp}: {e}")
                            current_exp_data_arr[i_exp][trans_key] = np.nan
            
            print("One Exp: ", current_exp_data_arr)
            print("One Model: ", reactions_list_for_sim)
            
            _, gammas_results_arr = self.simulator.solve_all_conditions(current_exp_data_arr, reactions_list_for_sim, solver_type="fixed_point")
            
            if gammas_results_arr is None: return None
            
            gammas_sum_list = []
            for gamma_dict in gammas_results_arr:
                if isinstance(gamma_dict, dict) and gamma_dict:
                    valid_values = [v for v in gamma_dict.values() if isinstance(v, (int, float)) and not np.isnan(v)]
                    gammas_sum_list.append(sum(valid_values) if valid_values else np.nan)
                else:
                    gammas_sum_list.append(np.nan)
            return np.array(gammas_sum_list)
        
        except Exception as e:
            logging.error(f"Error in _run_simulation_for_one_sample_exp: {e}", exc_info=True)
            num_experiments_fallback = len(base_exp_data_template)
            return np.full(num_experiments_fallback, np.nan)
        
        
        
    def propagate_model_parameter_uncertainty(self,
                                model_param_uncertainty_specs: List[Tuple[float, float, float, float]], # (mean, sigma, min_val, max_val)
                                num_samples_N: int = 3,
                                num_workers: int = 1
                                ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        
        if self.func_uncertainty_model_dict is None:
            logging.error("`func_uncertainty_model_dict` is not provided. Cannot propagate model parameter uncertainty.")
            raise ValueError("`func_uncertainty_model_dict` must be provided for model error propagation.")
        if not model_param_uncertainty_specs:
            logging.warning("`model_param_uncertainty_specs` is empty. No model parameters to vary.")
            return np.array([]), np.array([])

        num_model_params_to_vary = len(model_param_uncertainty_specs)
        sampled_model_params_array = np.zeros((num_samples_N, num_model_params_to_vary), dtype=float)

        for i_param, spec in enumerate(model_param_uncertainty_specs):
            mean, sigma, min_v, max_v = spec
            sampled_model_params_array[:, i_param] = self._generate_truncated_gaussian_sample(
                mean=mean, sigma=sigma, min_value=min_v, max_value=max_v, size=num_samples_N
            )
        
        if self.use_logging:
            logging.info(f"Propagating model parameter uncertainty with {num_samples_N} samples using {num_workers} worker(s).")
            
        # Use a deep copy of the base experimental data (read-only for these simulations)
        # and the base reactions list (template for modifications by each worker)
        exp_data_for_sim_template = self._base_exp_data_arr_template
        base_reactions_template = self._base_reactions_list

        pool_args = [(sample, base_reactions_template, exp_data_for_sim_template) for sample in sampled_model_params_array]
        results_list = []
        try:
            with ProcessPool(nodes=num_workers) as pool:
                results_list = pool.map(lambda p_args: self._run_simulation_for_one_sample_model(*p_args), pool_args)
        except Exception as e:
            logging.error(f"Error during parallel processing for model uncertainty: {e}", exc_info=True)
            return sampled_model_params_array, None
        
        resulting_gamma_sums_array = np.array([res if res is not None else np.full(len(exp_data_for_sim_template), np.nan) for res in results_list])
        return sampled_model_params_array, resulting_gamma_sums_array


    def propagate_experimental_condition_uncertainty(self,
                                    exp_condition_uncertainty_specs: List[Tuple[str, float, float, float]], # (name, sigma, min_val, max_val)
                                    num_samples_N: int = 3,
                                    num_workers: int = 1
                                    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        
        if not exp_condition_uncertainty_specs:
            logging.warning("`exp_condition_uncertainty_specs` is empty. No experimental conditions to vary.")
            return np.array([]), np.array([])

        base_exp_data_template = self._base_exp_data_arr_template
        num_experiments = len(base_exp_data_template)
        num_exp_params_to_vary = len(exp_condition_uncertainty_specs)

        sampled_exp_params_array = np.zeros((num_samples_N, num_experiments, num_exp_params_to_vary), dtype=float)
        exp_param_names_to_vary = [spec[0] for spec in exp_condition_uncertainty_specs]

        for i_exp in range(num_experiments): # For each original experiment
            for k_param_idx, spec in enumerate(exp_condition_uncertainty_specs):
                param_name, sigma, min_v, max_v = spec
                
                mean_val = base_exp_data_template[i_exp].get(param_name)
                if mean_val is None:
                    logging.error(f"Experimental parameter '{param_name}' not found in base data for experiment {i_exp}.")
                    raise ValueError(f"Parameter '{param_name}' missing in experiment {i_exp} data.")
                if not isinstance(mean_val, (int, float)):
                    logging.error(f"Experimental parameter '{param_name}' for experiment {i_exp} is not numeric: {mean_val}.")
                    raise ValueError(f"Parameter '{param_name}' in experiment {i_exp} is not numeric.")

                sampled_exp_params_array[:, i_exp, k_param_idx] = self._generate_truncated_gaussian_sample(
                    mean=mean_val, sigma=sigma, min_value=min_v, max_value=max_v, size=num_samples_N
                )
        
        if self.use_logging:
            logging.info(f"Propagating experimental condition uncertainty with {num_samples_N} samples using {num_workers} worker(s).")
            
        # Use a deep copy of the base reactions list (read-only for these simulations)
        reactions_list_for_sim_template = self._base_reactions_list
        
        # Each element of pool_args is one full set of perturbed exp conditions for ALL experiments
        pool_args = [(sampled_exp_params_array[i_N, :, :], base_exp_data_template, reactions_list_for_sim_template, exp_param_names_to_vary)
                    for i_N in range(num_samples_N)]
        
        results_list = []
        try:
            with ProcessPool(nodes=num_workers) as pool:
                results_list = pool.map(lambda p_args: self._run_simulation_for_one_sample_exp(*p_args), pool_args)
        except Exception as e:
            logging.error(f"Error during parallel processing for experimental uncertainty: {e}", exc_info=True)
            return sampled_exp_params_array, None
        
        resulting_gamma_sums_array = np.array([res if res is not None else np.full(num_experiments, np.nan) for res in results_list])
        return sampled_exp_params_array, resulting_gamma_sums_array