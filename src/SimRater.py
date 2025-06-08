import numpy as np
import torch
import os
import h5py
import logging
from typing import Dict, List, Callable, Any, Optional, Tuple

import src.rates as rates_module

# logger = logging.getLogger(__name__)


class SimulatorRates():
    """
    Manages experimental data loading from HDF5 buffer and computes reaction rates 
    for each experimental condition based on defined rates
    """
    
    def __init__(self, const_dict: Dict[str, float], exp_data_file: str, 
                output_parser: Dict[str, Any]): # Output from SimParser
        
        self.const_dict: Dict[str, float] = const_dict
        self.exp_data_file: str = exp_data_file
        self.output_parser: Dict[str, Any] = output_parser
        self.available_rates_functions: Dict[str, Callable] = {}
    
    
    
    def prepare_experimental_data(self, transformations_exp: Optional[Dict[str, Callable]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        
        if not os.path.exists(self.exp_data_file):
            logging.error(f"Experimental data HDF5 file not found: {self.exp_data_file}")
            return np.array([]), np.array([])
        
        logging.info("Preparing experimental data...")
        n_rows: Optional[int] = None
        data_dict_h5: Dict[str, np.ndarray] = {}
        
        with h5py.File(self.exp_data_file, "r") as file:
            keys_hdf5 = list(file.keys())
            for key in keys_hdf5:
                data_dict_h5[key] = file[key][:]
                if n_rows is None and key != 'File_Data': # Use any data column to get n_rows
                    n_rows = len(data_dict_h5[key])
                    
        if n_rows is None: # No data rows found
            logging.warning("No data rows found in HDF5 file.")
            return np.array([], dtype=object), data_dict_h5.get("gamma_exp", np.array([]))
        
        exp_conditions_list: List[Dict[str, Any]] = []
        for i in range(n_rows):
            current_exp_condition: Dict[str, Any] = {}
            for key in keys_hdf5:
                if key == 'File_Data' or key == 'gamma_exp': # 'gamma_exp' handled separately
                    continue
                current_exp_condition[key] = data_dict_h5[key][i]
                
            if transformations_exp:
                for trans_key, trans_func in transformations_exp.items():
                    try:
                        current_exp_condition[trans_key] = trans_func(self.const_dict, current_exp_condition)
                    except Exception as e:
                        logging.error(f"Error applying transformation '{trans_key}' for exp condition {i}: {e}")
                        current_exp_condition[trans_key] = np.nan

            exp_conditions_list.append(current_exp_condition)

        # Using np.array of dicts (dtype=object) can have performance implications. But so far the number of experimental conditions is small ~100 
        exp_vec_arr = np.array(exp_conditions_list, dtype=object)

        gamma_exp_vec = data_dict_h5.get("gamma_exp", None)
        if gamma_exp_vec is None:
            logging.info("No 'gamma_exp' dataset found in HDF5 file.")
        else:
            logging.info(f"Loaded 'gamma_exp' with shape {gamma_exp_vec.shape}")
        
        logging.info(f"Prepared {len(exp_vec_arr)} experimental conditions.")
        return exp_vec_arr, gamma_exp_vec
    
    
    
    def preload_rates_functions(self) -> None:
        logging.info("Preloading rate functions...")
        reactions = self.output_parser.get('reactions_list', [])
        if not reactions:
            logging.warning("No reactions found in output_parser. Cannot preload rate functions.")
            return
        
        different_rates_labels: Set[str] = set(reaction['rate'] for reaction in reactions)

        self.available_rates_functions = {}
        for name_label in different_rates_labels:
            if hasattr(rates_module, name_label):
                self.available_rates_functions[name_label] = getattr(rates_module, name_label)
                logging.debug(f"Successfully loaded rate function: {name_label}")
            else:
                logging.error(f"Rate function '{name_label}' is not defined in 'rates_module'.")
                raise ValueError(f"Rate function '{name_label}' not defined in rates_module.")
        logging.info("Rate functions preloaded.")
    
    
    
    # def compute_rates_simulation(self, exp_vec_arr: np.ndarray, reactions_list: List[Dict[str, Any]]) -> np.ndarray:
        
    #     if exp_vec_arr.size == 0:
    #         logging.warning("exp_vec_arr is empty. Cannot compute rates.")
    #         return np.array([[]])
    #     if not reactions_list:
    #         logging.warning("reactions_list is empty. Cannot compute rates.")
    #         return np.zeros((len(exp_vec_arr), 0))
        
    #     num_experiments = len(exp_vec_arr)
    #     num_reactions = len(reactions_list)
    #     rates_calculation_arr = np.zeros((num_experiments, num_reactions), dtype=float)
        
    #     logging.info(f"Computing rate constants for {num_experiments} experiments and {num_reactions} reactions...")
        
    #     for i in range(num_experiments):
    #         current_exp_condition = exp_vec_arr[i] # This is a dict
    #         for j in range(num_reactions):
    #             reaction = reactions_list[j]
    #             model_dict_params = reaction.get('model_dict', {})
                
    #             rate_params_for_func = model_dict_params.copy()
    #             rate_params_for_func['gas_specie'] = reaction.get('gas_specie') # e.g. "O2"
                
    #             rate_label = reaction['rate']
    #             if rate_label not in self.available_rates_functions:
    #                 logging.error(f"Rate function for label '{rate_label}' not found in preloaded functions. Reaction ID: {reaction.get('id')}")
    #                 rates_calculation_arr[i, j] = np.nan
    #                 continue
                
    #             rate_function = self.available_rates_functions[rate_label]
    #             try:
    #                 rate_value = rate_function(self.const_dict, current_exp_condition, rate_params_for_func)
    #                 rates_calculation_arr[i, j] = rate_value
    #             except Exception as e:
    #                 logging.error(f"Error computing rate for reaction '{reaction.get('id')}' (label: {rate_label}), exp condition {i}: {e}")
    #                 rates_calculation_arr[i, j] = np.nan # Mark as problematic

    #     logging.info("Rate constants computation complete.")
    #     return rates_calculation_arr
    
    
    def compute_rates_simulation(self, exp_vec_arr: np.ndarray, reactions_list: List[Dict[str, Any]], flag_arr = "numpy") -> np.ndarray:
        
        if exp_vec_arr.size == 0:
            logging.warning("exp_vec_arr is empty. Cannot compute rates.")
            return np.array([[]])
        if not reactions_list:
            logging.warning("reactions_list is empty. Cannot compute rates.")
            return np.zeros((len(exp_vec_arr), 0))
        
        num_experiments = len(exp_vec_arr)
        num_reactions = len(reactions_list)
        if flag_arr == "numpy":
            rates_calculation_arr = np.zeros((num_experiments, num_reactions), dtype=float)
            logging.info(f"Computing rate constants for {num_experiments} experiments and {num_reactions} reactions...")
            
            for i in range(num_experiments):
                current_exp_condition = exp_vec_arr[i] # This is a dict
                for j in range(num_reactions):
                    reaction = reactions_list[j]
                    model_dict_params = reaction.get('model_dict', {})
                    
                    rate_params_for_func = model_dict_params.copy()
                    rate_params_for_func['gas_specie'] = reaction.get('gas_specie') # e.g. "O2"
                    
                    rate_label = reaction['rate']
                    if rate_label not in self.available_rates_functions:
                        logging.error(f"Rate function for label '{rate_label}' not found in preloaded functions. Reaction ID: {reaction.get('id')}")
                        rates_calculation_arr[i,j] = np.nan
                        continue
                    
                    rate_function = self.available_rates_functions[rate_label]
                    try:
                        rate_value = rate_function(self.const_dict, current_exp_condition, rate_params_for_func)
                        rates_calculation_arr[i,j] = rate_value
                    except Exception as e:
                        logging.error(f"Error computing rate for reaction '{reaction.get('id')}' (label: {rate_label}), exp condition {i}: {e}")
                        rates_calculation_arr[i,j] = np.nan # Mark as problematic

            logging.info("Rate constants computation complete.")
            return rates_calculation_arr
        
        elif flag_arr == "torch": 
            
            rates_calculation_arr = []
            for i in range(num_experiments):
                rates_aux = torch.zeros(num_reactions, dtype=float)
                current_exp_condition = exp_vec_arr[i] # This is a dict
                for j in range(num_reactions):
                    reaction = reactions_list[j]
                    model_dict_params = reaction.get('model_dict', {})
                    
                    rate_params_for_func = model_dict_params.copy()
                    rate_params_for_func['gas_specie'] = reaction.get('gas_specie') # e.g. "O2"
                    
                    rate_label = reaction['rate']
                    if rate_label not in self.available_rates_functions:
                        logging.error(f"Rate function for label '{rate_label}' not found in preloaded functions. Reaction ID: {reaction.get('id')}")
                        rates_aux[j] = torch.nan
                        continue
                    
                    rate_function = self.available_rates_functions[rate_label]
                    try:
                        rate_value = rate_function(self.const_dict, current_exp_condition, rate_params_for_func)
                        rates_aux[j] = rate_value
                    except Exception as e:
                        logging.error(f"Error computing rate for reaction '{reaction.get('id')}' (label: {rate_label}), exp condition {i}: {e}")
                        rates_aux[j] = torch.nan # Mark as problematic
                rates_calculation_arr.append(rates_aux)
            
            rates_calculation_arr = torch.stack(rates_calculation_arr)
            
            logging.info("Rate constants computation complete.")
            return rates_calculation_arr
        
        else: 
            raise ValueError(f"Sim Rater - Comoute Rates Simulation - Invalid flag_arr value: {flag_arr}")
            