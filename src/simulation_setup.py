import os
import scipy as sp
import numpy as np
import src.Simulator as sim_system

# --- 1. Global Constants ---
const_dict = {
    "F0": 1.5e15,           # cm^-2
    "S0": 3e13,             # cm^-2
    "R": 0.00831442,        # kJ/mol*K
    "kBoltz": 1.380649e-23, # J/K
}


# --- 2. Initial State ---
initial_state_dict = {
    'O_F': 0.1, 'O2_F': 0.1, 'O_S': 0.1, 'Vdb_S': 0.1, 
    'Odb_S': 0.1, 'CO_F': 0.1, 'CO2_F': 0.1, 'CO_S': 0.1, 
    'COdb_S': 0.0
}


# --- 3. Helper Functions ---
def compute_flux(const_dict, exp_dict, specie, molar_mass):
    den = exp_dict.get(specie, 0.0)
    v_th = np.sqrt((8.0 * const_dict['R'] * 1000 * exp_dict['Tnw']) / (molar_mass * np.pi))
    flux = 0.25 * v_th * den * 100
    return flux


def get_interpolator():
    # EavgMB data extracted from the Booth et al. 2019 paper
    p_data_exp = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.5]
    EavgMB_data = [1.04, 0.91, 0.87, 0.83, 0.77, 0.5, 0.001]
    return sp.interpolate.interp1d(p_data_exp, EavgMB_data, kind='linear', fill_value=0.001, bounds_error=False)


# --- 4. Factory Function to Create Simulator ---
def create_common_simulator(base_path, reactions_file="reactions/reactionsCompleteV2.json", data_file="Experimental_data_CO_O_merged.hdf5"):
    """
    Creates and returns the Simulator object with standard settings.
    base_path: The root directory where 'reactions' and 'Buffer_Data' folders are located.
    """
    
    const_dict = {
        "F0": 1.5e15,           # cm^-2
        "S0": 3e13,             # cm^-2
        "R": 0.00831442,        # kJ/mol*K
        "kBoltz": 1.380649e-23, # J/K
    }


    # --- 2. Initial State ---
    initial_state_dict = {
        'O_F': 0.1, 'O2_F': 0.1, 'O_S': 0.1, 'Vdb_S': 0.1, 
        'Odb_S': 0.1, 'CO_F': 0.1, 'CO2_F': 0.1, 'CO_S': 0.1, 
        'COdb_S': 0.0
    }
        
    # 1. Setup Paths
    reactions_file = os.path.join(base_path, reactions_file)
    output_folder_path = os.path.join(base_path, "Buffer_Data")
    exp_file = os.path.join(output_folder_path, data_file)
    
    print("Data Buffer: ", exp_file)

    # 2. Setup Transformations
    interpolator = get_interpolator()
    
    transformations_exp = {
        'Tw':       lambda const_dict, exp_dict: exp_dict['Tw'] + 273.15,
        'fluxO' :   lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict, 'O', 0.016),
        'fluxO2' :  lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict, 'O2', 0.032),
        'fluxO3' :  lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict, 'O3', 0.048),
        'fluxC':    lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict, 'C', 0.012),
        'fluxCO':   lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict, 'CO', 0.028),
        'fluxCO2':  lambda const_dict, exp_dict: compute_flux(const_dict, exp_dict, 'CO2', 0.048),
        'EavgMB':   lambda const_dict, exp_dict: interpolator(exp_dict['pressure']).item(),
        'Ion':      lambda const_dict, exp_dict: 1e14 * exp_dict["current"]
    }

    # 3. Instantiate Simulator
    sim = sim_system.Simulator(
        reactions_file, 
        const_dict, 
        exp_file, 
        initial_state_dict, 
        transformations_exp=transformations_exp
    )
    
    return const_dict, sim