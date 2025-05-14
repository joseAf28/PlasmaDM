import DMsimulator as DMsim
import numpy as np
import scipy as sp
import os
import h5py
import time



if __name__ == "__main__":
    
    ###! Free parameters
    
    # # ###* Version Paper
    # A = 0.01634
    # B = 1.67e-4
    
    ###* Upgraded Version Paper V2
    
    A = 2.74332762e-03
    B = 6.23703570e-04 
    E_nu_d = 1.68496853e+01
    
    Ealpha = 3.26673523e+03 
    Emin = 2.70000000e+00 
    ED_db = 1.00000000e+01
    
    nu_d_func = lambda Tw, A, B, E_nu_d: 1e15 * (A + B * np.exp(E_nu_d /(const_dict["R"] * Tw)))
    
    
    file_input_data = "Experimental_data_Paper.hdf5"
    output_file = "results_TDPaperWithLowPressureOptimizedCheck.hdf5"
    
    # output_file = "results_TDPaperUpgradedLowV5.hdf5"
    
    output_file = os.path.join("simulations", output_file)
    
    #########* Constants and fixed parameters of the model
    start_time = time.time()
    const_dict = {
        "F0": 1.5e15,           # cm^-2
        "S0": 3e13,             # cm^-2
        
        "R": 0.00831442,        # kJ/mol*K
        "kBoltz": 1.380649e-23, # J/K
    }
    
    
    # #########* Steric factors reactions
    # steric_dict = {
    #     ###* Atomic oxygen
    #     "SF_O_F": 1.0, "SF_O_S": 1.0, "SF_O_SO": 1.0, "SF_O_FO": 1.0,
    #     "SF_FO_S": 1.0, "SF_FO_SO": 1.0, "SF_FO_FO": 1.0, "SF_FO": 1.0,
        
    #     ###* Molecular oxygen
    #     "SF_O2_F": 1.0, "SF_O2_FO": 1.0, "SF_O2_FO2": 1.0, "SF_O_FO2": 1.0,
    #     "SF_FO2_FO": 1.0, "SF_FO_FO2": 1.0, "SF_FO2": 1.0,
        
    #     ###* Metastable species
    #     "SF_O2fast_SO": 0.0, "SF_Ofast_SO": 0.0, "SF_O2fast_S": 0.0,  "SF_Ofast_S": 0.0,
    #     "SF_Ofast_Sdb": 0.0, "SF_Ofast_SOdb": 0.0, "SF_O2fast_Sdb": 0.0, "SF_O2fast_SOdb": 0.0,
    #     "SF_O_Sdb": 0.0, "SF_O_SOdb": 0.0, "SF_FO_SOdb": 0.0, "SF_FO_Sdb": 0.0,
    # }
    
    
    ##! Include the metastable species
    
    steric_dict = {
        ###* Atomic oxygen
        "SF_O_F": 1.0, "SF_O_S": 1.0, "SF_O_SO": 1.0, "SF_O_FO": 1.0,
        "SF_FO_S": 1.0, "SF_FO_SO": 1.0, "SF_FO_FO": 1.0, "SF_FO": 1.0,
        
        ###* Molecular oxygen
        "SF_O2_F": 1.0, "SF_O2_FO": 1.0, "SF_O2_FO2": 1.0, "SF_O_FO2": 1.0,
        "SF_FO2_FO": 1.0, "SF_FO_FO2": 1.0, "SF_FO2": 1.0,
        
        ###* Metastable species
        "SF_O2fast_SO": 1.0, "SF_Ofast_SO": 1.0, "SF_O2fast_S": 1.0,  "SF_Ofast_S": 1.0,
        "SF_Ofast_Sdb": 1.0, "SF_Ofast_SOdb": 1.0, "SF_O2fast_Sdb": 1.0, "SF_O2fast_SOdb": 1.0,
        "SF_O_Sdb": 1.0, "SF_O_SOdb": 1.0, "SF_FO_SOdb": 1.0, "SF_FO_Sdb": 1.0,
    }
    
    
    #########* Initialize the system
    init_conditions = [0.1, 0.1, 1.0, 0.0, 0.0]
    system = DMsim.SurfaceKineticsSimulator(const_dict, steric_dict, file_input_data, init_conditions)
    
    
    #########* Energy barriers and transition rates
    energy_dict = { # kJ/mol and s^-1
        "E_O_F": 0.0, "E_O_S": 0.0, "E_O_SO": 15.0, "E_O_FO": 0.0, 
        "E_FO_SO": 20.0, "E_FO_FO":0.0, "E_di_O": 15.0, "E_de_O": 30.0,
        
        "E_O2_F": 0.0, "E_O2_FO": 0.0, "E_O2_FO2": 0.0, "E_O_FO2": 0.0, 
        "E_FO2_FO": 0.0, "E_FO_FO2": 0.0, "E_di_O2": 15.0, "E_de_O2": 17.5,
        
        "E_O2fast_SO": 0.0, "E_O2fast_S": 0.0, "E_O2fast_SOdb": 0.0, "E_O2fast_Sdb": 0.0, "E_Ofast_Sdb": 0.0,
        "E_Ofast_SOdb": 0.0, "E_O_Sdb": 0.0, "E_O_SOdb": 0.0, "E_F_SOdb": 0.0, "E_FO_SOdb": 0.0,
        "ED_db": 14.999,
        
        "nu_D": 1.0e13, "nu_d": 1.0e15,
        
        "Emin": 2.90, # eV
        "Ealpha": 3400.0, # K
        
        # "Emin": 2.60, # eV
        # "Ealpha":  3.23383702e+03, # K
    }
    
    energy_dict["Emin"] = Emin
    energy_dict["Ealpha"] = Ealpha
    energy_dict["ED_db"] = ED_db
    
    ###* Multiple Cases with the input data from .xlsx files
    data_dict, recProbExp_vec = system.prepare_data()    
    nb_simulations = len(data_dict)
    
    steady_state_sol_vec = []
    gammas_results = []
    results_names = []
    
    input_names = ["Tnw", "Tw", "O_den", "pressure", "current", "FluxIon", "EavgMB"]
    exp_data = []
    
    new_energy_dict = energy_dict.copy()
    
    print("Number of simulations: ", nb_simulations)
    for i in range(nb_simulations):
        
        exp_dict = data_dict[i]
        new_energy_dict["nu_d"] = nu_d_func(exp_dict["Tw"], A, B, E_nu_d)
        
        ### Solve the system
        steady_state_sol, results_gammas, results_names_aux, _ = system.solve_system(exp_dict, new_energy_dict, solver="odeint")
        
        steady_state_sol_vec.append(steady_state_sol)
        gammas_results.append(results_gammas)
        if i == 0:
            results_names = results_names_aux
        
        exp_data.append([exp_dict[key] for key in input_names])
        
    
    steady_state_sol_vec = np.array(steady_state_sol_vec)
    gammas_results = np.array(gammas_results)
    exp_data = np.array(exp_data)
    
    
    time_end = time.time()
    print("Time: ", time_end - start_time)
    print("mean time", (time_end - start_time) / nb_simulations)
    
    ####* Save the results in a hdf5 file
    with h5py.File(output_file, "w") as file:
        file.create_dataset("ss_sol", data=steady_state_sol_vec)
        file.create_dataset("exp_data", data=exp_data)
        file.create_dataset("gammas_data", data=gammas_results)
        file.create_dataset("gammas_names", data=np.array(results_names, dtype=h5py.string_dtype(encoding='utf-8')))
        file.create_dataset("exp_names", data=np.array(input_names, dtype=h5py.string_dtype(encoding='utf-8')))
        file.create_dataset("recProbExp", data=recProbExp_vec)
    file.close()
