import numpy as np
import scipy as sp
import h5py
import time


class SurfaceKineticsSimulator():
    
    def __init__(self, const_dict, SF_dict_tuple, file_data="Experimental_data_TD.hdf5", 
                init_conditions=[0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05], timeSpace=np.linspace(0, 1_00, 5_00)):
        
        self.const_dict = const_dict
        self.init_conditions = init_conditions
        SF_O_dict, SF_CO_dict = SF_dict_tuple
        
        self.SF_dict = SF_O_dict.copy()
        self.SF_dict.update(SF_CO_dict)
        
        ##! 1e-5 minimal value to ensure that the model reactions are activate 
        self.act_SF_O = sum(list(SF_O_dict.values())) > 1e-5
        self.act_SF_CO = sum(list(SF_CO_dict.values())) > 1e-5
        
        if not self.act_SF_O:
            self.init_conditions = [0.0] * 5 * self.init_conditions[5:]
        if not self.act_SF_CO:
            self.init_conditions = self.init_conditions[:5] + [0.0] * 4
        
        print(self.init_conditions)
        
        self.timeSpace = timeSpace
        self.file_exp_data = file_data
    
    
    def prepare_data(self):
        #### the .hdf5 file was obtained by merging the data from the .xlsx files
        
        ### Read the experimental data from the hdf5 file
        data_dict = {}
        with h5py.File(self.file_exp_data, "r") as file:
            keys = list(file.keys())
            
            for key in keys:
                data_dict[key] = file[key][:]
        
        file.close()
        
        ####* Prepara the data for the simulation
        Pressure_vec = data_dict["Pressure"]
        Current_vec = data_dict["Current"]
        
        Tnw_vec = data_dict["TnwExp"]
        Tw_vec = data_dict["Twall"]
        
        zeros = np.zeros_like(Tnw_vec)
        Oden_vec  = data_dict.get("OmeanExp", zeros.copy())
        COden_vec = data_dict.get("COmeanExp",zeros.copy())
        Nexp_vec  = data_dict.get("NExp", zeros.copy())
        
        Recomb_prob_exp_vec = data_dict["recProbExp"]
        FluxIon_vec = 1e14 * Current_vec        
        
        ####? EavgMB data
        p_data_exp = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.5]
        EavgMB_data = [1.04, 0.91, 0.87, 0.83, 0.77, 0.5, 0.001]
        interpolator = sp.interpolate.interp1d(p_data_exp, EavgMB_data, kind='linear', fill_value=0.001, bounds_error=False)
        
        exp_vec = []
        for i in range(len(Tnw_vec)):
            exp_vec.append({
                "pressure": Pressure_vec[i], "current": Current_vec[i],
                "Tnw": Tnw_vec[i], "Tw": Tw_vec[i], 
                "O_den": Oden_vec[i], "CO_den": COden_vec[i], "N_den": Nexp_vec[i],
                "FluxIon": FluxIon_vec[i], "EavgMB": interpolator(Pressure_vec[i]).item()
            })
        exp_vec_arr = np.array(exp_vec, dtype=object)
        
        return exp_vec_arr, Recomb_prob_exp_vec
    
    
    def MB_func(self, E, TavgMB):
        return (2.0/ np.sqrt(np.pi) * ((1.0 / (self.const_dict["kBoltz"] * TavgMB)) ** 1.5) * np.exp(-E / (self.const_dict["kBoltz"] * TavgMB)) * (E**0.5))
    
    
    def remaining_flux(self, exp_dict, mass):
        ###! what do with this: flux of O2 or flux of CO2 or a mixture
        if exp_dict["N_den"] > 0:
            Nden = exp_dict["N_den"]
        else:
            Nden = exp_dict["pressure"] * 133.322368 * 1e-6 / (self.const_dict["kBoltz"] * exp_dict["Tnw"])
        
        remaining_den = Nden - exp_dict["CO_den"] - exp_dict["O_den"]
        FluxRemaining = 0.25 * np.sqrt((8.0 * self.const_dict["R"] * 1000 * exp_dict["Tnw"])/(mass *  np.pi)) * remaining_den * 100
        
        # print("Nden", Nden, "CO_den", exp_dict["CO_den"], "O_den", exp_dict["O_den"])
        
        return FluxRemaining


    def rates_O_definition(self, exp_dict, energy_dict):
        
        ### Constants and Experimental Parameters
        F0, S0 = self.const_dict["F0"], self.const_dict["S0"]
        surface = F0 + S0
        
        R = self.const_dict["R"]
        kBoltz = self.const_dict["kBoltz"]
        
        Tw, Tnw = exp_dict["Tw"], exp_dict["Tnw"]
        EavgMB = exp_dict["EavgMB"]     
        Emin, Ealpha = energy_dict["Emin"], energy_dict["Ealpha"]
        
        ###* Energy barriers with the Oxygen Atomic
        E_O_F, E_O_S, E_O_SO, E_O_FO, E_FO_SO, E_FO_FO  = energy_dict["E_O_F"], energy_dict["E_O_S"], energy_dict["E_O_SO"], energy_dict["E_O_FO"], energy_dict["E_FO_SO"], energy_dict["E_FO_FO"]
        E_di_O, E_de_O = energy_dict["E_di_O"], energy_dict["E_de_O"]
        nu_D_oxy1, nu_d_oxy1 = energy_dict["nu_D_O"], energy_dict["nu_d_O"]
        
        ###* Energy barriers with the Oxygen Molecule
        E_O2_F, E_O2_FO, E_O2_FO2, E_O_FO2, E_FO2_FO = energy_dict["E_O2_F"], energy_dict["E_O2_FO"], energy_dict["E_O2_FO2"], energy_dict["E_O_FO2"], energy_dict["E_FO2_FO"]
        E_di_O2, E_de_O2, E_FO_FO2 = energy_dict["E_di_O2"], energy_dict["E_de_O2"], energy_dict["E_FO_FO2"]
        nu_D_oxy2, nu_d_oxy2 = energy_dict["nu_D_CO"], energy_dict["nu_d_CO"]
        
        ###* Energy barriers with Oxygen Metastable
        E_O2fast_SO, E_O2fast_S = energy_dict["E_O2fast_SO"], energy_dict["E_O2fast_S"]
        E_O2fast_SOdb, E_O2fast_Sdb, E_Ofast_SOdb, E_Ofast_Sdb = energy_dict["E_O2fast_SOdb"], energy_dict["E_O2fast_Sdb"], energy_dict["E_Ofast_SOdb"], energy_dict["E_Ofast_Sdb"]
        E_O_Sdb, E_O_SOdb, E_FO_SOdb = energy_dict["E_O_Sdb"], energy_dict["E_O_Sdb"], energy_dict["E_FO_SOdb"]
        ED_db = energy_dict["ED_db"]
        
        
        ###* SF reactions
        SF_O_F, SF_O_S, SF_O_SO, SF_O_FO = self.SF_dict["SF_O_F"], self.SF_dict["SF_O_S"], self.SF_dict["SF_O_SO"], self.SF_dict["SF_O_FO"]
        SF_FO_SO, SF_FO_S, SF_FO_FO, SF_FO = self.SF_dict["SF_FO_SO"], self.SF_dict["SF_FO_S"], self.SF_dict["SF_FO_FO"], self.SF_dict["SF_FO"]
        
        SF_O2_F, SF_O2_FO, SF_O2_FO2, SF_O_FO2 = self.SF_dict["SF_O2_F"], self.SF_dict["SF_O2_FO"], self.SF_dict["SF_O2_FO2"], self.SF_dict["SF_O_FO2"]
        SF_FO2_FO, SF_FO2, SF_FO_FO2  = self.SF_dict["SF_FO2_FO"], self.SF_dict["SF_FO2"], self.SF_dict["SF_FO_FO2"]
        
        SF_O2fast_SO, SF_Ofast_SO, SF_O2fast_S, SF_Ofast_S = self.SF_dict["SF_O2fast_SO"], self.SF_dict["SF_Ofast_SO"], self.SF_dict["SF_O2fast_S"], self.SF_dict["SF_Ofast_S"]
        SF_Ofast_Sdb, SF_Ofast_SOdb, SF_O2fast_Sdb, SF_O2fast_SOdb = self.SF_dict["SF_Ofast_Sdb"], self.SF_dict["SF_Ofast_SOdb"], self.SF_dict["SF_O2fast_Sdb"], self.SF_dict["SF_O2fast_SOdb"] 
        SF_O_Sdb, SF_O_SOdb = self.SF_dict["SF_O_Sdb"], self.SF_dict["SF_O_SOdb"]
        SF_FO_SOdb, SF_FO_Sdb = self.SF_dict["SF_FO_SOdb"], self.SF_dict["SF_FO_Sdb"]
        
        
        ### Auxiliar quantities
        FluxO = 0.25 * np.sqrt((8.0 * R * 1000 * Tnw)/(0.016 *  np.pi)) * exp_dict["O_den"] * 100
        FluxIon = exp_dict["FluxIon"]
        
        ###! 0.032 only O2 for now
        FluxRemaining = self.remaining_flux(exp_dict, 0.032)
        
        ###? change later
        FluxCO2 = 0.0 * FluxRemaining
        FluxO2 = 1.0 * FluxRemaining
        
        flux_dict = {
            "FluxO": FluxO, "FluxO2": FluxO2, "FluxIon": FluxIon,
        }
        
        print(flux_dict)
        
        
        TavgMB = EavgMB * 1.60218e-19 / kBoltz
        Emin = Emin * 1.602 * 1e-19
        IntMB = sp.integrate.quad(self.MB_func, Emin, 40*Emin, args=(TavgMB))
        Ealpha = Ealpha * kBoltz
        Intalpha = sp.integrate.quad(self.MB_func, Ealpha, 40*Ealpha, args=(Tnw))
        
        
        ###! Transition rates - Oxigen atomic
        r1 = SF_O_F * FluxO / surface * np.exp(-E_O_F / (R * Tnw))
        r2 = SF_FO * nu_d_oxy1 * np.exp(-E_de_O / (R * Tw))
        r3 = SF_O_S * FluxO / surface * np.exp(-E_O_S / (R * Tnw))
        r4 = SF_O_FO * FluxO / surface * np.exp(-E_O_FO / (R * Tnw))
        r5 = SF_FO_S * 0.75 * nu_D_oxy1 * np.exp(-E_di_O / (R * Tw))
        r6 = SF_O_SO * FluxO / surface * np.exp(-E_O_SO / (R * Tnw))
        r7 = SF_FO_SO * nu_D_oxy1 * np.exp(-E_di_O / (R * Tw)) * np.exp(-E_FO_SO / (R * Tw))
        r8 = SF_FO_FO * nu_D_oxy1 * np.exp(-E_di_O / (R * Tw)) * np.exp(-E_FO_FO / (R * Tw))
        
        rates_O_dict = {
            "r1": r1, "r2": r2, "r3": r3, "r4": r4,
            "r5": r5, "r6": r6,"r7": r7, "r8": r8,
        }
        
        print("O: ", rates_O_dict)
        
        
        ###! Transition rates - Oxygen molecular
        r9 = SF_O2_F * FluxO2 / surface * np.exp(-E_O2_F / (R * Tnw))       # same
        r10 = SF_FO2 * nu_d_oxy2 * np.exp(-E_de_O2 / (R * Tw))              # same 
        r11 = SF_O2_FO * FluxO2 / surface * np.exp(-E_O2_FO / (R * Tnw))    # same
        r12 = SF_O2_FO2 * FluxO2 / surface * np.exp(-E_O2_FO2 / (R * Tnw))  # 
        ###* skip r13 for now: O_4(g) -> O_2(g) + O_2(g)
        r14 = SF_O_FO2 * FluxO / surface * np.exp(-E_O_FO2 / (R * Tnw))    # r13 paper Viegas et al. 2024
        r15a = SF_FO2_FO * nu_D_oxy2 * np.exp(-E_di_O2 / (R * Tw)) * np.exp(-E_FO2_FO / (R * Tw)) # r15 paper Viegas et al. 2024
        r15b = 0.0 * SF_FO_FO2 * nu_D_oxy2 * np.exp(-E_di_O2 / (R * Tw)) * np.exp(-E_FO_FO2 / (R * Tw)) # r16 paper Viegas et al. 2024
        
        rates_O2_dict = {
            "r9": r9, "r10": r10, "r11": r11, "r12": r12, 
            "r14": r14, "r15a": r15a, "r15b": r15b
        }
        
        
        print("O2: ", rates_O2_dict)
        
        ###! Transition rates - Metastable Surface Kinetics
        r16 = SF_O2fast_S * exp_dict["FluxIon"] / surface * IntMB[0] * np.exp(- E_O2fast_S / (R * Tnw))
        ###* correction r16 for the wall temperature: f(Tw), where f(Tw = 50*C) = 1
        r16 = r16 * (
        (np.exp(-E_di_O / (R * Tw)) + np.exp(-E_de_O / (R * Tw)))
        / (np.exp(-E_di_O / (R * 323.15)) + np.exp(-E_de_O / (R * 323.15)))
        )
        r18 = SF_O2fast_SO * exp_dict["FluxIon"] / surface * IntMB[0] * np.exp(- E_O2fast_SO / (R * Tnw))
        r18 = r18 * (
        (np.exp(-E_di_O / (R * Tw)) + np.exp(-E_de_O / (R * Tw)))
        / (np.exp(-E_di_O / (R * 323.15)) + np.exp(-E_de_O / (R * 323.15)))
        )
        
        ###* and the r17 and r19 are assumed to be included in the r16 and r18
        r20 = SF_O_Sdb * FluxO * (1.0 - Intalpha[0]) / surface * np.exp(-E_O_Sdb / (R * Tnw))
        r21 = SF_Ofast_Sdb * FluxO * Intalpha[0] / surface * np.exp(-E_Ofast_Sdb / (R * Tnw))
        
        r22 = SF_O2fast_Sdb * FluxO2 * Intalpha[0] / surface * np.exp(-E_O2fast_Sdb / (R * Tnw))
        r23 = SF_O2fast_SOdb * FluxO2 * Intalpha[0] / surface * np.exp(-E_O2fast_SOdb / (R * Tnw))
        
        r24 = SF_Ofast_SOdb * FluxO * Intalpha[0] / surface * np.exp(-E_Ofast_SOdb / (R * Tnw))
        r25 = SF_O_SOdb * FluxO * (1.0 - Intalpha[0]) / surface * np.exp(- E_O_SOdb / (R * Tnw))
        
        r26 = SF_FO_Sdb * 0.75 * nu_D_oxy1 * np.exp(-E_di_O / (R * Tw)) * np.exp(-ED_db / (R * Tw))
        r27 = SF_FO_SOdb * 0.75 * nu_D_oxy1 * np.exp(-E_di_O / (R * Tw)) * np.exp(-ED_db / (R * Tw)) * np.exp(-E_FO_SOdb / (R * Tw))
        
        rates_meta_dict = {
            "r16": r16, "r18": r18, "r20": r20, "r21": r21,
            "r22": r22, "r23": r23, "r24": r24, "r25": r25,
            "r26": r26, "r27": r27,
        }
        
        
        print("meta: ", rates_meta_dict)
        
        return (rates_O_dict, rates_O2_dict, rates_meta_dict, flux_dict)


    def rates_CO_definition(self, exp_dict, energy_dict):
        
        ### Constants and Experimental Parameters
        F0, S0 = self.const_dict["F0"], self.const_dict["S0"]
        surface = F0 + S0
        
        R = self.const_dict["R"]
        kBoltz = self.const_dict["kBoltz"]
        
        Tw, Tnw = exp_dict["Tw"], exp_dict["Tnw"]
        EavgMB = exp_dict["EavgMB"]     
        Ealpha = energy_dict["Ealpha"]
        
        EminO2 = energy_dict["Emin"]
        EminCO2 = EminO2 * (2*16.0 /(2*16.0 + 12.0))
        EminCO = EminO2 * (2*16.0 /(16.0 + 12.0))
        EminO = EminO2 * (2.0)
        
        
        ###* Energy barriers with the CO2 specie
        E_CO2_F, E_FCO2 = energy_dict["E_CO2_F"], energy_dict["E_FCO2"]
        
        ###* Energy barriers with the CO specie
        E_CO_F, E_FCO, E_CO_S, E_O_FCO  = energy_dict["E_CO_F"], energy_dict["E_FCO"], energy_dict["E_CO_S"], energy_dict["E_O_FCO"]
        E_CO_FO, E_FCO_FO, E_FCO_S, E_O_SCO = energy_dict["E_CO_FO"], energy_dict["E_FCO_FO"], energy_dict["E_FCO_S"], energy_dict["E_O_SCO"]
        E_CO_SO, E_FO_SCO, E_FCO_SO = energy_dict["E_CO_SO"], energy_dict["E_FO_SCO"], energy_dict["E_FCO_SO"]
        
        ###* Energy barries with Meta CO species
        E_COfast_SO, E_CO2fast_SO, E_Ofast_SCO, E_O2fast_SCO = energy_dict["E_COfast_SO"], energy_dict["E_CO2fast_SO"], energy_dict["E_Ofast_SCO"], energy_dict["E_O2fast_SCO"]
        E_COfast_SCO, E_CO2fast_SCO, E_COfast_S, E_CO2fast_S = energy_dict["E_COfast_SCO"], energy_dict["E_CO2fast_SCO"], energy_dict["E_COfast_S"], energy_dict["E_CO2fast_S"]
        E_CO_Sdb, E_CO_Sdb_2, E_CO2_Sdb, E_CO2_SOdb = energy_dict["E_CO_Sdb"], energy_dict["E_CO_Sdb_2"], energy_dict["E_CO2_Sdb"], energy_dict["E_CO2_SOdb"]
        E_CO_SOdb, E_O_SCOdb, E_CO_SCOdb, E_CO2_SCOdb = energy_dict["E_CO_SOdb"], energy_dict["E_O_SCOdb"], energy_dict["E_CO_SCOdb"], energy_dict["E_CO2_SCOdb"]
        
        E_O2_SCOdb, E_CO_SOdb, E_O_SCOdb, E_FCO_SOdb = energy_dict["E_O2_SCOdb"], energy_dict["E_CO_SOdb"], energy_dict["E_O_SCOdb"], energy_dict["E_FCO_SOdb"]
        E_FCO_SCOdb, E_FO_SCOdb = energy_dict["E_FCO_SCOdb"], energy_dict["E_FO_SCOdb"]
        
        E_di_CO, E_de_CO = energy_dict["E_di_CO"], energy_dict["E_de_CO"]
        
        nu_d = energy_dict["nu_d_CO"]
        nu_D = energy_dict["nu_D_CO"]
        
        ###* SF reactions
        
        SF_CO2_F, SF_FCO2 = self.SF_dict["SF_CO2_F"], self.SF_dict["SF_FCO2"]        
        
        SF_CO_F, SF_FCO, SF_CO_S, SF_O_FCO = self.SF_dict["SF_CO_F"], self.SF_dict["SF_FCO"], self.SF_dict["SF_CO_S"], self.SF_dict["SF_O_FCO"]
        SF_CO_FO, SF_FCO_FO, SF_FCO_S, SF_O_SCO = self.SF_dict["SF_CO_FO"], self.SF_dict["SF_FCO_FO"], self.SF_dict["SF_FCO_S"], self.SF_dict["SF_O_SCO"]
        SF_CO_SO, SF_FO_SCO, SF_FCO_SO = self.SF_dict["SF_CO_SO"], self.SF_dict["SF_FO_SCO"], self.SF_dict["SF_FCO_SO"]
        
        SF_COfast_SO, SF_CO2fast_SO, SF_Ofast_SCO, SF_O2fast_SCO = self.SF_dict["SF_COfast_SO"], self.SF_dict["SF_CO2fast_SO"], self.SF_dict["SF_Ofast_SCO"], self.SF_dict["SF_O2fast_SCO"]
        SF_COfast_SCO, SF_CO2fast_SCO, SF_COfast_S, SF_CO2fast_S = self.SF_dict["SF_COfast_SCO"], self.SF_dict["SF_CO2fast_SCO"], self.SF_dict["SF_COfast_S"], self.SF_dict["SF_CO2fast_S"]
        SF_CO_Sdb, SF_CO_Sdb_2, SF_CO2_Sdb, SF_CO2_SOdb = self.SF_dict["SF_CO_Sdb"], self.SF_dict["SF_CO_Sdb_2"], self.SF_dict["SF_CO2_Sdb"], self.SF_dict["SF_CO2_SOdb"]
        SF_CO_SOdb,SF_O_SCOdb,SF_CO_SCOdb, SF_CO2_SCOdb = self.SF_dict["SF_CO_SOdb"], self.SF_dict["SF_O_SCOdb"], self.SF_dict["SF_CO_SCOdb"], self.SF_dict["SF_CO2_SCOdb"]
        
        SF_O2_SCOdb, SF_CO_SOdb, SF_O_SCOdb, SF_FCO_SOdb = self.SF_dict["SF_O2_SCOdb"], self.SF_dict["SF_CO_SOdb"], self.SF_dict["SF_O_SCOdb"], self.SF_dict["SF_FCO_SOdb"]
        SF_FCO_SCOdb, SF_FO_SCOdb = self.SF_dict["SF_FCO_SCOdb"], self.SF_dict["SF_FO_SCOdb"]
        
        
        ### Auxiliar quantities
        FluxO = 0.25 * np.sqrt((8.0 * R * 1000 * Tnw)/(0.016 *  np.pi)) * exp_dict["O_den"] * 100
        FluxCO = 0.25 * np.sqrt((8.0 * R * 1000 * Tnw)/(0.028 *  np.pi)) * exp_dict["CO_den"] * 100
        FluxIon = exp_dict["FluxIon"]
        
        ###! 0.032 only O2 for now
        FluxRemaining = self.remaining_flux(exp_dict, 0.032)
        
        ###? change later
        FluxCO2 = 0.0 * FluxRemaining
        FluxO2 = 1.0 * FluxRemaining
        
        
        flux_dict = {
            "FluxCO": FluxCO, "FluxCO2": FluxCO2, "FluxIon": FluxIon,
        }
        
        TavgMB = EavgMB * 1.60218e-19 / kBoltz
        EminO2 = EminO2 * 1.602 * 1e-19
        EminCO2 = EminCO2 * 1.602 * 1e-19
        EminCO = EminCO * 1.602 * 1e-19
        EminO = EminO * 1.602 * 1e-19
        IntMBO2 = sp.integrate.quad(self.MB_func, EminO2, 40*EminO2, args=(TavgMB))
        IntMBCO2 = sp.integrate.quad(self.MB_func, EminCO2, 50*EminCO2, args=(TavgMB))
        IntMBCO = sp.integrate.quad(self.MB_func, EminCO, 50*EminCO, args=(TavgMB))
        IntMBO = sp.integrate.quad(self.MB_func, EminO, 50*EminO, args=(TavgMB))
        
        Ealpha = Ealpha * kBoltz
        Intalpha = sp.integrate.quad(self.MB_func, Ealpha, 40*Ealpha, args=(Tnw))
        
        ratio_meta = (
            (np.exp(-E_di_CO / (R * Tw)) + np.exp(-E_de_CO / (R * Tw)))
            / (np.exp(-E_di_CO / (R * 323.15)) + np.exp(-E_de_CO / (R * 323.15)))
        )
        
        ###! Transition rates - CO2
        r1 = SF_CO2_F * FluxCO2 / surface * np.exp(-E_CO2_F / (R * Tnw))
        r2 = SF_FCO2 * nu_d * np.exp(-E_FCO2 / (R * Tw))
        
        rates_CO2_dict = {
            "r1": r1, "r2": r2,
        }
        
        
        ###! Transition rates - CO
        r3 = SF_CO_F * FluxCO / surface * np.exp(-E_CO_F / (R * Tnw))
        r4 = SF_FCO * nu_d * np.exp(-E_FCO / (R * Tw))
        r5 = SF_CO_S * FluxCO / surface * np.exp(-E_CO_S / (R * Tnw))
        r6 = SF_O_FCO * FluxO / surface * np.exp(-E_O_FCO / (R * Tnw))
        r7 = SF_CO_FO * FluxCO / surface * np.exp(-E_CO_FO / (R * Tnw))
        r8 = SF_FCO_FO * nu_D * np.exp(-E_FCO_FO / (R * Tw))
        r9 = SF_FCO_S * nu_D * np.exp(-E_FCO_S / (R * Tw))
        r10 = SF_O_SCO * FluxO / surface * np.exp(-E_O_SCO / (R * Tnw))
        r11 = SF_CO_SO * FluxCO / surface * np.exp(-E_CO_SO / (R * Tnw))
        r12 = SF_FO_SCO * nu_D * np.exp(-E_FO_SCO / (R * Tw))
        r13 = SF_FCO_SO * nu_D * np.exp(-E_FCO_SO / (R * Tw))
        
        rates_CO_dict = {
            "r3": r3, "r4": r4, "r5": r5, "r6": r6,
            "r7": r7, "r8": r8, "r9": r9, "r10": r10,
            "r11": r11, "r12": r12, "r13": r13
        }
        
        ####! Transition rates -  Metastable Surface Kinetics
        ###* considering the O2^+ as the dominant ion
        
        r14 = SF_COfast_SO * FluxIon * ratio_meta / surface * IntMBCO[0] * np.exp(- E_COfast_SO / (R * Tnw))
        r15 = SF_CO2fast_SO * FluxIon * ratio_meta / surface * IntMBCO2[0] * np.exp(- E_CO2fast_SO / (R * Tnw))
        r16 = SF_Ofast_SCO * FluxIon * ratio_meta / surface * IntMBO[0] * np.exp(- E_Ofast_SCO / (R * Tnw))
        r17 = SF_O2fast_SCO * FluxIon * ratio_meta / surface * IntMBO2[0] * np.exp(- E_O2fast_SCO / (R * Tnw))
        
        r18 = SF_COfast_SCO * FluxIon * ratio_meta / surface * IntMBCO[0] * np.exp(- E_COfast_SCO / (R * Tnw))
        r19 = SF_CO2fast_SCO * FluxIon * ratio_meta / surface * IntMBCO[0] * np.exp(- E_CO2fast_SCO / (R * Tnw))
        r20 = SF_COfast_S * FluxIon * ratio_meta / surface * IntMBCO[0] * np.exp(- E_COfast_S / (R * Tnw))
        r21 = SF_CO2fast_S * FluxIon * ratio_meta / surface * IntMBCO2[0] * np.exp(- E_CO2fast_S / (R * Tnw))
        
        r22 = SF_CO_Sdb * FluxCO / surface * (1.0 - Intalpha[0]) * np.exp(-E_CO_Sdb / (R * Tnw))
        r23 = SF_CO_Sdb_2 * FluxCO * Intalpha[0] / surface * np.exp(-E_CO_Sdb_2 / (R * Tnw))
        r24 = SF_CO2_Sdb * FluxCO2 * Intalpha[0] / surface * np.exp(-E_CO2_Sdb / (R * Tnw))
        r25 = SF_CO2_SOdb * FluxCO2 * Intalpha[0] / surface * np.exp(-E_CO2_SOdb / (R * Tnw))
        
        r26 = SF_CO_SOdb * FluxCO * Intalpha[0] / surface * np.exp(-E_CO_SOdb / (R * Tnw))
        r27 = SF_O_SCOdb * FluxO * Intalpha[0] / surface * np.exp(-E_O_SCOdb / (R * Tnw))
        r28 = SF_CO_SCOdb * FluxCO * Intalpha[0] / surface * np.exp(-E_CO_SCOdb / (R * Tnw))
        r29 = SF_CO2_SCOdb * FluxCO2 * Intalpha[0] / surface * np.exp(-E_CO2_SCOdb / (R * Tnw))
        
        r30 = SF_O2_SCOdb * FluxO2 * Intalpha[0] / surface * np.exp(-E_O2_SCOdb / (R * Tnw))
        r31 = SF_CO_SOdb * FluxCO * Intalpha[0] / surface * np.exp(-E_CO_SOdb / (R * Tnw))
        r32 = SF_O_SCOdb * FluxO * Intalpha[0] / surface * np.exp(-E_O_SCOdb / (R * Tnw))
        r33 = SF_FCO_SOdb * nu_D * np.exp(-E_FCO_SOdb / (R * Tw))
        
        r34 = SF_FCO_SCOdb * nu_D * np.exp(-E_FCO_SCOdb / (R * Tw))
        r35 = SF_FO_SCOdb * nu_D * np.exp(-E_FO_SCOdb / (R * Tw))
        
        # print(E_COfast_SO, E_CO2fast_SO, E_Ofast_SCO, E_O2fast_SCO, E_COfast_SCO, E_CO2fast_SCO, E_COfast_S, E_CO2fast_S,\
        #     E_CO_Sdb, E_CO_Sdb_2, E_CO2_Sdb, E_CO2_SOdb, E_CO_SOdb, E_O_SCOdb, E_CO_SCOdb, E_CO2_SCOdb,\
        #     E_O2_SCOdb, E_CO_SOdb, E_O_SCOdb, E_FCO_SOdb, E_FCO_SCOdb, E_FO_SCOdb)
        
        # print(SF_COfast_SO, SF_CO2fast_SO, SF_Ofast_SCO, SF_O2fast_SCO, SF_COfast_SCO, SF_CO2fast_SCO, SF_COfast_S, SF_CO2fast_S,\
        #     SF_CO_Sdb, SF_CO_Sdb_2, SF_CO2_Sdb, SF_CO2_SOdb, SF_CO_SOdb, SF_O_SCOdb, SF_CO_SCOdb, SF_CO2_SCOdb,\
        #     SF_O2_SCOdb, SF_CO_SOdb, SF_O_SCOdb, SF_FCO_SOdb, SF_FCO_SCOdb, SF_FO_SCOdb)
        
        
        rates_metastable_dict = {
            "r14": r14, "r15": r15, "r16": r16, "r17": r17,
            "r18": r18, "r19": r19, "r20": r20, "r21": r21,
            "r22": r22, "r23": r23, "r24": r24, "r25": r25,
            "r26": r26, "r27": r27, "r28": r28, "r29": r29,
            "r30": r30, "r31": r31, "r32": r32, "r33": r33,
            "r34": r34, "r35": r35,
        }
        
        return (rates_CO2_dict, rates_CO_dict, rates_metastable_dict, flux_dict)
    
    
    #### Physical system model
    def system_ode(self, X, t, rates_tuple):
        
        S0, F0 = self.const_dict["S0"], self.const_dict["F0"]
        rates_O_tuple, rates_CO_tuple = rates_tuple
        
        frac_Of, frac_O2f, frac_Os, frac_Osdb, frac_Svdb, frac_COf, frac_CO2f, frac_COs, frac_COsdb = X
        frac_Fv = 1.0 - frac_Of - frac_O2f - frac_COf - frac_CO2f
        frac_Sv = 1.0 - frac_Os - frac_COs -  frac_Svdb - frac_Osdb - frac_COsdb

        rates_O_dict, rates_O2_dict, rates_meta_O_dict, _ = rates_O_tuple
        r1, r2, r3, r4, r5, r6, r7, r8 = rates_O_dict.values()
        r9, r10, r11, r12, r14, r15a, r15b = rates_O2_dict.values()
        r16, r18, r20, r21, r22, r23, r24, r25, r26, r27 = rates_meta_O_dict.values()
        
        rates_CO2_dict, rates_CO_dict, rates_meta_CO_dict, _ = rates_CO_tuple
        r1_CO, r2_CO = rates_CO2_dict.values()
        r3_CO, r4_CO, r5_CO, r6_CO, r7_CO, r8_CO, r9_CO, r10_CO, r11_CO, r12_CO, r13_CO = rates_CO_dict.values()
        r14_CO, r15_CO, r16_CO, r17_CO, r18_CO, r19_CO, r20_CO, r21_CO, r22_CO, r23_CO, r24_CO, \
        r25_CO, r26_CO, r27_CO, r28_CO, r29_CO, r30_CO, r31_CO, r32_CO, r33_CO, r34_CO, r35_CO = rates_meta_CO_dict.values()
        
        
        ###* Oxygen reactions
        frac_Of_equation = [
            + r1 * frac_Fv
            - r2 * frac_Of
            - r4 * frac_Of
            - r5 * frac_Of * (S0/F0) * frac_Sv
            - r7 * frac_Of * (S0/F0) * frac_Os
            - 2.0 * r8 * frac_Of * frac_Of
            
            - r11 * frac_Of
            - (r15a + r15b) * frac_O2f * frac_Of
            - r26 * frac_Of * (S0/F0) * frac_Svdb 
            - r27 * frac_Of * (S0/F0) * frac_Osdb
                
            - r7_CO * frac_Of
            - r8_CO * frac_COf * frac_Of
            
            - r35_CO * frac_Of * (S0/F0) * frac_Svdb
        ]
        
        frac_O2f_equation = [
            + r9 * frac_Fv
            - r10 * frac_O2f
            - r12 * frac_O2f
            - r14 * frac_O2f
            - (r15a + r15b) * frac_O2f * frac_Of
            ]
        
        frac_Os_equation = [
            + r3 * frac_Sv
            - r6 * frac_Os
            + r5 * frac_Of * frac_Sv
            - r7 * frac_Of * frac_Os
            - r16 * frac_Os
            
            - r11_CO * frac_Os
            - r13_CO * frac_COf * frac_Os
            
            - r14_CO * frac_Os
            - r15_CO * frac_Os
        ]
        
        frac_Osdb_equation = [
            + r20 * frac_Svdb
            - r23 * frac_Osdb
            - r24 * frac_Osdb
            - r25 * frac_Osdb 
            + r26 * frac_Of * frac_Svdb
            - r27 * frac_Of * frac_Osdb
            
            - r25_CO * frac_Osdb
            - r26_CO * frac_Osdb
            - r31_CO * frac_Osdb
            - r34_CO * frac_COf * frac_Osdb
            ]
        
        frac_Svdb_equation = [
            + r16 * frac_Sv
            + r18 * frac_Os
            - r20 * frac_Svdb
            - r21 * frac_Svdb
            - r22 * frac_Svdb #-
            + r25 * frac_Osdb
            - r26 * frac_Of * frac_Svdb
            + r27 * frac_Of * frac_Osdb
            
            + r14_CO * frac_Os
            + r15_CO * frac_Os
            + r16_CO * frac_COs
            + r17_CO * frac_COs
            + r18_CO * frac_COs
            + r19_CO * frac_COs
            + r20_CO * frac_Sv
            + r21_CO * frac_Sv
            - r22_CO * frac_Svdb
            - r23_CO * frac_Svdb
            - r24_CO * frac_Svdb
            + r31_CO * frac_Osdb
            + r32_CO * frac_COsdb
            - r33_CO * frac_Svdb * frac_COf
            + r34_CO * frac_COf * frac_Osdb
            + r35_CO * frac_Of * frac_Osdb
        ]
        
        ###* COx reactions
        frac_COf_equation = [
            + r3_CO * frac_Fv
            - r4_CO * frac_COf
            - r6_CO * frac_COf 
            - r8_CO * frac_COf * frac_Of
            - r9_CO * frac_COf * (S0/F0) * frac_Sv
            - r13_CO * frac_COf * (S0/F0) * frac_Os
            
            - r33_CO * frac_COf * (S0/F0) * frac_Svdb
            - r34_CO * frac_COf * (S0/F0) * frac_Osdb
        ]
        
        frac_CO2f_equation = [
            + r1_CO * frac_Fv
            - r2_CO * frac_CO2f
        ]
        
        frac_COs_equation = [
            + r5_CO * frac_Sv
            + r9_CO * frac_COf * frac_Sv
            - r10_CO * frac_COs
            - r12_CO * frac_Of * frac_COs
            
            - r16_CO * frac_COs
            - r17_CO * frac_COs
            - r18_CO * frac_COs
            - r19_CO * frac_COs
        ]
            
        frac_COsdb_equation = [
            + r22_CO * frac_Svdb
            - r27_CO * frac_COsdb
            - r28_CO * frac_COsdb
            - r29_CO * frac_COsdb
            - r30_CO * frac_COsdb
            - r32_CO * frac_COsdb
            + r33_CO * frac_COf * frac_Svdb
            - r35_CO * frac_Of * frac_COsdb
        ]
        
        
        func = [frac_Of_equation[0], frac_O2f_equation[0], frac_Os_equation[0], frac_Osdb_equation[0], 
            frac_Svdb_equation[0], frac_COf_equation[0], frac_CO2f_equation[0], frac_COs_equation[0], 
            frac_COsdb_equation[0]]
        
        return func    
    
    
    ### Solvers
    def solve_ode(self, rates_dict_tuple, method="stiff"):
        
        if method == "simple":
            
            def odeWrapper(X, t):
                return self.system_ode(X, t, rates_dict_tuple)
            
            timeSpace = np.linspace(0, 10.0, 1_00)
            solution = sp.integrate.odeint(
                func=odeWrapper,
                y0=self.init_conditions,
                t=timeSpace
            )
        elif method == "stiff":
            sol = sp.integrate.solve_ivp(
                fun=lambda t, X: self.system_ode(X, t, rates_dict_tuple),
                t_span=(self.timeSpace[0], self.timeSpace[-1]),
                y0=self.init_conditions,
                method="BDF",
                t_eval=self.timeSpace,
                atol=1e-5, rtol=1e-5
            )
            solution = sol.y.T
            
        else:
            raise ValueError("Invalid method - choose between 'simple' and 'stiff'")
        
        sucess = self.solution_check(solution, rates_dict_tuple)
        
        return solution, sucess
    
    
    def solve_fixed_point(self, rates_dict_tuple, max_time=15):
        
        # short_time = np.linspace(self.timeSpace[0], self.timeSpace[min(100, len(self.timeSpace)-1)], 100)
        short_time = np.linspace(self.timeSpace[0], self.timeSpace[min(30, len(self.timeSpace)-1)], 30)
        
        start_time = time.time()
        events = None
        if max_time is not None:
            def timeout_event(t, X):
                elapsed = time.time() - start_time
                return max_time - elapsed
            
            timeout_event.terminal = True
            timeout_event.direction = -1
            events = [timeout_event]
            
        
        sol_short = sp.integrate.solve_ivp(
            fun=lambda t, X: self.system_ode(X, t, rates_dict_tuple),
            t_span=(short_time[0], short_time[-1]),
            y0=self.init_conditions,
            method="Radau",
            t_eval=short_time,
            atol=1e-5, rtol=1e-5,
            events=events
        )
        
        refined_guess = sol_short.y.T[-1]
        
        ### Attempt to find the fixed point using the refined guess
        try:
            sol = sp.optimize.root(self.system_ode, refined_guess, args=(0, rates_dict_tuple), method="hybr")
            success = self.solution_check(np.atleast_2d(sol.x), rates_dict_tuple)
        except Exception as e:
            print("Fixed point solver failed with message: ", e)
            success = False
            sol = self.init_conditions
        
        return sol, success
    
    
    def solution_check(self, sol, rates_dict_tuple):
        vec_aux = self.system_ode(sol[-1], 0, rates_dict_tuple)
        absolute_error = np.sum(np.abs(vec_aux))
        
        print("frac_Of, frac_O2f, frac_Os, frac_Osdb, frac_Svdb, frac_COf, frac_CO2f, frac_COs, frac_COsdb")
        print(sol[-1])
        print("Absolute error: ", absolute_error)
        
        if absolute_error > 1e-4:
            return False
        else:
            return True
    
    
    ### Recombination Probability Computations
    def compute_gammas(self, frac_solutions, rates_tuple):
        
        S0, F0 = self.const_dict["S0"], self.const_dict["F0"]

        rates_O_tuple, rates_CO_tuple = rates_tuple
        
        frac_solutions = tuple(np.abs(frac_solutions))
        
        frac_Ofss, frac_O2fss, frac_Osss, frac_Osdbss, frac_Svdbss, \
        frac_COfss, frac_CO2fss, frac_COsss, frac_COsdbss = frac_solutions
        
        rates_O_dict, rates_O2_dict, rates_O_meta_dict, flux_dict = rates_O_tuple
        
        r1, r2, r3, r4, r5, r6, r7, r8 = rates_O_dict.values()
        r9, r10, r11, r12, r14, r15a, r15b = rates_O2_dict.values()
        r16, r18, r20, r21, r22, r23, r24, r25, r26, r27 = rates_O_meta_dict.values()
        FluxO, FluxO2, FluxIon = flux_dict.values()
        
        rates_CO2_dict, rates_CO_dict, rates_CO_meta_dict, flux_dict = rates_CO_tuple
        r1_CO, r2_CO = rates_CO2_dict.values()
        r3_CO, r4_CO, r5_CO, r6_CO, r7_CO, r8_CO, r9_CO, r10_CO, r11_CO, r12_CO, r13_CO = rates_CO_dict.values()
        r14_CO, r15_CO, r16_CO, r17_CO, r18_CO, r19_CO, r20_CO, r21_CO, r22_CO, r23_CO, r24_CO, \
        r25_CO, r26_CO, r27_CO, r28_CO, r29_CO, r30_CO, r31_CO, r32_CO, r33_CO, r34_CO, r35_CO = rates_CO_meta_dict.values()
        FluxCO, FluxCO2, _ = flux_dict.values()
        
        
        ###! Atomic Oxygen
        g_O_r4 = 2.0 * r4 * frac_Ofss * F0 / FluxO
        g_O_r6 = 2.0 * r6 * frac_Osss * S0 / FluxO
        g_O_r7 = 2.0 * r7 * frac_Osss * S0 * frac_Ofss / FluxO
        g_O_r8 = 2.0 * r8 * frac_Ofss * F0 * frac_Ofss / FluxO
        
        ###! Molecular Oxygen
        g_O_r11 = r11 * frac_Ofss * F0 / FluxO
        g_O_r12 = r12 * frac_O2fss * F0 / FluxO
        g_O_r14 = r14 * frac_O2fss * F0 / FluxO
        g_O_r15 = (r15a + r15b) * frac_O2fss * F0 * frac_Ofss / FluxO
        
        ###! Metastable Surface Kinetics
        g_O_r23 = 2.0 * r23 * frac_Osdbss * S0 / FluxO
        g_O_r24 = 2.0 * r24 * frac_Osdbss * S0 / FluxO
        g_O_r25 = 2.0 * r25 * frac_Osdbss * S0 / FluxO
        g_O_r27 = 2.0 * r27 * frac_Osdbss * S0 * frac_Ofss / FluxO
        
        ###! CO2 
        ###* empty
        
        ###! CO2
        g_CO_r6 = 2.0 * r6_CO * frac_COfss * F0 / FluxO
        g_CO_r7 = 2.0 * r7_CO * frac_Ofss * F0 / FluxO
        g_CO_r8 = 2.0 * r8_CO * frac_Ofss * F0 * frac_Ofss / FluxO
        g_CO_r10 = 2.0 * r10_CO * frac_COsss * S0 / FluxO
        g_CO_r11 = 2.0 * r11_CO * frac_Osss * S0 / FluxO
        g_CO_r12 = 2.0 * r12_CO * frac_Ofss * S0 * frac_COsss / FluxO
        g_CO_r13 = 2.0 * r13_CO * frac_Osss * S0 * frac_COfss / FluxO
        
        
        ###! CO metastables
        g_CO_r25 = 2.0 * r25_CO * frac_Osdbss * S0 / FluxO
        g_CO_r26 = 2.0 * r26_CO * frac_Osdbss * S0 / FluxO
        g_CO_r27 = 2.0 * r27_CO * frac_COsdbss * S0 / FluxO
        g_CO_r28 = 2.0 * r28_CO * frac_COsdbss * S0 / FluxO
        g_CO_r29 = 2.0 * r29_CO * frac_COsdbss * S0 / FluxO
        g_CO_r30 = 2.0 * r30_CO * frac_COsdbss * S0 / FluxO
        g_CO_r31 = 2.0 * r31_CO * frac_Osdbss * S0 / FluxO
        g_CO_r32 = 2.0 * r32_CO * frac_COsdbss * S0 / FluxO
        g_CO_r34 = 2.0 * r34_CO * frac_Osdbss * S0 * frac_COfss / FluxO
        g_CO_r35 = 2.0 * r35_CO * frac_COsdbss * S0 * frac_Ofss / FluxO
        
        results_gammas_O = [g_O_r4, g_O_r6, g_O_r7, g_O_r8, g_O_r11, g_O_r12, \
                            g_O_r14, g_O_r15, g_O_r23, g_O_r24, g_O_r25, g_O_r27]
        
        results_names_O = ["g_O_r4", "g_O_r6", "g_O_r7", "g_O_r8", "g_O_r11", "g_O_r12", \
                            "g_O_r14", "g_O_r15","g_O_r23", "g_O_r24", "g_O_r25", "g_O_r27"] 
        
        results_gammas_CO = [g_CO_r6, g_CO_r7, g_CO_r8, g_CO_r10, g_CO_r11,  g_CO_r12,\
                            g_CO_r13, g_CO_r25, g_CO_r26, g_CO_r27, g_CO_r28, g_CO_r29, \
                            g_CO_r30, g_CO_r31, g_CO_r32, g_CO_r34, g_CO_r35]
        
        results_names_CO = ["g_CO_r6", "g_CO_r7", "g_CO_r8", "g_CO_r10", "g_CO_r11", \
                            "g_CO_r11", "g_CO_r12", "g_CO_r13", "g_CO_r25", "g_CO_r26", \
                            "g_CO_r27", "g_CO_r28", "g_CO_r29", "g_CO_r30", "g_CO_r31", \
                            "g_CO_r32", "g_CO_r34", "g_CO_r35"]
        
        results_gammas = results_gammas_O + results_gammas_CO
        results_names = results_names_O + results_names_CO
        
        # print(results_gammas)
        # print("gamma", sum(results_gammas))
        # print()
        
        return results_gammas, results_names
    
    
    
    def solve_system(self, exp_dict, energy_dict, solver="fixed_point", max_time=None):
        
        
        print(exp_dict['pressure'], exp_dict['current'], exp_dict['Tnw'], exp_dict['Tw'])
        
        rates_O_dict = self.rates_O_definition(exp_dict, energy_dict)
        rates_CO_dict = self.rates_CO_definition(exp_dict, energy_dict)
        rates_dict = (rates_O_dict, rates_CO_dict)
        
        if solver == "odeint":
            sol, success = self.solve_ode(rates_dict)
            steady_state_sol = tuple(sol[-1])
            
            if success == False:
                print("Failed to find the steady state solution")
                print("Exp dict: ", exp_dict)
            
        elif solver == "fixed_point":
            
            steady_state_sol, success  = self.solve_fixed_point(rates_dict, max_time=max_time)
            if success == False:
                print("Failed to find the steady state solution")
                print("Exp dict: ", exp_dict)
            
            steady_state_sol = tuple(steady_state_sol.x)
        else:
            raise ValueError("Invalid solver")
        
        result_gammas, result_names = self.compute_gammas(steady_state_sol, rates_dict)
        
        return steady_state_sol, result_gammas, result_names, success
