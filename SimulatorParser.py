import json
import re
import numpy as np
import sympy as sy
import scipy as sp


class SimulatorParser():
    
    def __init__(self, reactions_file, const_dict):
        self.reactions_filename = reactions_file
        self.const_dict = const_dict
    
    
    @staticmethod
    def parse_side(side_str):
        species = {}
        gas = ''
        for term in side_str.split('+'):
            term = term.strip()
            
            m = re.match(r'(\d*)\s*([A-Za-z0-9_]+)', term)
            if not m:
                raise ValueError(f"Cannot Parse '{side_str}'")
            
            name = m.group(2)
            name_split = name.split("_")
            if len(name_split) == 1:
                gas = name
            
            elif len(name_split) == 2:
                coeff = int(m.group(1)) if m.group(1) else 1
                species[name] = coeff
                
            else:
                raise ValueError(f"Cannot Parse '{side_str}'. It has more than one '_' ")
        
        return species, gas
    
    
    @staticmethod
    def load_reactions(reactions_filename):
        with open(reactions_filename) as f:
            raw = json.load(f)
        
        reactions_list = []
        all_species = set()
        for r in raw:
            left, right = r['equation'].split('<->') if '<->' in r['equation'] else r['equation'].split('->')
            rev = '<->' in r['equation']
            left, gas = SimulatorParser.parse_side(left)
            right, _ = SimulatorParser.parse_side(right)
            
            all_species |= left.keys() | right.keys()
            reactions_list.append({
                'id':           r['id'],
                'left':         left,
                'right':        right,
                'rate':         r['rate'],
                'gas_specie':   gas,
                'model_dict':   r['model_dict'],
                'gamma':        r['gamma']
            })
        
        species_list = sorted(all_species)
        rates_list = ["r_"+str(r["id"]) for r in reactions_list]
        
        groups_species_split = [ele.split('_') for ele in species_list]
        group_species = list(set([ele[1] for ele in groups_species_split]))
        
        return reactions_list, species_list, rates_list, group_species, groups_species_split
    
    
    @staticmethod
    def make_symbols(name_list):
        syms = sy.symbols(' '.join(name_list))
        return list(syms), dict(zip(name_list, syms))
    
    
    def create_physical_model(self, flag_debugging = True):
    
        ####* load reactions from the file
        reactions_list, species_list, rates_list, \
        group_species, groups_species_split = SimulatorParser.load_reactions(self.reactions_filename)
        
        ####* create symbolic variables for the state variabled and rates
        species_vars, species_map = SimulatorParser.make_symbols(species_list)
        rates_vars, rates_map = SimulatorParser.make_symbols(rates_list)
        
        ####* conservations laws for the V_* species (with * = {F, V, ...})
        old_expressions = []
        new_expressions = []
        conservation_laws_dict = {}
        
        for group in group_species:
            vec_species = []
            for specie in groups_species_split:
                
                if specie[1] == group and specie[0] != "V":
                    vec_species.append("_".join(specie))
            if len(vec_species) == 0:
                raise ValueError(f"The species _{group} don't have the vacancy state V_{group}. Please modify the reactions list")
            
            conservation_laws_dict["V_" + group] = vec_species
        
        for old, new_vec in conservation_laws_dict.items():
            
            old_expressions.append(species_map[old])
            new_term  = 0.0
            for ele in new_vec:
                new_term = new_term + species_map[ele]
                
            new_expressions.append(1.0 - new_term)
            
        pairs_laws = list(zip(old_expressions, new_expressions))
        
        
        ####* write the symbolic expressions
        laws_values_list = list(conservation_laws_dict.values())
        eff_species_list = [item for sublist in laws_values_list for item in sublist]
        
        ratio_S_F = self.const_dict["S0"] / self.const_dict["F0"]
        
        equations_sym = []
        for specie in eff_species_list:
            
            F_flag = specie.split('_')[1] == "F"
            
            line = []
            for idx, eq in enumerate(reactions_list):
                
                S_flag = False
                double_flag = False
                
                if specie in eq["left"].keys():
                    
                    rate = "r_" + str(eq["id"])
                    term = -1.0
                    for ele, nu in eq["left"].items():
                        
                        term = term * species_map[ele] ** nu
                        if nu == 2:
                            double_flag = True
                        if F_flag and not S_flag:
                            S_flag = ele.split('_')[1] == "S"
                            
                    term = term * rates_map[rate]
                    if F_flag and S_flag:
                        term = term * ratio_S_F
                    if double_flag:
                        term = term * 2.0
                        
                    line.append(term)
                elif specie in eq["right"].keys():
                    
                    rate = "r_" + str(eq["id"])
                    term = 1.0
                    for ele, nu in eq["left"].items():
                        
                        term = term * species_map[ele] ** nu
                        if nu == 2:
                            double_flag = True
                        if F_flag and not S_flag:
                            S_flag = ele.split('_')[1] == "S"
                    
                    term = term * rates_map[rate]
                    if F_flag and S_flag:
                        term = term * ratio_S_F
                    if double_flag:
                        term = term * 2.0
                    
                    line.append(term)
                else:
                    pass 
                
            equations_sym.append(sum(line))
        
        equations_sym_new = [equation.subs(pairs_laws) for equation in equations_sym]
        
        if flag_debugging:
            for i in range(len(eff_species_list)):
                print(f"d{eff_species_list[i]}/dt = {equations_sym_new[i]}")
        
        eff_species_vars = [species_map[ele] for ele in eff_species_list]
        
        model_fun = sy.lambdify((eff_species_vars, rates_vars), equations_sym_new, modules='numpy')
        
        output_dict = dict()
        output_dict['model'] = model_fun
        output_dict['species_model'] = eff_species_list
        output_dict['rates_model'] = rates_list
        output_dict['reactions_list'] = reactions_list
        
        return output_dict