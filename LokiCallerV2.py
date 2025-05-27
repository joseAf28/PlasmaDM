from scipy.optimize import root_scalar
import concurrent.futures
from abc import ABC, abstractmethod

import yaml
from pathlib import Path
import numpy as np
import scipy as sp
import re
import logging
import os
import io

logging.basicConfig(level=logging.INFO)


class RightThreshold(Exception):
    """Raised when f(x) falls within the allowed limit."""
    pass


class PhysicalSimulator(ABC):
    
    def __init__(self, eng, base_dir, input_file, chem_file, print_flag=False):
        self.eng = eng
        
        self.base_dir = Path(base_dir)
        self.input_path = self.base_dir / 'Input' / input_file
        self.chem_path = self.base_dir / 'Input' / chem_file        
        
        with open(self.input_path, "r") as file:
            text = file.read()
        yaml_data = text.replace('%', '#')
        self.data = yaml.safe_load(yaml_data)
        
        self.chamber_radius = 1e-2
        self.electric_charge_const = 1.6021766208e-19
        
        self.functional_calls = 0
        self.print_flag = print_flag
        
        self.drift_name = 'Drift velocity'
        self.densities_file = 'chemFinalDensities.txt'
        self.features_file = 'swarmParameters.txt'
    
    
    @abstractmethod
    def modify_input_data(self, params):
        pass
    
    
    @abstractmethod
    def modify_params_chem(self, str_vec, content_vec, params_mod):
        pass
    
    
    def modify_param_objective(self, electron_den):
        self.data['workingConditions']['electronDensity'] = float(electron_den)
    
    
    def write_input_file(self):
        
        yaml_out = yaml.dump(self.data, sort_keys=False)
        in_text_out = yaml_out.replace('#', '%')
        with open(self.input_path, "w") as file:
            file.write(in_text_out)
    
    
    def _load_chem_file(self, file_path):
        
        str_vec = []
        content_vec = []
        with open(file_path) as f:
            
            for raw in f:
                line = raw.rstrip()
                # Skip section header
                m = re.match(r'%[- ]*([^-]+?) *[-]*$', line)
                if m:
                    continue
                # Skip blank or pure‐comment lines
                if not line or line.lstrip().startswith('%'):
                    continue
                # Split on the first 3 '|'s
                parts = [p.strip() for p in line.split('|', 3)]
                if len(parts) != 4:
                    continue

                reac_prod, rate_type, rate_expr, comment = parts
                rate_expr = rate_expr.split(',')
                if not '%' in comment:
                    comment = '%' + comment
                    
                str_vec.append(reac_prod)
                content_vec.append({
                    'rate_type':  rate_type,
                    'rate_expr':  rate_expr,
                    'reference':  comment
                })
                
        
        return str_vec, content_vec
    
    
    
    def write_chem_file(self, str_vec, content_vec):
        
        ###* write the output
        output_txt = []
        for i in range(len(content_vec)):
            content_vec[i]['rate_expr'] = ','.join(content_vec[i]['rate_expr'])
            line2 = '\t| '.join(list(content_vec[i].values()))
            line = str_vec[i] + "\t| " + line2 + "\n"
            output_txt.append(line)
        
        with open(self.chem_path, 'w') as fout:
            fout.writelines(output_txt)
    
    
    
    def _run_matlab(self):
        
        buf = io.StringIO()
        try:
            # if self.print_flag:
            # print("Running MATLAB ... ")
            self.eng.loki(self.input_path.name, nargout=0, stdout=buf, stderr=buf)
        except Exception as e:
            logging.error(f"MATLAB error: {e}")
            logging.debug(buf.getvalue())
            raise
    
    
    
    def _read_output_feat(self, output_feat_path):
        pattern = re.compile(r'^(.*?)=\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)')
        results = []
        names = []
        with open(output_feat_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = pattern.match(line)
                if m:
                    name, val_str = m.groups()
                    name = name.strip()
                    value = float(val_str)
                    names.append(name)
                    results.append(value)
                else:
                    print(f"Unparsed line: {line}")
                    pass
        return results, names
    
    
    
    def _read_output_densities(self, output_densities_path, skiprows=5):
        
        parts = []
        for line in output_densities_path.read_text().splitlines()[skiprows:]:
            toks = [tok for tok in line.split() if tok]
            if len(toks) < 2:
                continue
            elif toks[0] == '|':
                continue
            
            parts.append(toks)
        
        max_len = max(len(p) for p in parts)
        full_rows = [p for p in parts if len(p) == max_len]
        
        arr = np.array(full_rows, dtype=object)
        names = arr[:, 0].tolist()
        values = arr[:, 1].astype(float)
        
        return values, names
    
    
    
    def objective_current(self, electron_density, current_exp, output_feat_file, print_flag=True, epsilon=0.0025):
        
        self.functional_calls += 1
        
        # print("electron_density:", electron_density*1e-15)
        
        self.modify_param_objective(electron_density)
        self.write_input_file()
        
        self._run_matlab()
        
        results, names = self._read_output_feat(output_feat_file)
        idx_drift = names.index(self.drift_name)
        drift_velocity = results[idx_drift]
        
        current = self.electric_charge_const * electron_density * drift_velocity * np.pi * self.chamber_radius**2
        
        print(f"Call: {self.functional_calls} I: {current} I_exp: {current_exp}")
        
        if np.abs(current-current_exp) < epsilon:
            raise RightThreshold()
        else:    
            return current
    
    
    def solver_one_point(self, params_tuple, params_chem, current_exp, ne_min, ne_max):
        
        folder_name = params_tuple[-1]
        
        folder_path = self.base_dir / 'Output' / folder_name
        output_densities_file = folder_path / self.densities_file
        output_feat_file = folder_path / self.features_file        
        
        self.modify_input_data(params_tuple)
        self.write_input_file()
        
        chem_str_vec, chem_content_vec = self._load_chem_file(self.chem_path)
        chem_content_vec = self.modify_params_chem(chem_str_vec, chem_content_vec, params_chem)
        self.write_chem_file(chem_str_vec, chem_content_vec)
        
        
        def f(x):
            return self.objective_current(x, current_exp, output_feat_file) - current_exp
        
        a = ne_min
        b = ne_max
        try:
            sol = root_scalar(
                f,
                method='ridder',
                bracket=[a, b],
                xtol=1e-3,
                maxiter=25
            )
        except RightThreshold as e:
            return self._read_output_densities(output_densities_file)
        
        return self._read_output_densities(output_densities_file)



    def solver_one_pointV2(self, params_tuple, params_chem, current_exp, ne_min, ne_max, 
                        expand_factor=2.0, max_expansions=5):
        
        
        
        folder_name = params_tuple[-1]
        
        folder_path = self.base_dir / 'Output' / folder_name
        output_densities_file = folder_path / self.densities_file
        output_feat_file = folder_path / self.features_file        
        
        self.modify_input_data(params_tuple)
        self.write_input_file()
        
        chem_str_vec, chem_content_vec = self._load_chem_file(self.chem_path)
        chem_content_vec = self.modify_params_chem(chem_str_vec, chem_content_vec, params_chem)
        self.write_chem_file(chem_str_vec, chem_content_vec)
        
        def f(x):
            return self.objective_current(x, current_exp, output_feat_file) - current_exp
        
        a = ne_min
        b = ne_max
        
        for attempt in range(max_expansions):
            try:
                sol = root_scalar(
                    f,
                    method='ridder',
                    bracket=[a, b],
                    xtol=1e-3,
                    maxiter=25
                )
            except RightThreshold as e:
                return self._read_output_densities(output_densities_file)
            
            except ValueError as e:
                if "f(a) and f(b)" in str(e):
                    a /= expand_factor
                    b *= expand_factor
                    continue
                else:
                    pass
            
            if sol.converged:
                ne_star = sol.root
                return self._read_output_densities(output_densities_file)
            
        raise RuntimeError(
            f"Failed to bracket & converge after {max_expansions} expansions; "
            f"last bracket was [{a:.3e}, {b:.3e}]"
        )