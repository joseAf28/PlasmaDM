import numpy as np
import scipy as sp
import pandas as pd
import os 
import h5py


class DataLoader():
    
    def __init__(self, schema, folder_path=None, files_path=None, output_file="Experimental_data.hdf5" ,output_folder="Buffer_Data", extension=".xlsx"):
        
        self.schema = schema
        if folder_path is not None:
            file_name_list = os.listdir(folder_path)
            self.data_files_path = [os.path.join(folder_path, ele) for ele in file_name_list if ele.endswith(extension)]
            
        elif files_path is not None:
            self.data_files_path = files_path
        else:
            raise ValueError("Please prove the path of the data files or folder")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        self.output_file = os.path.join(output_folder, output_file)


    def create_hdf5_buffer(self):
        
        data_tensor = {name: [] for name in self.schema.values()}
        for file_name in self.data_files_path:
            df = pd.read_excel(file_name)
            
            for name_exp, name_sim in self.schema.items():
                if name_exp in df.columns:
                    vec_aux = df[name_exp].to_numpy()
                    data_tensor[name_sim].append(vec_aux[~np.isnan(vec_aux)])
        
        n_size = None
        for name in self.schema.values():
            if len(data_tensor[name]) > 0:
                data_tensor[name] = np.concatenate(data_tensor[name])
                
                if n_size is None:
                    n_size = len(data_tensor[name])
                
        for name in self.schema.values():
            if len(data_tensor[name]) < n_size:
                data_tensor[name] = np.zeros(n_size)
                
        
        with h5py.File(self.output_file, "w") as file:
            file.create_dataset("File_Data", data=self.data_files_path)
            for name, data in data_tensor.items():
                file.create_dataset(name, data=data)
            file.close()
    
    
    def get_summary(self):
        
        print("Summary Buffer:")
        with h5py.File(self.output_file, "r") as file:
            print(f"File Data : {file['File_Data'][:]}")
            keys = list(file.keys())
            for key in keys:
                print(f"col {key}: shape{file[key][:].shape}")
            
            file.close()
    
    
    def load_data(self, force_update=False):
        
        if not os.path.exists(self.output_file) or force_update:
            self.create_hdf5_buffer()
        
        self.get_summary()
    