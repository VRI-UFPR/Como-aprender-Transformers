# Codigo baseado no codigo 
# https://github.com/KristinaRay/Deep-Learning-School-part-2/blob/main/modules.py
#
# Data: 09/05/2024
#
# Modificado por: 
# - Luan Matheus Trindade Dalmazo 


# =============================================================================
#  Header
# =============================================================================

from torch.utils.data import Dataset
import pandas as pd 
import numpy as np
import torch 

# =============================================================================
#  Dataset Manager
# =============================================================================

class SpeechDataset(Dataset):

    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path).to_numpy()

    def __getitem__(self, index):
        sample = self.data[index][0]
        label = self.data[index][-1:]

    
        return sample, label

    def __len__(self):
        return len(self.data)