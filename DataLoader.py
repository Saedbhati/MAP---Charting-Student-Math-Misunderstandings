import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class QAEDataset(Dataset):
    def __init__(self,x,y):
        super().__init__()

        self.x= x

        self.y = y
    def __len__(self):
        return len(self.x['input_ids'])
    
    def __getitem__(self, index):
        input_ids = torch.tensor(self.x['input_ids'][index], dtype=torch.long)
        attention_mask = torch.tensor(self.x['attention_mask'][index], dtype=torch.long)
        label = torch.tensor(self.y[index], dtype=torch.long)
        
        return {
            'input': input_ids,     # shape: [seq_len]
            'attn': attention_mask, # shape: [seq_len]
            'label': label          # shape: []
        }