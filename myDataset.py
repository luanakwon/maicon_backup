import os
from PIL import Image
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self,
                df, # dataframe containing frame(img)path, label(real/fake)
                transformer,
                shuffle = True,
                seed:int = None):
        self.df = df
        if shuffle:
            self.df = df.sample(frac=1,random_state=seed)
        self.transformer = transformer
        self.seed = seed
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        record = self.df.iloc[index]
        img_path = record['face_path']
        img = Image.open(img_path)
        img = np.array(img)
        img = self.transformer(image=img)['image']
        
        label = 0.0 if record['label'] == 'fake' else 1.0
        
        return img, label