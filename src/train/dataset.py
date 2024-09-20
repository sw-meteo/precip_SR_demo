import os.path
import numpy as np
import pandas as pd
import random
import rootutils
from typing import List
import torch
from torch.utils.data import Dataset
base = rootutils.find_root(search_from=__file__, indicator=".project-root")


class SubImageDataset(Dataset):
    '''
    read memmap files and filter with conditions
    every iter locate one certain timestep 
    then randomly select a patch from this field
    '''
    def __init__(self, 
                 fdir: str = "data/dataset",  # relative path
                 sdir: str = "dependency/stats",  # relative path
                 HR_shape: List[int] = [1040, 1600],  # whole field shape
                 LR_shape: List[int] = [260, 400],
                 window: int = 5,  # values are pentad statistics
                 chunk_width: int = 64,  # width of LR patch
                 upscaling_factor: int = 4,
                 mth_sel: List[int] = None, # e.g. [1, 2, 3]
                 year_lim: List[int] = None, # e.g. [2008, 2016]
                 year_sel: List[int] = None, # e.g. [2010]
                ):

        self.chunk_width = chunk_width
        self.upscaling_factor = upscaling_factor
        
        meta_fname = os.path.join(base, fdir, f"CLDAS.meta.5d.csv")
        HR_fname = os.path.join(base, fdir, f"CLDAS.pre.{'x'.join([str(i) for i in HR_shape])}.{window}d.dat")
        LR_fname = os.path.join(base, fdir, f"CLDAS.pre.{'x'.join([str(i) for i in LR_shape])}.LR.{window}d.dat")
        stats_fname = os.path.join(base, sdir, f"CONST.CLDAS.pre.{window}d.txt")

        self.meta = pd.read_csv(meta_fname)
        self.arr_HR = np.memmap(HR_fname, dtype='float32', mode='r+', 
                                 shape=(len(self.meta), *HR_shape))
        self.arr_LR = np.memmap(LR_fname, dtype='float32', mode='r+',
                                 shape=(len(self.meta), *LR_shape))
        if window in [5, 10]:
            self.hard_bnd = [-1000, 1000] # drop very unreasonable values
            with open(stats_fname, "r") as f:
                MIN, LOGMIN, LOGMAX = map(float, f.read().strip().split())
            transform1 = lambda x: torch.log(x - MIN + 1e-6)
            transform2 = lambda x: (x - LOGMIN) / (LOGMAX - LOGMIN)
            self.transform = lambda x: transform2(transform1(x))
        else:
            raise NotImplementedError
        self.bnd = [0, 1] # for final clipping

        conditions = []
        if year_lim is not None:
            assert len(year_lim) == 2
            assert year_lim[0] <= year_lim[1]
            conditions.append(self.meta['year'] >= year_lim[0])
            conditions.append(self.meta['year'] <= year_lim[1])
        if year_sel is not None:
            conditions.append(self.meta['year'].isin(year_sel))
        if mth_sel is not None:
            conditions.append(self.meta['month'].isin(mth_sel))
        if len(conditions) > 0:
            self.used_indices = self.meta[np.all(conditions, axis=0)].index.to_list()
        else:
            self.used_indices = self.meta.index.to_list()
        
        self.meta = self.meta.iloc[self.used_indices]
    
    
    def __len__(self):
        return len(self.used_indices) * 10
    
    
    def drop_nan(self, arr, bnd=[-10, 10]):
        arr = torch.nan_to_num(arr, nan=0.0, posinf=bnd[1], neginf=bnd[0])
        arr = torch.clamp(arr, bnd[0], bnd[1])
        return arr

    
    def __getitem__(self, idx):
        '''
        load data and select patch
            ensure no nan values in the patch
            ensure the values in the patch are within the hard bounds (resonable)
        then apply transform 
        finally clip the final values to [0, 1]
        '''
        protect_pos = 0
        while True:
            assert protect_pos < 100, "No valid sample found!"
            idx = idx % len(self.used_indices)
            idx_of_all = self.used_indices[idx]
            X = torch.from_numpy(self.arr_LR[idx_of_all, :, :])
            Y = torch.from_numpy(self.arr_HR[idx_of_all, :, :])
            X_subimage, Y_subimage = self.get_subimage_sample(X, Y, self.chunk_width, self.upscaling_factor)
            if torch.sum(torch.isnan(X_subimage)) < 1e-5\
                and torch.sum(torch.isnan(Y_subimage)) < 1e-5\
                and torch.min(X_subimage) > self.hard_bnd[0]\
                and torch.max(X_subimage) < self.hard_bnd[1]\
                and torch.min(Y_subimage) > self.hard_bnd[0]\
                and torch.max(Y_subimage) < self.hard_bnd[1]:
                break
            idx = random.randint(0, len(self.used_indices)-1)
            protect_pos += 1
            
        X_subimage = self.transform(X_subimage)
        Y_subimage = self.transform(Y_subimage)
        X_subimage = self.drop_nan(X_subimage, self.bnd)
        Y_subimage = self.drop_nan(Y_subimage, self.bnd)

        return X_subimage.unsqueeze(0), Y_subimage.unsqueeze(0)

    def get_subimage_sample(self, X, Y, chunk_width, upscaling_factor):
        chunk_width = min(chunk_width, X.shape[0], X.shape[1])
        idx_x = np.random.randint(0, X.shape[0] - chunk_width)
        idx_y = np.random.randint(0, X.shape[1] - chunk_width)
        X_subimage = X[idx_x:idx_x + chunk_width, idx_y:idx_y + chunk_width]
        Y_subimage = Y[idx_x * upscaling_factor:(idx_x + chunk_width) * upscaling_factor,
                        idx_y * upscaling_factor:(idx_y + chunk_width) * upscaling_factor]
        return X_subimage, Y_subimage


if __name__ == '__main__':
    dataset = SubImageDataset()
    print(len(dataset))
    print(dataset[0])
    for item in dataset[0]:
        print(item.shape)
