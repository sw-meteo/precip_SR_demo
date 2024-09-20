'''
downscale dataset to make LR version 
current x4: LR=0.25°, HR=0.0625°
brute-force, each window takes more than 1 hour
make sure scale-factor can be divided by the original shape
'''

import xarray as xr
import pandas as pd
import numpy as np
import os.path
import rootutils
from tqdm import tqdm


'''
basic info
'''
base = rootutils.find_root(search_from=__file__, indicator=".project-root")
fdir = os.path.join(base, "data/dataset")
cdir = os.path.join(base, "dependency/coords")
pname = "CLDAS"; var_name = "pre"
window = 5

coords = np.load(os.path.join(cdir, f"{pname}.coords.npz"))
n_lat = len(coords['lat']); n_lon = len(coords['lon'])
scale_factor = 4
n_lat_LR = n_lat // scale_factor; n_lon_LR = n_lon // scale_factor

time_range = pd.date_range(start=pd.to_datetime('2008-01-01'),
                          end=pd.to_datetime('2017-12-31'), 
                          freq='D')
time_range = time_range[:-window+1]  # locate start of each window
n_time_total = len(time_range)

pre_ifname = os.path.join(fdir, f"{pname}.{var_name}.{n_lat}x{n_lon}.{window}d.dat")
pre_ofname = os.path.join(fdir, f"{pname}.{var_name}.{n_lat_LR}x{n_lon_LR}.LR.{window}d.dat")


'''
initialize
'''
HR_dataset = np.memmap(pre_ifname, dtype=np.float32, mode='r+', 
              shape=(n_time_total, n_lat, n_lon))

_ = np.memmap(pre_ofname, dtype=np.float32, mode='w+',
                           shape=(n_time_total, n_lat_LR, n_lon_LR)); del _
arr_dataset_save_LR = np.zeros((n_time_total, n_lat_LR, n_lon_LR), dtype=np.float32)


'''
downscaling
'''
for i in tqdm(range(n_time_total)):
    for j in range(n_lat_LR):
        for k in range(n_lon_LR):
            arr_dataset_save_LR[i, j, k] = HR_dataset[i, 
                                                      j*scale_factor:(j+1)*scale_factor, 
                                                      k*scale_factor:(k+1)*scale_factor]\
                                                          .mean()


'''
save data
'''
save_arr_dataset_LR = np.memmap(pre_ofname, dtype=np.float32, mode='r+',
                            shape=(n_time_total, n_lat_LR, n_lon_LR))
save_arr_dataset_LR[:] = arr_dataset_save_LR[:]

del save_arr_dataset_LR
