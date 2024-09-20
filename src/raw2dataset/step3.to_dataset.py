'''
extra samples from DOY, calculate anom value and save to memmap
NOT do extra transform here (e.g. log, standardize)

single window takes ~30G RAM and 10 min
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
idir = os.path.join(base, "data/interim/DOY")
odir = os.path.join(base, "data/dataset")
cdir = os.path.join(base, "dependency/coords")
sdir = os.path.join(base, "dependency/stats")
os.makedirs(odir, exist_ok=True)

pname = "CLDAS"; var_name = "pre"
window = 5

coords = np.load(os.path.join(cdir, f"{pname}.coords.npz"))
n_lat = len(coords['lat']); n_lon = len(coords['lon'])

time_range = pd.date_range(start=pd.to_datetime('2008-01-01'),
                          end=pd.to_datetime('2017-12-31'), 
                          freq='D')
time_range = time_range[:-window+1]  # locate start of each window
n_time_total = len(time_range)

meta_header = ['year', 'month', 'day', 'DOY']
pre_fname = os.path.join(odir, f"CLDAS.pre.{n_lat}x{n_lon}.{window}d.dat")
meta_fname = os.path.join(odir, f"CLDAS.meta.{window}d.csv")


'''
initialize
'''
_ = np.memmap(pre_fname, dtype=np.float32, mode='w+', 
              shape=(n_time_total, n_lat, n_lon)); del _
_ = pd.DataFrame(columns=meta_header).to_csv(meta_fname, index=False); del _

arr_value_save = np.zeros((n_time_total, n_lat, n_lon), dtype=np.float32)
arr_std_save = np.zeros((n_time_total, n_lat, n_lon), dtype=np.float32)

i_time = 0


'''
save data
'''
fname_clim = os.path.join(sdir, f"clim.{pname}.{var_name}.smooth.nc")
ds_clim = xr.open_dataset(fname_clim)

pos = 0
for sdt in tqdm(time_range):
    DOY = sdt.dayofyear
    dss_window = []
    for iw in range(window):
        dt = sdt + pd.DateOffset(days=iw)
        fname = os.path.join(idir, f"CLDAS_pre_{dt.strftime('%m%d')}_2008-2017.nc")
        ds = xr.open_dataset(fname).sel(time=dt)\
            .transpose('lat', 'lon')
        dss_window.append(ds)
        ds.close(); del ds
    ds_window = xr.concat(dss_window, dim='time')
    ds_sample = ds_window.mean('time') 
    
    ds_window_clim = ds_clim.sel(dayofyear=slice(DOY, DOY+window-1)).mean('dayofyear')  
    
    if window in [5, 10]:
        ds_save = ds_sample - ds_window_clim  # anomaly values
    elif window == 30:
        ds_save = (ds_sample - ds_window_clim) / ds_window_clim
    
    arr_value_save[pos] = ds_save.pre.values
    pos += 1

    df_samples = pd.DataFrame(columns=meta_header)
    df_samples['year'] = [sdt.year]
    df_samples['month'] = [sdt.month]
    df_samples['day'] = [sdt.day]
    df_samples['DOY'] = [DOY]
    assert len(df_samples) == 1

    df_samples.to_csv(meta_fname, mode='a', header=False, index=False)

    del ds_window, ds_sample, ds_window_clim, ds_save, df_samples
    
save_arr_value_all = np.memmap(pre_fname, dtype=np.float32, mode='r+',
                            shape=(n_time_total, n_lat, n_lon))
save_arr_value_all[:] = arr_value_save[:]
del save_arr_value_all
