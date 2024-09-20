'''
collect mean of all DOY, concat to a single file
then smooth the annual cycle

takes 1.5 min
'''

import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm
import os.path
import rootutils
from utils.nc_handler import smooth_annual_cycle


'''
basic info
'''
base = rootutils.find_root(search_from=__file__, indicator=".project-root")
idir = os.path.join(base, "data/interim/DOY")
sdir = os.path.join(base, "dependency/stats")
os.makedirs(sdir, exist_ok=True)

pname = "CLDAS"; var_name = "pre"
smooth_window = 31  # smooth annual cycle

# iter over all DOY
DOY_range = pd.date_range(start=pd.to_datetime('2008-01-01'),
                          end=pd.to_datetime('2008-12-31'), 
                          freq='D')


'''
process
'''
dss_clim = []
for dt_DOY in tqdm(DOY_range):
    fname_DOY = os.path.join(idir, f"CLDAS_pre_{dt_DOY.strftime('%m%d')}_2008-2017.nc")
    ds_DOY = xr.open_dataset(fname_DOY)
    dss_clim.append(ds_DOY.mean('time'))
    ds_DOY.close(); del ds_DOY
ds_clim = xr.concat(dss_clim, dim='dayofyear')
ds_clim.to_netcdf(os.path.join(sdir, f"clim.{pname}.{var_name}.nc"))
ds_clim_smooth = smooth_annual_cycle(ds_clim, smooth_window)
ds_clim_smooth.to_netcdf(os.path.join(sdir, f"clim.{pname}.{var_name}.smooth.nc"))
