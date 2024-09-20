'''
CLDAS data is saved in daily files
1. merge all years but save every DOY separately
    because single data is too large
    e.g. 0101 = [ 20080101, 20090101, ..., 20170101 ]
    no extra fix for leap day because annual cycle will be smoothed later
2. extract PRCP_DAY_SUM variable
3. BTW, save coords
'''

import pandas as pd
import xarray as xr
import numpy as np
import os.path
import rootutils


'''
basic info
'''
base = rootutils.find_root(search_from=__file__, indicator=".project-root")
idir = os.path.join(base, "data/raw")
odir = os.path.join(base, "data/interim/DOY")
cdir = os.path.join(base, "dependency/coords")  # save coords BTW
os.makedirs(odir, exist_ok=True)
os.makedirs(cdir, exist_ok=True)

pname = "CLDAS"
encoding = {'lat': {'_FillValue': False},
            'lon': {'_FillValue': False},
            'pre': {'_FillValue': -999, 
                    'dtype': 'float32'}}

# iter over all DOY
DOY_range = pd.date_range(start=pd.to_datetime('2008-01-01'),
                          end=pd.to_datetime('2008-12-31'), 
                          freq='D')


'''
process
'''
for dt_DOY in DOY_range:
    mth = dt_DOY.strftime('%m'); day = dt_DOY.strftime('%d')
    date_range = [f"{yr}-{mth}-{day}" for yr in range(2008, 2018)]
    if dt_DOY.strftime('%m%d') == "0229":
        date_range = date_range[::4]
    len_time = len(date_range)
    arr_save = np.zeros((len_time, 1040, 1600))
    
    for idate, date in enumerate(date_range):
        date = pd.to_datetime(date)
        yr = date.strftime('%Y'); mth = date.strftime('%m')
        fname = os.path.join(idir, yr, mth, "CLDAS_NRT_ASI_0P0625_DAY-PRE-" + date.strftime('%Y%m%d') + "00.nc")
        ds = xr.open_dataset(fname).transpose('LAT', 'LON')
        if 'lat' not in dir():  # save only for the first time
            lat = ds.LAT.values; lon = ds.LON.values
            np.savez(os.path.join(cdir, f"{pname}.coords.npz"), lat=lat, lon=lon)
        ds.close()
        arr_save[idate, :, :] = ds.PRCP_DAY_SUM.values
        del ds
    
    ds_save = xr.Dataset(
        {'pre': 
            (['time', 'lat', 'lon'], arr_save)},
        coords={
            'time': list(pd.to_datetime(date_range)),
            'lat': lat,
            'lon': lon,
        },
    )
    ds_save.astype('float').to_netcdf(os.path.join(odir, f"CLDAS_pre_{dt_DOY.strftime('%m%d')}_2008-2017.nc"), encoding=encoding)
