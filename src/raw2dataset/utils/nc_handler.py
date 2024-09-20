import xarray as xr

def smooth_annual_cycle(ds_DOY, window):
    ds_DOY_before = ds_DOY.isel(dayofyear=slice(-window//2+1, None))
    ds_DOY_before = ds_DOY_before.assign_coords(dayofyear=(ds_DOY.dayofyear.values[-window//2+1:] - 366))
    ds_DOY_after = ds_DOY.isel(dayofyear=slice(window//2))
    ds_DOY_after = ds_DOY_after.assign_coords(dayofyear=(ds_DOY.dayofyear.values[:window//2] + 366))
    ds_DOY_circular = xr.concat([ds_DOY_before, ds_DOY, ds_DOY_after], dim='dayofyear').sortby('dayofyear')
    
    ds_DOY_smooth = ds_DOY_circular.rolling(dayofyear=window).mean().isel(dayofyear=slice(window-1, None))
    ds_DOY_smooth = ds_DOY_smooth.assign_coords(dayofyear=ds_DOY_circular.dayofyear.values[window//2:-window//2+1])
    return ds_DOY_smooth
