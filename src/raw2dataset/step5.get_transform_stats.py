'''
get critical stats (min, max) for later transform
each window takes 40G RAM & 2.5 min
'''

import pandas as pd
import numpy as np
import os.path
import rootutils
from decimal import Decimal


'''
basic info
'''
base = rootutils.find_root(search_from=__file__, indicator=".project-root")
idir = os.path.join(base, "data/interim/DOY")
odir = os.path.join(base, "data/dataset")
cdir = os.path.join(base, "dependency/coords")
sdir = os.path.join(base, "dependency/stats")

pname = "CLDAS"; var_name = "pre"
window = 5

coords = np.load(os.path.join(cdir, f"{pname}.coords.npz"))
n_lat = len(coords['lat']); n_lon = len(coords['lon'])

time_range = pd.date_range(start=pd.to_datetime('2008-01-01'),
                          end=pd.to_datetime('2017-12-31'), 
                          freq='D')
time_range = time_range[:-window+1]  # locate start of each window
n_time_total = len(time_range)

pre_fname = os.path.join(odir, f"CLDAS.pre.{n_lat}x{n_lon}.{window}d.dat")
stats_fname = f"CONST.{pname}.{var_name}.{window}d.txt"

arr_dataset = np.memmap(pre_fname, dtype=np.float32, mode='r+', 
              shape=(n_time_total, n_lat, n_lon))


'''
get transform stats
'''
if window in [5, 10]:
    MIN = np.nanmin(arr_dataset)
    log_samples = np.log(arr_dataset - MIN + 1e-6)
    del arr_dataset
    LOGMIN = np.nanpercentile(log_samples, 1)
    LOGMAX = np.nanpercentile(log_samples, 99)
    with open(os.path.join(sdir, stats_fname), "w") as f:
        f.write(f"{Decimal(MIN.item())} {Decimal(LOGMIN.item())} {Decimal(LOGMAX.item())}")
