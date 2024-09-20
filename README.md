# precip_SR_demo

simple precipitation super-resolution demo in Pytorch-Lightning

### Usage

#### set up env

**critical packages: **`rootutils`, `wandb`, `omegaconf`, `torch`, `pytorch_lightning`

#### set up data

**raw & interim data is NOT provided dure to our license, please download at **[国家气象科学数据中心](https://data.cma.cn/data/detail/dataCode/NAFP_CLDAS2.0_NRT.html).

**after obtaining the raw data, execute files in **`src/raw2dataset` in numerical order, this will calculate pentad mean anomalies (i.e., deviation from grid annual cycle).

**if you are looking for daily values rather than x-day-mean anomalies, modify line 79 in step3.**

#### set up model

**only simple UNet is implemented in this public version, and this will definitely lead to undesirable results.**

**but you can easily try other available SR models e.g. SRGAN, **[RCAN](https://github.com/yjn870/RCAN-pytorch), etc.

**just add some model to **`src/models.py`

#### start training

**modify config in **`src/config.yaml`, or in cmd line, e.g., `python train.py plmodule_config.model_name='RCAN'`

### project file tree

```
 .
 ├── data
 │   ├── dataset
 │   │   ├── CLDAS.meta.5d.csv
 │   │   ├── CLDAS.pre.1040x1600.5d.dat
 │   │   └── CLDAS.pre.260x400.LR.5d.dat
 │   ├── interim
 │   │   └── DOY
 │   │       ├── CLDAS_pre_0101_2008-2017.nc
 ...
 │   └── raw
 │       ├── 2008
 │       │   ├── 01
 ...
 ├── dependency
 │   ├── coords
 │   │   └── CLDAS.coords.npz
 │   └── stats
 │       ├── clim.CLDAS.pre.nc
 │       ├── clim.CLDAS.pre.smooth.nc
 │       └── CONST.CLDAS.pre.5d.txt
 ├── notebooks
 │   ├── check_whole_field.ipynb
 │   ├── read_patched_dataset.ipynb
 │   └── smooth_annual_cycle.ipynb
 └── src
     ├── raw2dataset
     │   ├── step1.concat_data.py
     │   ├── step2.save_clim.py
     │   ├── step3.to_dataset.py
     │   ├── step4.make_LR.py
     │   ├── step5.get_transform_stats.py
     │   └── utils
     │       └── nc_handler.py
     └── train
         ├── callbacks.py
         ├── config.yaml
         ├── dataset.py
         ├── models.py
         ├── plmodules.py
         └── train.py
```
