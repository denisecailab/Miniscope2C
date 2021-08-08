#%% imports and definitions
import os
import pickle
import re

import cv2
import holoviews as hv
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

from routine.alignment import apply_transform, est_affine
from routine.utilities import open_minian, save_minian

pbar = ProgressBar(minimum=2)
pbar.register()
hv.notebook_extension("bokeh")

INTPATH = "./intermediate"
IN_FM = slice(0, 300)
OUT_TX = "./store/affine_tx.pkl"

#%% load data
minian_ds = open_minian(INTPATH)
va_top = minian_ds["va_top"]
va_side = minian_ds["va_side"]
#%% compute alignment
fm_top = va_top.isel(frame=IN_FM).compute().median("frame")
fm_side = va_side.isel(frame=IN_FM).compute().median("frame")
tx, param_dict = est_affine(fm_side.values, fm_top.values, lr=1)
param_df = (
    pd.Series(param_dict)
    .reset_index(name="metric")
    .rename(lambda c: re.sub("level_", "param_", c), axis="columns")
)
fm_side_reg = xr.apply_ufunc(
    apply_transform,
    fm_side,
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    kwargs={"tx": tx},
)
#%% plot result
im_opts = {"cmap": "viridis"}
(
    hv.Image(fm_top, ["width", "height"], label="top").opts(**im_opts)
    + hv.Image(fm_side, ["width", "height"], label="side").opts(**im_opts)
    + hv.Image(fm_side_reg, ["width", "height"], label="side_reg").opts(**im_opts)
    + hv.Image((fm_side_reg - fm_top), ["width", "height"], label="affine").opts(
        **im_opts
    )
).cols(2)

#%% save result
os.makedirs(os.path.split(OUT_TX)[0], exist_ok=True)
with open(OUT_TX, "wb") as pklf:
    pickle.dump(tx, pklf)

#%% align videos
with open(OUT_TX, "rb") as pklf:
    tx = pickle.load(pklf)
va_side_reg = xr.apply_ufunc(
    apply_transform,
    va_side,
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    vectorize=True,
    kwargs={"tx": tx, "fill": 0},
    dask="parallelized",
    output_dtypes=float,
)
va_side_reg = save_minian(va_side_reg.rename("va_side_reg"), INTPATH, overwrite=True)
