#%% imports and definitions
import os
import pickle
import re

import dask.array as darr
import holoviews as hv
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

from routine.alignment import apply_transform, est_affine
from routine.preprocessing import denoise, rebase, remove_background
from routine.utilities import load_v4_folder, save_minian

pbar = ProgressBar(minimum=2)
pbar.register()
hv.notebook_extension("bokeh")

IN_DPATH = "data/2color_pilot_tdTomato/bench/2021_06_28/14_56_08"
IN_FM = slice(0, 15 * 30)
OUT_TX = "./store/tx_tdTomato.pkl"
OUT_FIG = "./output/tdTomato/alignment.html"


def pre_process(va: xr.DataArray) -> xr.DataArray:
    va = denoise(va, method="median", ksize=3)
    va = remove_background(va.astype(float), method="uniform", wnd=30)
    va = xr.apply_ufunc(
        rebase,
        va,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        dask="parallelized",
        vectorize=True,
        kwargs={"q": 0.8},
    )
    return va


#%% load data and preprocess
va_top, va_side = load_v4_folder(IN_DPATH)
va_side = xr.apply_ufunc(
    darr.flip,
    va_side,
    input_core_dims=[["frame", "height", "width"]],
    output_core_dims=[["frame", "height", "width"]],
    kwargs={"axis": 1},
    dask="allowed",
)
va_top = pre_process(va_top)
va_side = pre_process(va_side)

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
hv_align = (
    hv.Image(fm_top, ["width", "height"], label="top").opts(**im_opts)
    + hv.Image(fm_side, ["width", "height"], label="side").opts(**im_opts)
    + hv.Image(fm_side_reg, ["width", "height"], label="side_reg").opts(**im_opts)
    + hv.Image((fm_side_reg - fm_top), ["width", "height"], label="affine").opts(
        **im_opts
    )
).cols(2)
hv.save(hv_align, OUT_FIG)

#%% save result
os.makedirs(os.path.split(OUT_TX)[0], exist_ok=True)
with open(OUT_TX, "wb") as pklf:
    pickle.dump(tx, pklf)
