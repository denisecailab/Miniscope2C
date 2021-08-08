#%% imports and definitions
import shutil

import cv2
import dask.array as darr
import holoviews as hv
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from routine.preprocessing import denoise, rebase, remove_background
from routine.utilities import load_v4_folder, save_minian

pbar = ProgressBar(minimum=2)
pbar.register()
hv.notebook_extension("bokeh")

DPATH = "./data/2color_pilot_tdTomato/bench/2021_06_28/14_56_08"
INTPATH = "./intermediate"
param_mc = {"alt_error": 1}
param_thres = {
    "maxValue": 1,
    "adaptiveMethod": cv2.ADAPTIVE_THRESH_MEAN_C,
    "thresholdType": cv2.THRESH_BINARY,
    "blockSize": 31,
    "C": -8,
}


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


#%% process data
va_top, va_side = load_v4_folder(DPATH)
shutil.rmtree(INTPATH, ignore_errors=True)
print("flipping array")
va_side = xr.apply_ufunc(
    darr.flip,
    va_side,
    input_core_dims=[["frame", "height", "width"]],
    output_core_dims=[["frame", "height", "width"]],
    kwargs={"axis": 1},
    dask="allowed",
)
print("pre-processing")
va_top = pre_process(va_top)
va_side = pre_process(va_side)
va_top = save_minian(va_top.rename("va_top"), INTPATH, overwrite=True)
va_side = save_minian(va_side.rename("va_side"), INTPATH, overwrite=True)

#%%plot pre-process result
opts_im = {"cmap": "viridis"}
hv.Image(va_top.isel(frame=0), ["width", "height"], label="top").opts(
    **opts_im
) + hv.Image(va_side.isel(frame=0), ["width", "height"], label="side").opts(**opts_im)
