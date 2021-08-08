#%% imports and definitions
import os
import pickle

import cv2
import dask.array as darr
import holoviews as hv
import xarray as xr
from dask.diagnostics import ProgressBar

from routine.alignment import apply_transform
from routine.preprocessing import denoise, rebase, remove_background
from routine.utilities import load_v4_folder, save_minian

pbar = ProgressBar(minimum=2)
pbar.register()
hv.notebook_extension("bokeh")

IN_DPATH = "./data/2color_pilot_tdTomato/m00/2021_08_05/16_22_03"
IN_TX = "./store/tx_tdTomato.pkl"
OUT_DSPATH = "~/var/miniscope_2s/tdTomato/minian_ds"
param_mc = {"alt_error": 5}
param_thres = {
    "maxValue": 1,
    "adaptiveMethod": cv2.ADAPTIVE_THRESH_MEAN_C,
    "thresholdType": cv2.THRESH_BINARY,
    "blockSize": 31,
    "C": -8,
}
OUT_DSPATH = os.path.normpath(os.path.expanduser(OUT_DSPATH))
os.makedirs(OUT_DSPATH, exist_ok=True)


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
va_top, va_side = load_v4_folder(IN_DPATH)
print("preprocess")
va_top = pre_process(va_top)
va_side = pre_process(va_side)
print("flip array")
va_side = xr.apply_ufunc(
    darr.flip,
    va_side,
    input_core_dims=[["frame", "height", "width"]],
    output_core_dims=[["frame", "height", "width"]],
    kwargs={"axis": 1},
    dask="allowed",
)
print("align array")
with open(IN_TX, "rb") as pklf:
    tx = pickle.load(pklf)
va_side = xr.apply_ufunc(
    apply_transform,
    va_side,
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    vectorize=True,
    kwargs={"tx": tx, "fill": 0},
    dask="parallelized",
    output_dtypes=float,
)
va_top = save_minian(va_top.rename("va_top"), OUT_DSPATH, overwrite=True)
va_side = save_minian(va_side.rename("va_side"), OUT_DSPATH, overwrite=True)

#%% plot pre-process result
opts_im = {"cmap": "viridis"}
hv.Image(va_top.isel(frame=0), ["width", "height"], label="top").opts(
    **opts_im
) + hv.Image(va_side.isel(frame=0), ["width", "height"], label="side").opts(**opts_im)
