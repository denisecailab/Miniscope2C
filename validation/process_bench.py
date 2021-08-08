#%% imports and definitions
import os
import pickle
import shutil

import cv2
import dask.array as darr
import holoviews as hv
import numpy as np
import pandas as pd
import SimpleITK as sitk
import xarray as xr
from dask.diagnostics import ProgressBar
from skimage.registration import phase_cross_correlation

from .routine.motion_correction import apply_transform, estimate_motion
from .routine.preprocessing import remove_background
from .routine.utilities import (
    apply_transform,
    load_videos,
    map_ts,
    norm,
    open_minian,
    save_minian,
)

pbar = ProgressBar(minimum=2)
pbar.register()
hv.notebook_extension("bokeh")

DPATH = "./data/2color_pilot_tdTomato/bench/2021_06_08/12_08_29"
IN_TX = "./store/affine_tx.pkl"
INTPATH = "./intermediate_bench"
# SUBSET_MC = {"height": slice(50, 550), "width": slice(50, 550)}
SUBSET_MC = dict()
param_mc = {"alt_error": 1}
param_thres = {
    "maxValue": 1,
    "adaptiveMethod": cv2.ADAPTIVE_THRESH_MEAN_C,
    "thresholdType": cv2.THRESH_BINARY,
    "blockSize": 31,
    "C": -8,
}


def rebase(x: np.ndarray, q: float = 0.1) -> np.ndarray:
    return np.clip(x, max(np.quantile(x, q), 0), None)


def pre_process(va: xr.DataArray) -> xr.DataArray:
    # va = denoise(va, method="median", ksize=3)
    va = remove_background(va.astype(float), method="uniform", wnd=30)
    va = xr.apply_ufunc(
        rebase,
        va,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        dask="parallelized",
        vectorize=True,
    ).astype(np.uint8)
    clahe = cv2.createCLAHE(tileGridSize=(6, 6), clipLimit=1)
    va = xr.apply_ufunc(
        clahe.apply,
        va,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        dask="parallelized",
        vectorize=True,
    )
    return va


#%% load data
va_top = load_videos(
    os.path.join(DPATH, "miniscope_top"), pattern=r"[0-9]+\.avi$", dtype=np.uint8
)
va_side = load_videos(
    os.path.join(DPATH, "miniscope_side"), pattern=r"[0-9]+\.avi$", dtype=np.uint8
)
ts_top = pd.read_csv(os.path.join(DPATH, "miniscope_top", "timeStamps.csv"))
ts_side = pd.read_csv(os.path.join(DPATH, "miniscope_side", "timeStamps.csv"))
ts_top["camNum"] = 1
ts_side["camNum"] = 0
ts = (
    pd.concat([ts_top, ts_side], axis="index")
    .rename({"Frame Number": "frameNum", "Time Stamp (ms)": "sysClock"}, axis="columns")
    .sort_values("sysClock")
    .reset_index(drop=True)
)
ts_map = map_ts(ts).astype({"fmCam0": int, "fmCam1": int})
va_top = va_top.sel(frame=ts_map["fmCam1"].values).chunk({"frame": "auto"})
va_top = va_top.assign_coords(frame=np.arange(va_top.sizes["frame"]))
va_side = va_side.sel(frame=ts_map["fmCam0"].values).chunk({"frame": "auto"})
va_side = va_side.assign_coords(frame=np.arange(va_side.sizes["frame"]))


#%%process array
shutil.rmtree(INTPATH, ignore_errors=True)
print("flipping array and registering")
va_side = xr.apply_ufunc(
    darr.flip,
    va_side,
    input_core_dims=[["frame", "height", "width"]],
    output_core_dims=[["frame", "height", "width"]],
    kwargs={"axis": 1},
    dask="allowed",
)
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
).astype(np.uint8)
va_top = save_minian(va_top.rename("va_top"), INTPATH, overwrite=True)
va_side = save_minian(va_side.rename("va_side"), INTPATH, overwrite=True)
print("pre-processing")
va_top_ps = pre_process(va_top)
va_side_ps = pre_process(va_side)
va_top_ps = save_minian(va_top_ps.rename("va_top_ps"), INTPATH, overwrite=True)
va_side_ps = save_minian(va_side_ps.rename("va_side_ps"), INTPATH, overwrite=True)

#%%plot pre-process result
opts_im = {"cmap": "viridis"}
hv.Image(va_top_ps.isel(frame=10), ["width", "height"]).opts(**opts_im) + hv.Image(
    va_side_ps.isel(frame=10), ["width", "height"]
).opts(**opts_im)

#%%motion correction
print("motion-correction")
sh_top = estimate_motion(va_top_ps.sel(**SUBSET_MC), **param_mc).compute()
# sh_side = estimate_motion(va_side_ps.sel(**SUBSET_MC)).compute()
va_top_mc = apply_transform(va_top_ps, sh_top, fill=0)
va_side_mc = apply_transform(va_side_ps, sh_top, fill=0)
va_top_mc = save_minian(
    va_top_mc.rename("va_top_mc").astype(np.uint8), INTPATH, overwrite=True
)
va_side_mc = save_minian(
    va_side_mc.rename("va_side_mc").astype(np.uint8), INTPATH, overwrite=True
)

#%% plot mc result
ds = open_minian(INTPATH)
va_side_ps = ds["va_side_ps"]
va_side_mc = ds["va_side_mc"]
opts_im = {"cmap": "viridis"}
hv.Image(va_side_ps.max("frame").compute().rename("before")).opts(**opts_im) + hv.Image(
    va_side_mc.max("frame").compute().rename("after")
).opts(**opts_im)
