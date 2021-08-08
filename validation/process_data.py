#%% imports and definitions
import os
import shutil

import dask.array as darr
import holoviews as hv
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

from .routine.motion_correction import (
    apply_transform,
    est_motion_perframe,
    estimate_motion,
    transform_perframe,
)
from .routine.preprocessing import denoise, remove_background
from .routine.utilities import load_videos, norm, open_minian, save_minian

pbar = ProgressBar(minimum=2)
pbar.register()
hv.notebook_extension("bokeh")

DPATH = "./data/2color_pilot_tdTomato/2C1/2021_05_28/13_50_04"
REFPATH = "./data/2color_pilot_tdTomato/2C1/2021_05_27/15_40_06/miniscope_top"
INTPATH = "./intermediate"
IN_SH = "./store/shift.npy"
SUBSET_MC = {"height": slice(130, 330), "width": slice(100, 520)}
param_denoise = {"method": "median", "ksize": 7}
param_background_removal = {"method": "tophat", "wnd": 15}
param_motion_correction = {"alt_error": 2, "aggregation": "max"}


def map_ts(ts: pd.DataFrame) -> pd.DataFrame:
    """map frames from Cam1 to Cam0 with nearest neighbour using the timestamp file from miniscope recordings.

    Parameters
    ----------
    ts : pd.DataFrame
        input timestamp dataframe. should contain field 'frameNum', 'camNum' and 'sysClock'

    Returns
    -------
    pd.DataFrame
        output dataframe. should contain field 'fmCam0' and 'fmCam1'
    """
    ts_sort = ts.sort_values("sysClock")
    ts_sort["ts_behav"] = np.where(ts_sort["camNum"] == 1, ts_sort["sysClock"], np.nan)
    ts_sort["ts_forward"] = ts_sort["ts_behav"].fillna(method="ffill")
    ts_sort["ts_backward"] = ts_sort["ts_behav"].fillna(method="bfill")
    ts_sort["diff_forward"] = np.absolute(ts_sort["sysClock"] - ts_sort["ts_forward"])
    ts_sort["diff_backward"] = np.absolute(ts_sort["sysClock"] - ts_sort["ts_backward"])
    ts_sort["fm_behav"] = np.where(ts_sort["camNum"] == 1, ts_sort["frameNum"], np.nan)
    ts_sort["fm_forward"] = ts_sort["fm_behav"].fillna(method="ffill")
    ts_sort["fm_backward"] = ts_sort["fm_behav"].fillna(method="bfill")
    ts_sort["fmCam1"] = np.where(
        ts_sort["diff_forward"] < ts_sort["diff_backward"],
        ts_sort["fm_forward"],
        ts_sort["fm_backward"],
    )
    ts_map = (
        ts_sort[ts_sort["camNum"] == 0][["frameNum", "fmCam1"]]
        .dropna()
        .rename(columns=dict(frameNum="fmCam0"))
        .astype(dict(fmCam1=int))
    )
    return ts_map


def lst_sq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert x.shape == y.shape
    x, y = x.reshape(-1), y.reshape(-1)
    X = np.stack([np.ones_like(x), x], axis=1)
    return np.linalg.lstsq(X, y, rcond=None)[0]


def rebase(x: np.ndarray, q: float = 0.1) -> np.ndarray:
    return np.clip(x, max(np.quantile(x, q), 0), None)


def pre_process(va: xr.DataArray) -> xr.DataArray:
    va = denoise(va, **param_denoise)
    va = remove_background(va.astype(float), method="uniform", wnd=30)
    va = xr.apply_ufunc(
        rebase,
        va,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        dask="parallelized",
        vectorize=True,
    ).astype(np.uint8)
    va = remove_background(va, **param_background_removal)
    # return (norm(va) * 255).astype(np.uint8)
    return va


#%% load data
va_top = load_videos(
    os.path.join(DPATH, "miniscope_top"), pattern=r"[0-9]+\.avi$", dtype=np.uint8
)
va_side = load_videos(
    os.path.join(DPATH, "miniscope_side"), pattern=r"[0-9]+\.avi$", dtype=np.uint8
)
va_ref = load_videos(REFPATH, pattern=r"[0-9]+\.avi$", dtype=np.uint8)
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
print("flipping array")
va_side = xr.apply_ufunc(
    darr.flip,
    va_side,
    input_core_dims=[["frame", "height", "width"]],
    output_core_dims=[["frame", "height", "width"]],
    kwargs={"axis": 1},
    dask="allowed",
)
va_top = save_minian(va_top.rename("va_top"), INTPATH, overwrite=True)
va_side = save_minian(va_side.rename("va_side"), INTPATH, overwrite=True)
va_ref = save_minian(va_ref.rename("va_ref"), INTPATH, overwrite=True)
print("pre-processing")
va_top_ps = pre_process(va_top)
va_side_ps = pre_process(va_side)
va_ref_ps = pre_process(va_ref)
va_top_ps = save_minian(va_top_ps.rename("va_top_ps"), INTPATH, overwrite=True)
va_side_ps = save_minian(va_side_ps.rename("va_side_ps"), INTPATH, overwrite=True)
va_ref_ps = save_minian(va_ref_ps.rename("va_ref_ps"), INTPATH, overwrite=True)
print("motion-correction")
sh_top = estimate_motion(va_top_ps.sel(**SUBSET_MC)).compute()
# sh_side = estimate_motion(va_side_ps.sel(**SUBSET_MC)).compute()
va_top_mc = apply_transform(va_top_ps, sh_top, fill=0)
va_side_mc = apply_transform(va_side_ps, sh_top, fill=0)
va_top_mc = save_minian(
    va_top_mc.rename("va_top_mc").astype(np.uint8), INTPATH, overwrite=True
)
va_side_mc = save_minian(
    va_side_mc.rename("va_side_mc_unreg").astype(np.uint8), INTPATH, overwrite=True
)
print("registering channels")
fm_top = va_top_mc.compute().median("frame")
fm_side = va_side_mc.max("frame").compute()
sh = est_motion_perframe(fm_side.values, fm_top.values, upsample=100)
va_side_mc = xr.apply_ufunc(
    transform_perframe,
    va_side_mc,
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    vectorize=True,
    kwargs={"tx_coef": sh, "fill": 0},
    dask="parallelized",
    output_dtypes=float,
)
va_side_mc = save_minian(
    va_side_mc.rename("va_side_mc").astype(np.uint8), INTPATH, overwrite=True
)

#%% plot mc result
ds = open_minian(INTPATH)
va_side_ps = ds["va_top_ps"]
va_side_mc = ds["va_top_mc"]
opts_im = {"cmap": "viridis"}
hv.Image(va_side_ps.max("frame").compute().rename("before")).opts(**opts_im) + hv.Image(
    va_side_mc.max("frame").compute().rename("after")
).opts(**opts_im)


#%% demixing
ds = open_minian(INTPATH)
va_top_mc = ds["va_top_mc"]
va_side_mc = ds["va_side_mc"]
med_top = va_top_mc.compute().median("frame")
med_side = va_side_mc.compute().median("frame")
va_side_dm = norm((va_side_mc.astype(float) - med_side).clip(0, None)) * 255
va_side_dm = save_minian(
    va_side_dm.rename("va_side_dm").astype(np.uint8), INTPATH, overwrite=True
)
bleed_est = (va_top_mc.astype(float) - med_top).clip(0, None)
fm_idx = np.sort(np.random.choice(bleed_est.coords["frame"].values, 1000))
coefs = xr.apply_ufunc(
    lst_sq,
    va_side_dm.sel(frame=fm_idx).compute(),
    bleed_est.sel(frame=fm_idx).compute(),
    input_core_dims=[["frame", "height", "width"], ["frame", "height", "width"]],
    output_core_dims=[["variable"]],
).values
bleed_true = va_side_dm * coefs[1] + coefs[0]
va_top_dm = norm((va_top_mc.astype(float) - bleed_true).clip(0, None)) * 255
va_top_dm = save_minian(
    va_top_dm.rename("va_top_dm").astype(np.uint8), INTPATH, overwrite=True
)
