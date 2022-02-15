import os

import cv2
import numpy as np
import pandas as pd
import xarray as xr
from minian.utilities import load_videos


def norm(x: np.ndarray):
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    diff = xmax - xmin
    if diff > 0:
        return (x - xmin) / diff
    else:
        return x


def norm_xr(x: xr.DataArray):
    xmin = x.min().compute().values
    xmax = x.max().compute().values
    diff = xmax - xmin
    if diff > 0:
        return (x - xmin) / diff
    else:
        return x


def load_v4_folder(dpath, folders=["miniscope_top", "miniscope_side"]):
    va0 = load_videos(
        os.path.join(dpath, folders[0]), pattern=r"[0-9]+\.avi$", dtype=np.uint8
    )
    va1 = load_videos(
        os.path.join(dpath, folders[1]), pattern=r"[0-9]+\.avi$", dtype=np.uint8
    )
    ts0 = pd.read_csv(os.path.join(dpath, folders[0], "timeStamps.csv"))
    ts1 = pd.read_csv(os.path.join(dpath, folders[1], "timeStamps.csv"))
    ts0["camNum"] = 0
    ts1["camNum"] = 1
    ts = (
        pd.concat([ts0, ts1], axis="index")
        .rename(
            {"Frame Number": "frameNum", "Time Stamp (ms)": "sysClock"}, axis="columns"
        )
        .sort_values("sysClock")
        .reset_index(drop=True)
    )
    ts_map = map_ts(ts).dropna().astype({"fmCam0": int, "fmCam1": int})
    va0 = va0.sel(frame=ts_map["fmCam0"].values).chunk({"frame": "auto"})
    va0 = va0.assign_coords(frame=np.arange(va0.sizes["frame"]))
    va1 = va1.sel(frame=ts_map["fmCam1"].values).chunk({"frame": "auto"})
    va1 = va1.assign_coords(frame=np.arange(va1.sizes["frame"]))
    return va0, va1


def map_ts(ts: pd.DataFrame) -> pd.DataFrame:
    """map frames from Cam1 to Cam0 with nearest neighbour using the timestamp
    file from miniscope recordings.

    Parameters
    ----------
    ts : pd.DataFrame
        input timestamp dataframe. should contain field 'frameNum', 'camNum' and
        'sysClock'

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
    return np.clip(x, np.quantile(x, q), None)


def bin_fm(fm: np.ndarray, **kwargs):
    return cv2.adaptiveThreshold((norm(fm) * 255).astype(np.uint8), **kwargs).astype(
        fm.dtype
    )
