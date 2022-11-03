#%% import and definitions
import os

import cv2
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from routine.utilities import norm_xr, open_minian, save_minian, write_video

ANNT_COL = (255, 255, 255)
IN_DSPATH = "~/var/miniscope_2s/tdTomato/minian_ds"
IN_DSPATH = os.path.normpath(os.path.expanduser(IN_DSPATH))
OUT_DSPATH = IN_DSPATH
OUT_PATH = "./output/tdTomato/result.mp4"
pbar = ProgressBar(minimum=2)
pbar.register()


def video_annt(fm, title):
    fm = cv2.cvtColor(fm, cv2.COLOR_GRAY2RGB)
    fm = cv2.putText(fm, title, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, ANNT_COL, 2)
    fm = cv2.circle(fm, (550, 550), 6, ANNT_COL, -1)
    fm = cv2.putText(
        fm, "20 um", (520, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ANNT_COL, 2
    )
    return cv2.cvtColor(fm, cv2.COLOR_RGB2GRAY)


#%% generate video
ds = open_minian(IN_DSPATH)
va_top = ds["va_top_mc"]
va_side = ds["va_side_dm"]
va_top = save_minian(
    (norm_xr(va_top) * 255).astype(np.uint8).rename("va_top_out"),
    OUT_DSPATH,
    overwrite=True,
)
va_side = save_minian(
    (norm_xr(va_side) * 255).astype(np.uint8).rename("va_side_out"),
    OUT_DSPATH,
    overwrite=True,
)
diff = save_minian(
    (norm_xr(va_top.astype(float) - va_side.astype(float)) * 255)
    .astype(np.uint8)
    .rename("va_diff_out"),
    OUT_DSPATH,
    overwrite=True,
)
va_top_out = xr.apply_ufunc(
    video_annt,
    va_top.clip(0, 255).astype(np.uint8),
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    vectorize=True,
    kwargs={"title": "tdTomato"},
    dask="parallelized",
)
va_side_out = xr.apply_ufunc(
    video_annt,
    va_side.clip(0, 255).astype(np.uint8),
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    vectorize=True,
    kwargs={"title": "GFP"},
    dask="parallelized",
)
diff = xr.apply_ufunc(
    video_annt,
    diff.clip(0, 255).astype(np.uint8),
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    vectorize=True,
    kwargs={"title": "diff"},
    dask="parallelized",
)
os.makedirs(os.path.split(OUT_PATH)[0], exist_ok=True)
write_video(
    xr.concat([va_top_out, va_side_out, diff], "width"),
    os.path.split(OUT_PATH)[1],
    os.path.split(OUT_PATH)[0],
)
