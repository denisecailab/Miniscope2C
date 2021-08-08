#%% import and definitions
import os

import cv2
import numpy as np
import xarray as xr
from routine.utilities import open_minian, norm, write_video

ANNT_COL = (255, 255, 255)
INTPATH = "./intermediate"
OUTPATH = "./output/result.mp4"


def video_annt(fm, title):
    fm = cv2.cvtColor(fm, cv2.COLOR_GRAY2RGB)
    fm = cv2.putText(fm, title, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, ANNT_COL, 2)
    fm = cv2.circle(fm, (550, 550), 6, ANNT_COL, -1)
    fm = cv2.putText(
        fm, "20 um", (520, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ANNT_COL, 2
    )
    return cv2.cvtColor(fm, cv2.COLOR_RGB2GRAY)


#%% generate video
ds = open_minian(INTPATH)
va_top = ds["va_top"].astype(float)
va_side = ds["va_side_reg"].astype(float)
diff = norm(va_top - va_side) * 255
va_top = norm(va_top) * 255
va_side = norm(va_side) * 255
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
os.makedirs(os.path.split(OUTPATH)[0], exist_ok=True)
write_video(
    xr.concat([va_top_out, va_side_out, diff], "width"),
    os.path.split(OUTPATH)[1],
    os.path.split(OUTPATH)[0],
)
