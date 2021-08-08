#%% imports and definition
import os

import colorcet as cc
import cv2
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm
from routine.utilities import open_minian, norm

hv.notebook_extension("bokeh")

INTPATH = "./intermediate"
CLIP_LOW_SIDE = 0.035
CLIP_LOW_TOP = 0.03
CLIP_HIGH = 0.95
BRT_OFFSET = 0.1
SUBSET = dict()
OUTPATH = "./output"
param_ahe = {"tileGridSize": (20, 20), "clipLimit": 20}
param_background = {"method": "tophat", "wnd": 10}


def adaptive_thres(fm: np.ndarray, **kwargs):
    clahe = cv2.createCLAHE(**kwargs)
    return clahe.apply(fm)


def process_frame(fm: np.ndarray, clip_high: float, clip_low: float):
    fm = norm(fm)
    # selem = disk(param_background["wnd"])
    # fm = remove_background_perframe(fm, selem=selem, **param_background)
    fm = (norm(np.clip(fm, clip_low, clip_high)) * 255).astype(np.uint8)
    fm = adaptive_thres(fm, **param_ahe)
    return fm


#%% load data
ds = open_minian(INTPATH)
va_top = ds["va_top_mc"]
va_side = ds["va_side_dm"]
fm_top = va_top.sel(**SUBSET).compute().median("frame")
fm_side = va_side.sel(**SUBSET).max("frame").compute()

#%% process frame
fm_top_ps = xr.apply_ufunc(
    process_frame,
    fm_top,
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    kwargs={"clip_high": CLIP_HIGH, "clip_low": CLIP_LOW_TOP},
).astype(float)
fm_side_ps = xr.apply_ufunc(
    process_frame,
    fm_side,
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    kwargs={"clip_high": CLIP_HIGH, "clip_low": CLIP_LOW_SIDE},
).astype(float)
fm_side_reg = fm_side_ps

#%% generate plot
def plot_im(a, ax):
    ax.imshow(a)
    ax.set_axis_off()


# fm_top_clp = norm(fm_top.clip(CLIP_LOW, CLIP_HIGH))
fm_top_clp = norm(fm_top_ps)
# fm_side_clp = norm(fm_side.clip(CLIP_LOW, CLIP_HIGH))
fm_side_clp = norm(fm_side_ps)
# fm_side_reg_clp = norm(fm_side_reg.clip(CLIP_LOW, CLIP_HIGH))
fm_side_reg_clp = norm(fm_side_reg)
plt.rcParams.update({"axes.titlesize": 11, "font.sans-serif": "Arial"})
aspect = 2.2
fig, (ax_top, ax_side, ax_ovly) = plt.subplots(
    1, 3, figsize=(8.5, 8.5 / aspect), dpi=500
)
fm_top_pcolor = np.clip(
    cm.ScalarMappable(cmap=cc.m_linear_ternary_red_0_50_c52).to_rgba(fm_top_clp.values)
    + BRT_OFFSET,
    0,
    1,
)
fm_side_pcolor = np.clip(
    cm.ScalarMappable(cmap=cc.m_linear_ternary_green_0_46_c42).to_rgba(
        fm_side_clp.values
    )
    + BRT_OFFSET,
    0,
    1,
)
fm_side_pcolor_reg = cm.ScalarMappable(cmap=cc.m_linear_ternary_green_0_46_c42).to_rgba(
    fm_side_reg_clp.fillna(0).values
)
fm_ovly = np.clip(fm_top_pcolor + fm_side_pcolor_reg, 0, 1)
plt.subplots_adjust(0, 0, 1, 1, 0.05, 0.05)
ax_top.set_title("tdTomato")
ax_side.set_title("GCamp")
ax_ovly.set_title("Overlay")
plot_im(fm_top_pcolor, ax_top)
plot_im(fm_side_pcolor, ax_side)
plot_im(fm_ovly, ax_ovly)
# rect = Rectangle(
#     xy=(0, 0),
#     height=fm_ovly.shape[0] - sh[0],
#     width=fm_ovly.shape[1] - sh[1],
#     edgecolor="white",
#     fill=False,
#     linestyle=":",
#     linewidth=0.7,
# )
# ax_ovly.add_patch(rect)
hv.renderer("bokeh").theme = "dark_minimal"
aspect = fm_top_pcolor.shape[1] / fm_top_pcolor.shape[0]
opts_im = {
    "frame_width": 480,
    "aspect": aspect,
    "xaxis": None,
    "yaxis": None,
    "fontsize": {"title": 20},
}
hv_plt = (
    hv.RGB(fm_top_pcolor, ["width", "height"], label="tdTomato").opts(**opts_im)
    + hv.RGB(fm_side_pcolor, ["width", "height"], label="GCamp").opts(**opts_im)
    + hv.RGB(fm_ovly, ["width", "height"], label="Overlay").opts(**opts_im)
)
os.makedirs(OUTPATH, exist_ok=True)
fig.savefig(os.path.join(OUTPATH, "fov.svg"))
fig.savefig(os.path.join(OUTPATH, "fov.png"))
hv.save(hv_plt, os.path.join(OUTPATH, "fov.html"))
