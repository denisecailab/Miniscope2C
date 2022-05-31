#%% imports and definition
import os

import colorcet as cc
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm

hv.notebook_extension("bokeh")

IN_DSPATH = "store/align_ds.nc"
CLIP = (0, 50)
BRT_OFFSET = 0
OUTPATH = "./output/alignment/alignment"
TITLES = {"top": "Red Channel", "side": "Green Channel", "ovly": "Overlay"}

#%% load data
minian_ds = xr.open_dataset(IN_DSPATH)
fm_side = minian_ds["fm_side_reg"].clip(*CLIP).compute()
fm_top = minian_ds["fm_top"].clip(*CLIP).compute()

#%% generate plot
def plot_im(a, ax):
    ax.imshow(a)
    ax.set_axis_off()


plt.rcParams.update({"axes.titlesize": 11, "font.sans-serif": "Arial"})
aspect = 1.4
# fig, (ax_top, ax_side, ax_ovly, ax_top_max, ax_side_max, ax_ovly_max) = plt.subplots(
#     2, 3, figsize=(8.5, 8.5 / aspect), dpi=500
# )
fig, axs = plt.subplots(1, 3, figsize=(8.5, 8.5 / aspect), dpi=500)
ax_top, ax_side, ax_ovly = axs[0], axs[1], axs[2]
fm_top_pcolor = np.clip(
    cm.ScalarMappable(cmap=cc.m_linear_ternary_red_0_50_c52).to_rgba(fm_top.values)
    + BRT_OFFSET,
    0,
    1,
)
fm_side_pcolor = np.clip(
    cm.ScalarMappable(cmap=cc.m_linear_ternary_green_0_46_c42).to_rgba(fm_side.values)
    + BRT_OFFSET,
    0,
    1,
)
fm_ovly = np.clip(fm_top_pcolor + fm_side_pcolor, 0, 1)
plt.subplots_adjust(0, 0, 1, 1, 0.05, 0.05)
ax_top.set_title(TITLES["top"])
ax_side.set_title(TITLES["side"])
ax_ovly.set_title(TITLES["ovly"])
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
# hv.renderer("bokeh").theme = "dark_minimal"
aspect = fm_top_pcolor.shape[1] / fm_top_pcolor.shape[0]
opts_im = {
    "frame_width": 500,
    "aspect": aspect,
    "xaxis": None,
    "yaxis": None,
    "fontsize": {"title": 15},
}
hv_plt = (
    hv.RGB(fm_top_pcolor, ["width", "height"], label=TITLES["top"]).opts(**opts_im)
    + hv.RGB(fm_side_pcolor, ["width", "height"], label=TITLES["side"]).opts(**opts_im)
    + hv.RGB(fm_ovly, ["width", "height"], label=TITLES["ovly"]).opts(**opts_im)
).cols(3)
os.makedirs(os.path.dirname(OUTPATH), exist_ok=True)
# fig.savefig(os.path.join(OUTPATH, "fov.svg"))
# fig.savefig(os.path.join(OUTPATH, "fov.png"))
# hv.save(hv_plt, os.path.join(OUTPATH, "fov.html"))
hv.save(hv_plt, OUTPATH + ".html")
fig.savefig(OUTPATH + ".svg")

# fig_top, ax_top = plt.subplots(1, 1, dpi=500)
# ax_top.set_title("tdTomato")
# plot_im(fm_top_pcolor, ax_top)
# fig_top.savefig(os.path.join(OUTPATH, "tdTomato.png"))

# fig_side, ax_side = plt.subplots(1, 1, dpi=500)
# ax_side.set_title("GCamp")
# plot_im(fm_side_pcolor, ax_side)
# fig_side.savefig(os.path.join(OUTPATH, "GCamp.png"))

# fig_ovly, ax_ovly = plt.subplots(1, 1, dpi=500)
# ax_ovly.set_title("Overlay")
# plot_im(fm_ovly, ax_ovly)
# fig_ovly.savefig(os.path.join(OUTPATH, "overlay.png"))
