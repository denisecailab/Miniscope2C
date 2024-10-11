# %% imports and definition
import os

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from routine.plotting import pcolor_ovly

IN_GREEN_PATH = "./store/alignment"
IN_RED_PATH = "./store/alignment"
IN_GREEN_VAR_NAME = "fm_side_reg"
IN_RED_VAR_NAME = "fm_top"
IN_SS_CSV = "./data/sessions.csv"
PARAM_ASPECT = 1.4
PARAM_FONT = {"axes.titlesize": 11, "font.sans-serif": "Arial"}
PARAM_TITLES = {"red": "Red Channel", "green": "Green Channel", "ovly": "Overlay"}
PARAM_PS = {
    "res": {"flip": True, "clip_red": (0, 40), "clip_green": (0, 50)},
    "res-grin": {"clip_red": (0, 60), "clip_green": (0, 120)},
}
PARAM_BRT_OFFSET = 0
PARAM_SUBSET = {"height": slice(100 - 25, 349 + 25), "width": slice(240 - 25, 479 + 25)}
FIG_PATH = "./output/overlap"

plt.rcParams.update(**PARAM_FONT)
os.makedirs(FIG_PATH, exist_ok=True)

# %% load data and plot
ss_csv = pd.read_csv(IN_SS_CSV)
ss_csv = ss_csv[ss_csv["session"].notnull()].copy()
zm_ext = {
    "res": [(322, 402, 343, 423), (455, 535, 148, 228)],
    "res-grin": [(295, 375, 285, 365), (420, 500, 90, 170)],
}
for _, row in ss_csv.iterrows():
    anm, ss = row["animal"], row["session"]
    green_ds = xr.open_dataset(os.path.join(IN_GREEN_PATH, "{}-{}.nc".format(anm, ss)))
    red_ds = xr.open_dataset(os.path.join(IN_RED_PATH, "{}-{}.nc".format(anm, ss)))
    im_red, im_green = red_ds[IN_RED_VAR_NAME], green_ds[IN_GREEN_VAR_NAME]
    param = PARAM_PS[ss]
    if param.get("flip", False):
        im_red = im_red.isel(height=slice(None, None, -1), width=slice(None, None, -1))
        im_green = im_green.isel(
            height=slice(None, None, -1), width=slice(None, None, -1)
        )
    clip_red = param.get("clip_red", None)
    if clip_red is not None:
        im_red = im_red.clip(*clip_red)
    clip_green = param.get("clip_green", None)
    if clip_green is not None:
        im_green = im_green.clip(*clip_green)
    im_red, im_green, im_ovly = pcolor_ovly(
        im_red, im_green, brt_offset=PARAM_BRT_OFFSET
    )
    fig, axs = plt.subplots(1, 3, figsize=(5.5, 5.5 / PARAM_ASPECT), dpi=800)
    ax_dict = {"red": axs[0], "green": axs[1], "ovly": axs[2]}
    im_dict = {"red": im_red, "green": im_green, "ovly": im_ovly}
    for aname, ax in ax_dict.items():
        ax.set_title(PARAM_TITLES[aname])
        ax.imshow(im_dict[aname])
        ax.set_axis_off()
    scalebar = AnchoredSizeBar(
        ax.transData,
        50 / 1.55,
        r"$50 \mu m$",
        "lower left",
        pad=0.1,
        sep=4,
        color="white",
        frameon=False,
        size_vertical=2,
    )
    ax_dict["ovly"].add_artist(scalebar)
    for iext, ext in enumerate(zm_ext[ss]):
        x1, x2, y1, y2 = ext
        axins = ax_dict["ovly"].inset_axes(
            (1.03, iext * 0.53, 0.47, 0.47),
            xlim=(x1, x2),
            ylim=(y2, y1),
            xticklabels=[],
            yticklabels=[],
        )
        axins.set_axis_off()
        axins.imshow(im_dict["ovly"], origin="lower", aspect="auto")
        box = ax_dict["ovly"].indicate_inset(
            (x1, y1, x2 - x1, y2 - y1),
            edgecolor="white",
            transform=ax_dict["ovly"].transData,
            alpha=1,
            lw=0.8,
        )
        cp1 = ConnectionPatch(
            xyA=(x2, y1),
            xyB=(0, 1),
            axesA=ax_dict["ovly"],
            axesB=axins,
            coordsA="data",
            coordsB="axes fraction",
            lw=0.8,
            color="silver",
            ls=(0, (1, 3)),
        )
        cp2 = ConnectionPatch(
            xyA=(x2, y2),
            xyB=(0, 0),
            axesA=ax_dict["ovly"],
            axesB=axins,
            coordsA="data",
            coordsB="axes fraction",
            lw=0.8,
            color="silver",
            ls=(0, (1, 3)),
        )
        ax_dict["ovly"].add_patch(cp1)
        ax_dict["ovly"].add_patch(cp2)
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_PATH, "{}-{}.svg".format(anm, ss)), bbox_inches="tight"
    )
    plt.close(fig)
