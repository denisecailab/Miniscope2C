#%% imports and definition
import os

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

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
    "res": {"flip": True, "clip_red": (0, 80), "clip_green": (0, 100)},
    "res-grin": {"clip_red": (0, 100), "clip_green": (0, 180)},
}
PARAM_BRT_OFFSET = 0
PARAM_SUBSET = {"height": slice(100 - 25, 349 + 25), "width": slice(240 - 25, 479 + 25)}
FIG_PATH = "./output/overlap"

plt.rcParams.update(**PARAM_FONT)
os.makedirs(FIG_PATH, exist_ok=True)

#%% load data
ss_csv = pd.read_csv(IN_SS_CSV)
ss_csv = ss_csv[ss_csv["session"].notnull()].copy()
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
        im_red = im_red.clip(*clip_green)
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
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_PATH, "{}-{}.svg".format(anm, ss)), bbox_inches="tight"
    )
    plt.close(fig)
