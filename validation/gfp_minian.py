#%% imports and definitions
import os

import numpy as np
from minian.utilities import open_minian

from routine.minian_pipeline import minian_process

IN_DSPATH = "~/var/miniscope_2s/tdTomato/minian_ds"
IN_DSPATH = os.path.normpath(os.path.expanduser(IN_DSPATH))
INTPATH = "~/var/miniscope_2s/minian_int"
INTPATH = os.path.normpath(os.path.expanduser(INTPATH))
os.makedirs(INTPATH, exist_ok=True)

MINIAN_PARAMS = {
    "save_minian": {"meta_dict": None, "overwrite": True, "dpath": IN_DSPATH},
    "load_videos": {
        "pattern": ".*\.avi$",
        "dtype": np.uint8,
        "downsample": dict(frame=1, height=1, width=1),
        "downsample_strategy": "subset",
    },
    "denoise": {"method": "median", "ksize": 5},
    "background_removal": {"method": "tophat", "wnd": 10},
    "estimate_motion": {
        "dim": "frame",
        "aggregation": "max",
        "alt_error": 5,
        "upsample": 10,
    },
    "seeds_init": {
        "wnd_size": 2000,
        "method": "rolling",
        "stp_size": 1000,
        "max_wnd": 20,
        "diff_thres": 5,
    },
    "pnr_refine": {"noise_freq": 0.06, "thres": 1.5},
    "seeds_merge": {"thres_dist": 10, "thres_corr": 0.8, "noise_freq": 0.06},
    "initialize": {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.06},
    "init_merge": {"thres_corr": 0.8},
    "get_noise": {"noise_range": (0.06, 0.5)},
    "first_spatial": {
        "dl_wnd": 3,
        "sparse_penal": 1e-4,
        "update_background": True,
        "size_thres": (25, None),
    },
    "first_temporal": {
        "noise_freq": 0.06,
        "sparse_penal": 0.3,
        "p": 1,
        "add_lag": 20,
        "jac_thres": 0.2,
        "med_wd": 1000,
    },
    "first_merge": {"thres_corr": 0.6},
    "second_spatial": {
        "dl_wnd": 3,
        "sparse_penal": 1e-4,
        "update_background": True,
        "size_thres": (25, None),
    },
    "second_temporal": {
        "noise_freq": 0.06,
        "sparse_penal": 0.3,
        "p": 1,
        "add_lag": 20,
        "jac_thres": 0.4,
        "med_wd": 1000,
    },
}
#%% process gfp channel
if __name__ == "__main__":
    varr = open_minian(IN_DSPATH)["va_side"]
    minian_process(".", INTPATH, 16, MINIAN_PARAMS, varr=varr)
