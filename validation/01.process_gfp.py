import os
import pickle

import numpy as np
from dask.distributed import Client, LocalCluster
from minian.utilities import TaskAnnotation
from routine.minian_pipeline import minian_process

IN_DPATH = "./data/2color_pilot_tdTomato/cv03/2022_06_21/16_01_56/miniscope_side"
INT_PATH = "~/var/miniscope_2s/side_int"
INT_PATH = os.path.abspath(os.path.expanduser(INT_PATH))
IN_TX = "./store/alignment/cv03-rec0.pkl"
PARAM = {
    "save_minian": {"meta_dict": None, "overwrite": True},
    "load_videos": {
        "pattern": ".*\.avi$",
        "dtype": np.uint8,
        "downsample": dict(frame=1, height=1, width=1),
        "downsample_strategy": "subset",
    },
    "subset": {"frame": slice(0, 2999)},
    "denoise": {"method": "median", "ksize": 3},
    "background_removal": {"method": "uniform", "wnd": 50},
    "background_removal_it2": {"method": "tophat", "wnd": 15},
    "estimate_motion": {
        "dim": "frame",
        "aggregation": "mean",
        "alt_error": 5,
        "upsample": 100,
    },
    "seeds_init": {
        "wnd_size": 1000,
        "method": "rolling",
        "stp_size": 500,
        "max_wnd": 15,
        "diff_thres": 8,
    },
    "pnr_refine": {"noise_freq": 0.06, "thres": 1.5},
    "seeds_merge": {"thres_dist": 15, "thres_corr": 0.75, "noise_freq": 0.06},
    "initialize": {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.06},
    "init_merge": {"thres_corr": 0.8},
    "get_noise": {"noise_range": (0.06, 0.5)},
    "first_spatial": {
        "dl_wnd": 10,
        "sparse_penal": 0.01,
        "size_thres": (25, None),
    },
    "first_temporal": {
        "noise_freq": 0.06,
        "sparse_penal": 1,
        "p": 1,
        "add_lag": 20,
        "jac_thres": 0.2,
        "med_wd": 1000,
    },
    "first_merge": {"thres_corr": 0.8},
    "second_spatial": {
        "dl_wnd": 10,
        "sparse_penal": 5e-3,
        "size_thres": (25, None),
    },
    "second_temporal": {
        "noise_freq": 0.06,
        "sparse_penal": 1,
        "p": 1,
        "add_lag": 20,
        "jac_thres": 0.4,
        "med_wd": 1000,
    },
}


def process_gfp(dpath, client, **kwargs):
    save_path = os.path.join(dpath, "minian_ds")
    param = PARAM.copy()
    param["save_minian"] = {"dpath": save_path, "overwrite": True}
    minian_process(dpath, INT_PATH, param=param, client=client, **kwargs)


if __name__ == "__main__":
    cluster = LocalCluster(
        n_workers=8,
        memory_limit="8GB",
        resources={"MEM": 1},
        threads_per_worker=2,
        dashboard_address="0.0.0.0:12345",
    )
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)
    with open(IN_TX, "rb") as pklf:
        tx = pickle.load(pklf)
    process_gfp(IN_DPATH, client, flip=True, tx=tx, glow_rm=False)
    client.close()
    cluster.close()
