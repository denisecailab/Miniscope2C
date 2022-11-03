import os

import holoviews as hv
import numpy as np
import cv2
from dask.distributed import Client, LocalCluster
from minian.utilities import TaskAnnotation, save_minian

from routine.minian_pipeline import minian_process

hv.notebook_extension("bokeh")

IN_DPATH = "./data/2color_pilot_tdTomato/cv03/2022_06_21/16_01_56/miniscope_top"
INT_PATH = "~/var/miniscope_2s/top_int"
INT_PATH = os.path.abspath(os.path.expanduser(INT_PATH))
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
    "pnr_refine": {"noise_freq": 0.06, "thres": 0},
    "seeds_merge": {"thres_dist": 10, "thres_corr": 0.7, "noise_freq": 0.06},
    "initialize": {"thres_corr": 0.85, "wnd": 10, "noise_freq": 0.06},
    "get_noise": {"noise_range": (0.06, 0.5)},
    "first_spatial": {
        "dl_wnd": 2,
        "sparse_penal": 0,
        "size_thres": (9, 400),
    },
}


def process_static(dpath, client, **kwargs):
    save_path = os.path.join(dpath, "minian_ds")
    param = PARAM.copy()
    param["save_minian"] = {"dpath": save_path, "overwrite": True}
    A, C, b, f = minian_process(
        dpath,
        INT_PATH,
        param=param,
        client=client,
        return_stage="first-spatial",
        **kwargs,
    )
    save_minian(A.rename("A"), save_path, overwrite=True)
    save_minian(C.rename("C"), save_path, overwrite=True)
    save_minian(b.rename("b"), save_path, overwrite=True)
    save_minian(f.rename("f"), save_path, overwrite=True)


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
    process_static(IN_DPATH, client, glow_rm=False)
    client.close()
    cluster.close()
