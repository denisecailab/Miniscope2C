import os

import holoviews as hv
import numpy as np
from dask.distributed import Client, LocalCluster
from minian.utilities import TaskAnnotation, save_minian

from routine.minian_pipeline import minian_process

hv.notebook_extension("bokeh")

IN_DPATH = "validation/data/2color_pilot_tdTomato/m00/2021_08_05/16_22_03/miniscope_top"
INT_PATH = "~/var/miniscope_2s/minian_int"
INT_PATH = os.path.abspath(os.path.expanduser(INT_PATH))
PARAM = {
    "save_minian": {"meta_dict": None, "overwrite": True},
    "load_videos": {
        "pattern": ".*\.avi$",
        "dtype": np.uint8,
        "downsample": dict(frame=1, height=1, width=1),
        "downsample_strategy": "subset",
    },
    "subset": None,
    "denoise": {"method": "median", "ksize": 5},
    "background_removal": {"method": "tophat", "wnd": 15},
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
        "diff_thres": 6,
    },
    "pnr_refine": {"noise_freq": 0.06, "thres": 1},
    "seeds_merge": {"thres_dist": 10, "thres_corr": 0.7, "noise_freq": 0.06},
    "initialize": {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.06},
    "init_merge": {"thres_corr": 0.8},
    "get_noise": {"noise_range": (0.06, 0.5)},
    "first_spatial": {
        "dl_wnd": 5,
        "sparse_penal": 0.005,
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
        "sparse_penal": 1e-3,
        "size_thres": (25, None),
    },
    "second_temporal": {
        "noise_freq": 0.06,
        "sparse_penal": 0.5,
        "p": 1,
        "add_lag": 20,
        "jac_thres": 0.4,
        "med_wd": 1000,
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
        n_workers=16,
        memory_limit="4GB",
        resources={"MEM": 1},
        threads_per_worker=2,
        dashboard_address="0.0.0.0:12345",
    )
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)
    process_static(IN_DPATH, client)
    client.close()
    cluster.close()
