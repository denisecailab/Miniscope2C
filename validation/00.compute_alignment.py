import os
import pickle

import holoviews as hv
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from minian.utilities import TaskAnnotation

from routine.alignment import apply_affine, est_affine
from routine.minian_pipeline import minian_process

hv.notebook_extension("bokeh")

IN_DPATH = "validation/data/2color_pilot_mCherry/bench/2022_02_21/13_02_06"
INT_PATH = "~/var/miniscope_2s/minian_int"
WORKER_PATH = "~/var/miniscope_2s/dask-worker-space"
OUT_TX = "validation/store/tx_mCherry.pkl"
OUT_DS = "validation/store/align_ds.nc"
OUT_FIG = "validation/output/mCherry/alignment.html"
INT_PATH = os.path.abspath(os.path.expanduser(INT_PATH))
WORKER_PATH = os.path.abspath(os.path.expanduser(WORKER_PATH))
PARAM = {
    "load_videos": {"pattern": "\.avi$", "dtype": np.uint8},
    "subset": None,
    "denoise": {"method": "median", "ksize": 3},
    "background_removal": {"method": "uniform", "wnd": 50},
    "estimate_motion": {"dim": "frame"},
}


def align_preprocess(
    dpath, client, return_stage="motion-correction", template="max", **kwargs
):
    save_path = os.path.join(dpath, "minian_ds")
    param = PARAM.copy()
    param["save_minian"] = {"dpath": save_path, "overwrite": True}
    out = minian_process(
        dpath,
        INT_PATH,
        param=param,
        return_stage=return_stage,
        client=client,
        glow_rm=False,
        **kwargs
    )
    if return_stage == "motion-correction":
        va = out[1]
    else:
        va = out
    if template == "max":
        sum_fm = va.max("frame")
    elif template == "mean":
        sum_fm = va.mean("frame")
    return sum_fm.rename("sum_fm").compute()


if __name__ == "__main__":
    # load data and preprocess
    cluster = LocalCluster(
        n_workers=4,
        memory_limit="16GB",
        resources={"MEM": 1},
        threads_per_worker=2,
        dashboard_address="0.0.0.0:12345",
        local_directory=WORKER_PATH,
    )
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)
    top_path = os.path.join(IN_DPATH, "miniscope_top")
    fm_top = align_preprocess(
        top_path, client, return_stage="preprocessing", template="mean"
    )
    side_path = os.path.join(IN_DPATH, "miniscope_side")
    fm_side = align_preprocess(
        side_path, client, flip=True, return_stage="preprocessing", template="mean"
    )
    # compute alignment
    tx, param_df = est_affine(fm_side.values, fm_top.values, lr=1)
    fm_side_reg = xr.apply_ufunc(
        apply_affine,
        fm_side,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        kwargs={"tx": tx},
    )
    # plot result
    im_opts = {"cmap": "viridis"}
    hv_align = (
        hv.Image(fm_top, ["width", "height"], label="top").opts(**im_opts)
        + hv.Image(fm_side, ["width", "height"], label="side").opts(**im_opts)
        + hv.Image(fm_side_reg, ["width", "height"], label="side_reg").opts(**im_opts)
        + hv.Image((fm_side_reg - fm_top), ["width", "height"], label="affine").opts(
            **im_opts
        )
    ).cols(2)
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    hv.save(hv_align, OUT_FIG)
    # save result
    os.makedirs(os.path.dirname(OUT_TX), exist_ok=True)
    ds = xr.merge(
        [
            fm_top.rename("fm_top"),
            fm_side.rename("fm_side"),
            fm_side_reg.rename("fm_side_reg"),
        ]
    )
    ds.to_netcdf(OUT_DS)
    with open(OUT_TX, "wb") as pklf:
        pickle.dump(tx, pklf)
