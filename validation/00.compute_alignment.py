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

IN_DPATH = "validation/data/2color_pilot_mCherry/bench/2022_02_20/18_32_55"
INT_PATH = "~/var/miniscope_2s/minian_int"
OUT_TX = "validation/store/tx_mCherry.pkl"
OUT_FIG = "validation/output/mCherry/alignment.html"
INT_PATH = os.path.abspath(os.path.expanduser(INT_PATH))
PARAM = {
    "load_videos": {"pattern": "\.avi$", "dtype": np.uint8},
    "subset": None,
    "denoise": {"method": "median", "ksize": 3},
    "background_removal": {"method": "tophat", "wnd": 15},
    "estimate_motion": {"dim": "frame"},
}


def align_preprocess(dpath, client, preprocess=True, template="max", **kwargs):
    save_path = os.path.join(dpath, "minian_ds")
    param = PARAM.copy()
    param["save_minian"] = {"dpath": save_path, "overwrite": True}
    ret = "motion-correction" if preprocess else "load"
    out = minian_process(
        dpath,
        INT_PATH,
        param=param,
        return_stage=ret,
        client=client,
        glow_rm=False,
        **kwargs
    )
    if preprocess:
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
        n_workers=16,
        memory_limit="4GB",
        resources={"MEM": 1},
        threads_per_worker=2,
        dashboard_address="0.0.0.0:12345",
    )
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)
    top_path = os.path.join(IN_DPATH, "miniscope_top")
    fm_top = align_preprocess(top_path, client, preprocess=False, template="mean")
    side_path = os.path.join(IN_DPATH, "miniscope_side")
    fm_side = align_preprocess(
        side_path, client, flip=True, preprocess=False, template="mean"
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
    with open(OUT_TX, "wb") as pklf:
        pickle.dump(tx, pklf)
