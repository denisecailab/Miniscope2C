import os
import pickle

import holoviews as hv
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from minian.utilities import TaskAnnotation, save_minian

from routine.alignment import apply_affine, est_affine
from routine.minian_pipeline import minian_process

hv.notebook_extension("bokeh")

IN_DPATH = "validation/data/2color_pilot_tdTomato/m00/2021_08_05/16_22_03"
INT_PATH = "~/var/miniscope_2s/minian_int"
OUT_TX = "validation/store/tx_tdTomato.pkl"
OUT_FIG = "validation/output/tdTomato/alignment.html"
INT_PATH = os.path.abspath(os.path.expanduser(INT_PATH))
PARAM = {
    "load_videos": {"pattern": "\.avi$", "dtype": np.uint8},
    "subset": None,
    "denoise": {"method": "median", "ksize": 3},
    "background_removal": {"method": "tophat", "wnd": 15},
    "estimate_motion": {"dim": "frame"},
}


def align_preprocess(dpath, client, **kwargs):
    save_path = os.path.join(dpath, "minian_ds")
    param = PARAM.copy()
    param["save_minian"] = {"dpath": save_path, "overwrite": True}
    motion, va, va_chk = minian_process(
        dpath,
        INT_PATH,
        param=param,
        return_stage="motion-correction",
        client=client,
        glow_rm=False,
        **kwargs
    )
    max_proj = save_minian(
        va.max("frame").rename("max_proj"), save_path, overwrite=True
    )
    return max_proj.compute()


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
    fm_top = align_preprocess(top_path, client)
    side_path = os.path.join(IN_DPATH, "miniscope_side")
    fm_side = align_preprocess(side_path, client, flip=True)
    # compute alignment
    tx, param_df = est_affine(fm_side.values, fm_top.values, lr=1)
    fm_side_reg = xr.apply_ufunc(
        apply_transform,
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
    hv.save(hv_align, OUT_FIG)
    # save result
    os.makedirs(os.path.dirname(OUT_TX), exist_ok=True)
    with open(OUT_TX, "wb") as pklf:
        pickle.dump(tx, pklf)
