import os
import pickle

import holoviews as hv
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client, LocalCluster
from minian.utilities import TaskAnnotation
from routine.alignment import apply_affine, est_affine
from routine.minian_pipeline import align_preprocess

hv.notebook_extension("bokeh")

IN_DPATH = "./data"
IN_SS_CSV = "./data/sessions.csv"
INT_PATH = "~/var/miniscope_2s/minian_int"
WORKER_PATH = "~/var/miniscope_2s/dask-worker-space"
# OUT_TX = "./store/tx_tdTomato.pkl"
# OUT_DS = "./store/align_ds.nc"
OUT_PATH = "./store/alignment"
# OUT_FIG = "./output/tdTomato/alignment.html"
FIG_PATH = "./output/alignment"
INT_PATH = os.path.abspath(os.path.expanduser(INT_PATH))
WORKER_PATH = os.path.abspath(os.path.expanduser(WORKER_PATH))
PARAM = {
    "load_videos": {"pattern": "\.avi$", "dtype": np.uint8},
    "subset": None,
    "denoise": {"method": "median", "ksize": 3},
    # "background_removal": {"method": "uniform", "wnd": 50},
    "estimate_motion": {"dim": "frame"},
}
PARAM_SPECIFIC = {
    "res": {"subset": {"frame": slice(0, 1000)}},
    "res-grin": {"subset": {"frame": slice(500, 510)}},
}
os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)


if __name__ == "__main__":
    ss_csv = pd.read_csv(IN_SS_CSV)
    ss_csv = ss_csv[ss_csv["session"].notnull()].copy()
    for _, row in ss_csv.iterrows():
        # handle paths
        dpath = os.path.join(
            IN_DPATH, row["experiment"], row["animal"], row["date"], row["time"]
        )
        ss = row["session"]
        anm = row["animal"]
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
        param = PARAM.copy()
        param.update(PARAM_SPECIFIC[ss])
        top_path = os.path.join(dpath, "miniscope_top")
        fm_top = align_preprocess(
            top_path,
            client,
            INT_PATH,
            param=param,
            return_stage="preprocessing",
            template="mean",
        )
        side_path = os.path.join(dpath, "miniscope_side")
        fm_side = align_preprocess(
            side_path,
            client,
            INT_PATH,
            param=param,
            flip=True,
            return_stage="preprocessing",
            template="mean",
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
            + hv.Image(fm_side_reg, ["width", "height"], label="side_reg").opts(
                **im_opts
            )
            + hv.Image(
                (fm_side_reg - fm_top), ["width", "height"], label="affine"
            ).opts(**im_opts)
        ).cols(2)
        hv.save(hv_align, os.path.join(FIG_PATH, "{}-{}.html".format(anm, ss)))
        # save result
        ds = xr.merge(
            [
                fm_top.rename("fm_top"),
                fm_side.rename("fm_side"),
                fm_side_reg.rename("fm_side_reg"),
            ]
        )
        ds.to_netcdf(os.path.join(OUT_PATH, "{}-{}.nc".format(anm, ss)))
        with open(os.path.join(OUT_PATH, "{}-{}.pkl".format(anm, ss)), "wb") as pklf:
            pickle.dump(tx, pklf)
        cluster.close()
        client.close()
