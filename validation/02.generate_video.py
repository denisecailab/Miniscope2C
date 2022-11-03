#%% imports and definition
import os
import shutil

import colorcet as cc
import cv2
import ffmpeg
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm
from minian.cnmf import compute_AtC
from minian.utilities import load_videos, open_minian, save_minian

from routine.utilities import norm_xr

hv.notebook_extension("bokeh")

IN_SIDE_INTPATH = "~/var/miniscope_2s/side_int"
IN_SIDE_INTPATH = os.path.abspath(os.path.expanduser(IN_SIDE_INTPATH))
IN_TOP_INTPATH = "~/var/miniscope_2s/top_int"
IN_TOP_INTPATH = os.path.abspath(os.path.expanduser(IN_TOP_INTPATH))
IN_DSPATH = "./data/2color_pilot_tdTomato/cv03/2022_06_21/16_01_56"
INTPATH = "~/var/miniscope_2s/minian_int"
INTPATH = os.path.abspath(os.path.expanduser(INTPATH))
SUBSET = {
    "frame": slice(0, 2999),
    "height": slice(100 - 25, 349 + 25),
    "width": slice(240 - 25, 479 + 25),
}
SUBSET_BEHAV = {
    "frame": slice(0, 2999),
    "height": slice(240, 370),
    "width": slice(90, 530),
}
ROT_BEHAV = -5.5
ANNT_COL = (255, 255, 255)
CLIP = (5, 25)
BRT_OFFSET = 0
OUTPATH = "./output/tdTomato/video/cv03"
TITLES = {
    "top": "tdTomato",
    "side": "GCamp",
    "ovly": "Overlay",
    "top_max": "tdTomato Max Projection",
    "ovly_max": "Overlay",
}


def annt(fm, text):
    return cv2.putText(
        (fm * 255).astype(np.uint8),
        text,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        ANNT_COL,
        2,
    )


def make_combined_frame(gfm, gfm_ps, rfm, rfm_ps, behav=None):
    gcm = cm.ScalarMappable(cmap=cc.m_linear_ternary_green_0_46_c42)
    rcm = cm.ScalarMappable(cmap=cc.m_linear_ternary_red_0_50_c52)
    gfm, gfm_ps = (
        gcm.to_rgba(gfm, norm=False)[:, :, :3],
        gcm.to_rgba(gfm_ps, norm=False)[:, :, :3],
    )
    rfm, rfm_ps = (
        rcm.to_rgba(rfm, norm=False)[:, :, :3],
        rcm.to_rgba(rfm_ps, norm=False)[:, :, :3],
    )
    ovlp, ovlp_ps = np.clip(gfm + rfm * 0.75, 0, 1), np.clip(gfm_ps + rfm_ps, 0, 1)
    gfm, gfm_ps, rfm, rfm_ps, ovlp, ovlp_ps = (
        annt(gfm, "GCaMP"),
        annt(gfm_ps, "GCaMP ps"),
        annt(rfm, "tdTomato"),
        annt(rfm_ps, "tdTomato ps"),
        annt(ovlp, "Overlap"),
        annt(ovlp_ps, "Overlap ps"),
    )
    # out_im = np.concatenate(
    #     [
    #         np.concatenate([gfm, rfm, ovlp], axis=1),
    #         np.concatenate([gfm_ps, rfm_ps, ovlp_ps], axis=1),
    #     ],
    #     axis=0,
    # )
    out_im = np.concatenate([gfm, rfm, ovlp], axis=1)
    if behav is not None:
        w = out_im.shape[1]
        behav = (
            cm.ScalarMappable(cmap=cc.m_gray)
            .to_rgba(behav, norm=False, bytes=True)[:, :, :3]
            .astype(np.uint8)
        )
        behav = cv2.resize(behav, dsize=(w, int(w / behav.shape[1] * behav.shape[0])))
        out_im = np.concatenate([behav, out_im], axis=0)
    return out_im.astype(np.uint8)


def write_video(
    arr: xr.DataArray,
    vname: str = None,
    options={"crf": "18", "preset": "ultrafast"},
) -> str:
    w, h = arr.sizes["width"], arr.sizes["height"]
    process = (
        ffmpeg.input(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="{}x{}".format(w, h), r=60
        )
        .filter("pad", int(np.ceil(w / 2) * 2), int(np.ceil(h / 2) * 2))
        # .filter("fps", fps=60, round="up")
        .output(vname, pix_fmt="yuv420p", vcodec="libx264", r=60, **options)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for blk in arr.data.blocks:
        process.stdin.write(np.array(blk).tobytes())
    process.stdin.close()
    process.wait()
    return vname


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


#%% load data
shutil.rmtree(INTPATH, ignore_errors=True)
side_ds = open_minian(os.path.join(IN_DSPATH, "miniscope_side", "minian_ds")).sel(
    **SUBSET
)
side_int_ds = open_minian(IN_SIDE_INTPATH, return_dict=True)
top_ds = open_minian(os.path.join(IN_DSPATH, "miniscope_top", "minian_ds")).sel(
    **SUBSET
)
top_int_ds = open_minian(IN_TOP_INTPATH, return_dict=True)
behav_vid = load_videos(os.path.join(IN_DSPATH, "behavcam"), pattern=r"[0-9]+\.avi$")
behav_vid = xr.apply_ufunc(
    rotate_image,
    behav_vid,
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    vectorize=True,
    dask="parallelized",
    kwargs={"angle": ROT_BEHAV},
)
behav_vid = save_minian(
    behav_vid.sel(**SUBSET_BEHAV).rename("behav_vid"), INTPATH, overwrite=True
)
#%% compute and normalization
gRaw = side_int_ds["Y_fm_chk"].sel(**SUBSET)
rRaw = top_int_ds["Y_fm_chk"].sel(**SUBSET)
gAC = save_minian(
    compute_AtC(side_ds["A"], side_ds["C"]).rename("gAC"), INTPATH, overwrite=True
)
rAC = save_minian(
    compute_AtC(top_ds["A"], top_ds["C"]).rename("rAC"), INTPATH, overwrite=True
)
gRaw, rRaw, gAC, rAC = (
    norm_xr(gRaw, q=0.999),
    norm_xr(rRaw, q=0.98),
    norm_xr(gAC, q=0.999),
    norm_xr(rAC, q=0.9),
)
behav_vid = norm_xr(behav_vid)

#%% generate video
test_fm = make_combined_frame(
    gRaw.isel(frame=0).values,
    gAC.isel(frame=0).values,
    rRaw.isel(frame=0).values,
    rAC.isel(frame=0).values,
    behav_vid.isel(frame=0).values,
)
out = xr.apply_ufunc(
    make_combined_frame,
    gAC,
    gAC,
    rRaw,
    rAC,
    behav_vid.rename({"height": "height_b", "width": "width_b"}),
    input_core_dims=[["height", "width"]] * 4 + [["height_b", "width_b"]],
    output_core_dims=[["height_new", "width_new", "rgb"]],
    vectorize=True,
    output_sizes={
        "height_new": test_fm.shape[0],
        "width_new": test_fm.shape[1],
        "rgb": test_fm.shape[2],
    },
    dask="parallelized",
)
out = out.rename({"height_new": "height", "width_new": "width"})
os.makedirs(OUTPATH, exist_ok=True)
write_video(out, os.path.join(OUTPATH, "combined.mp4"))
