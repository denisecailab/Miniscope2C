#%% imports and definitions
import os

import holoviews as hv
from dask.diagnostics import ProgressBar

from routine.alignment import apply_transform
from routine.motion_correction import apply_transform, estimate_motion
from routine.utilities import open_minian, save_minian

pbar = ProgressBar(minimum=2)
pbar.register()
hv.notebook_extension("bokeh")

IN_DSPATH = "~/var/miniscope_2s/tdTomato/minian_ds"
OUT_DSPATH = "~/var/miniscope_2s/tdTomato/minian_ds"
param_mc = {"alt_error": None}
IN_DSPATH = os.path.normpath(os.path.expanduser(IN_DSPATH))
OUT_DSPATH = os.path.normpath(os.path.expanduser(OUT_DSPATH))

#%% motion correction
minian_ds = open_minian(IN_DSPATH)
va_top = minian_ds["va_top"]
va_side = minian_ds["va_side"]
mo = estimate_motion(va_top, **param_mc)
mo = save_minian(mo.rename("motion"), OUT_DSPATH, overwrite=True)
va_top_mc = apply_transform(va_top, mo)
va_side_mc = apply_transform(va_side, mo)
va_top_mc = save_minian(va_top_mc.rename("va_top_mc"), OUT_DSPATH, overwrite=True)
va_side_mc = save_minian(va_side_mc.rename("va_side_mc"), OUT_DSPATH, overwrite=True)

# %% plot motion correction result
opts_im = {"cmap": "viridis"}
(
    hv.Image(va_top.max("frame").compute().rename("top before")).opts(**opts_im)
    + hv.Image(va_top_mc.max("frame").compute().rename("top after")).opts(**opts_im)
    + hv.Image(va_side.max("frame").compute().rename("side before")).opts(**opts_im)
    + hv.Image(va_side_mc.max("frame").compute().rename("side after")).opts(**opts_im)
).cols(2)
