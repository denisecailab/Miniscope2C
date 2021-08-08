#%% imports and definitions
import os

import holoviews as hv
from dask.diagnostics import ProgressBar

from routine.utilities import open_minian, save_minian

pbar = ProgressBar(minimum=2)
pbar.register()
hv.notebook_extension("bokeh")

IN_DSPATH = "~/var/miniscope_2s/tdTomato/minian_ds"
OUT_DSPATH = "~/var/miniscope_2s/tdTomato/minian_ds"
IN_DSPATH = os.path.normpath(os.path.expanduser(IN_DSPATH))
OUT_DSPATH = os.path.normpath(os.path.expanduser(OUT_DSPATH))

#%% demixing
ds = open_minian(IN_DSPATH)
va_side_mc = ds["va_side_mc"]
med_side = va_side_mc.median("frame").compute()
va_side_dm = (va_side_mc - med_side).clip(0)
va_side_dm = save_minian(va_side_dm.rename("va_side_dm"), OUT_DSPATH, overwrite=True)
