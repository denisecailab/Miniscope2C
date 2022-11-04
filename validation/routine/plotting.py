import numpy as np
import colorcet as cc
from matplotlib import cm


def pcolor_ovly(fm_red, fm_green, brt_offset=0):
    fm_red_pcolor = np.clip(
        cm.ScalarMappable(cmap=cc.m_linear_ternary_red_0_50_c52).to_rgba(
            np.array(fm_red)
        )
        + brt_offset,
        0,
        1,
    )
    fm_green_pcolor = np.clip(
        cm.ScalarMappable(cmap=cc.m_linear_ternary_green_0_46_c42).to_rgba(
            fm_green.values
        )
        + brt_offset,
        0,
        1,
    )
    fm_ovly = np.clip(fm_red_pcolor + fm_green_pcolor, 0, 1)
    return fm_red_pcolor, fm_green_pcolor, fm_ovly
