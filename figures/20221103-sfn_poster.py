#%% imports and definition
import os

from svgutils.compose import SVG, Figure, Panel, Text

PARAMT_TEXT = {"size": 11, "weight": "bold"}
OUT_PATH = "./20221103-sfn_poster"
os.makedirs(OUT_PATH, exist_ok=True)


def make_panel(label, im_path, im_scale=1, sh=None):
    im = SVG(im_path, fix_mpl=True).scale(im_scale)
    lab = Text(label, **PARAMT_TEXT)
    tsize = PARAMT_TEXT["size"]
    if sh is None:
        x_sh, y_sh = 0.6 * tsize, 1 * tsize
    else:
        x_sh, y_sh = sh
    pan = Panel(im.move(x=x_sh, y=y_sh), lab.move(x=0, y=tsize))
    pan.height = im.height * im_scale + y_sh
    pan.width = im.width * im_scale + x_sh
    return pan


#%% make fig1
w_gap = 15
h_gap = 5
sh_left = (0, 0)
panA = make_panel("A", "../drawings/miniscope_2s_lightpath_filters.svg", sh=sh_left)
panB = make_panel("B", "../drawings/slide.svg", sh=sh_left, im_scale=0.4)
panC = make_panel("C", "../validation/output/overlap/bench-res.svg", sh=sh_left)
panD = make_panel("D", "../validation/output/overlap/bench-res-grin.svg", sh=sh_left)
fig_h = panA.height + panB.height + panC.height + panD.height + 3 * h_gap
fig_w = max(panA.width, panB.width, panC.width, panD.width)
fig = Figure(
    fig_w,
    fig_h,
    panA.move(x=(fig_w - panA.width) / 2, y=0),
    panB.move(x=(fig_w - panB.width) / 2, y=panA.height + h_gap),
    panC.move(x=(fig_w - panC.width) / 2, y=panA.height + panB.height + 2 * h_gap),
    panD.move(
        x=(fig_w - panD.width) / 2,
        y=panA.height + panB.height + panC.height + 3 * h_gap,
    ),
)
fig.save(os.path.join(OUT_PATH, "fig1.svg"))
