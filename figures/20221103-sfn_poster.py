#%% imports and definition
import os

from svgutils.compose import SVG, Figure, Panel, Text, Grid

PARAMT_TEXT = {"size": 11, "weight": "bold"}
OUT_PATH = "./20221103-sfn_poster"
os.makedirs(OUT_PATH, exist_ok=True)


def make_panel(label, im_path, im_scale=1):
    im = SVG(im_path, fix_mpl=True).scale(im_scale)
    lab = Text(label, **PARAMT_TEXT)
    tsize = PARAMT_TEXT["size"]
    x_sh, y_sh = 0.6 * tsize, 1 * tsize
    pan = Panel(im.move(x=x_sh, y=y_sh), lab.move(x=0, y=y_sh))
    pan.height = im.height * im_scale + y_sh
    pan.width = im.width * im_scale + x_sh
    return pan


#%% make fig1
w_gap = 10
h_gap = 10
panA = make_panel("A", "../drawings/miniscope_2s_lightpath.svg", 1.2)
panB = make_panel("B", "../validation/output/overlap/bench-res.svg")
panC = make_panel("C", "../validation/output/overlap/bench-res-grin.svg")
fig_h = max(panA.height, panB.height + panC.height + h_gap)
fig_w = panA.width + w_gap + max(panB.width, panC.width)
fig = Figure(
    fig_w,
    fig_h,
    panA.move(x=0, y=(fig_h - panA.height) / 2),
    panB.move(
        x=panA.width + w_gap + (fig_w - panA.width - panB.width) / 2,
        y=(fig_h - panB.height - panC.height) / 3,
    ),
    panC.move(
        x=panA.width + w_gap + (fig_w - panA.width - panC.width) / 2,
        y=(fig_h - panB.height - panC.height) / 3 * 2 + panB.height + h_gap,
    ),
)
fig.save(os.path.join(OUT_PATH, "fig1.svg"))
