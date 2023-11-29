# %% imports and definition
import os

from svgutils.compose import SVG, Figure, Image, Panel, Text

PARAMT_TEXT = {"size": 11, "weight": "bold"}
OUT_PATH = "./20221228-2s_paper"
os.makedirs(OUT_PATH, exist_ok=True)


def make_panel(
    label, im_path, im_scale=1, sh=None, is_svg=True, width=None, height=None
):
    if is_svg:
        im = SVG(im_path, fix_mpl=True).scale(im_scale)
    else:
        im = Image(fname=im_path, width=width, height=height).scale(im_scale)
    lab = Text(label, **PARAMT_TEXT)
    tsize = PARAMT_TEXT["size"]
    if sh is None:
        x_sh, y_sh = 0.6 * tsize, 1 * tsize
    else:
        x_sh, y_sh = sh
    pan = Panel(im.move(x=x_sh, y=y_sh), lab.move(x=0, y=tsize))
    if width is not None and height is not None:
        pan.height = height * im_scale + y_sh
        pan.width = width * im_scale + x_sh
    else:
        pan.height = im.height * im_scale + y_sh
        pan.width = im.width * im_scale + x_sh
    return pan


# %% make design figure
w_gap = 25
h_gap = 10
sh_left = (15, 0)
panA = make_panel(
    "A", "../drawings/miniscope_2s_dimensions.svg", sh=sh_left, im_scale=0.3
)
panB = make_panel(
    "B",
    "../drawings/miniscope_2s_photo.png",
    sh=sh_left,
    is_svg=False,
    width=1767,
    height=2469,
    im_scale=0.094,
)
panC = make_panel(
    "C",
    "../drawings/miniscope_2s_section_labeled/miniscope_2s_labels.svg",
    sh=sh_left,
    im_scale=0.55,
)
row1_height = max(panA.height, panB.height)
row1_width = panA.width + panB.width + w_gap
fig_h = row1_height + panC.height + h_gap
fig_w = max(row1_width, panC.width)
fig = Figure(
    fig_w,
    fig_h,
    panA,
    panB.move(x=panA.width + w_gap, y=0),
    panC.move(x=(fig_w - panC.width) / 2, y=row1_height + h_gap),
)
fig.save(os.path.join(OUT_PATH, "miniscope_2s-design.svg"))

# %% make slide figure
w_gap = 15
h_gap = 5
sh_left = (0, 0)
panA = make_panel("A", "../drawings/slide.svg", sh=sh_left, im_scale=0.45)
panB = make_panel("B", "../validation/output/overlap/bench-res.svg", sh=sh_left)
panC = make_panel("C", "../validation/output/overlap/bench-res-grin.svg", sh=sh_left)
fig_h = panA.height + panB.height + panC.height + 2 * h_gap
fig_w = max(panA.width, panB.width, panC.width)
fig = Figure(
    fig_w,
    fig_h,
    panA.move(x=(fig_w - panA.width) / 2, y=0),
    panB.move(x=(fig_w - panB.width) / 2, y=panA.height + h_gap),
    panC.move(x=(fig_w - panC.width) / 2, y=panA.height + panB.height + 2 * h_gap),
)
fig.save(os.path.join(OUT_PATH, "miniscope_2s-slide.svg"))
