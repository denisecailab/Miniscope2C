---
title: Project Miniscope_2s notes
---

# Project Miniscope_2s notes

## Summary

The miniscope_2s desgin, with two filter set configurations: GFP+mCherry and GFP+tdTomato, has been validated *in vivo* with following animals:

* WT - GCamp_tdTomato_conjugated.
  This is the most promising animal setup.
  Confocal imaging has shown a near 100% overlap between the GCamp channel and the tdTomato channel.
  *in vivo* imaging show good signals in both channel and a fair amount of overlap.
  However, during *in vivo* imaging, there are still cells showing GCamp signal but not tdTomato signal, suggesting an imperfect alignment of focal plane.
  See [the data section](#data-gcamp_tdtomato_conjugated) for detail.

* WT - fosTTA-mCherry and syn-GCamp
  This animal setup has a cocktail of 3 virus.
  Initially during baseplatting, there were no signal in the red channel.
  We then decided to take the animals off-dox for a few month.
  After that we did some *in vivo* imaging.
  Both channels had clear signals.
  However the cell shape in red channels look weird, and the overlap was not good.
  See [the data section](#data-fostta-mcherry-and-syn-gcamp) for detail.

* WT - mCherry and syn-GCamp
  This is a 2 cocktail setup.
  During baseplating, we see clear signal in the red channel.
  However the cell shape in the red channels look weird.
  At the same time, we see no fluctuation in the green channel at all.
  We think that the mCherry virus might killed all the cells.
  We decided to not baseplate these animals.

* GAD-Cre - flex-RCamp and CamKII-GCamp
  The primary goal for this setup is to validate whether the scope can image two dynamic signals from two different population of neurons.
  However, we could see neither red nor green signals during baseplating.
  Slice imaging confirm there are good expression of GCamp under FOV.

## Data: GCamp_tdTomato_conjugated

All mice were wild type and injected with `AAV1-hSyn-GCaMP6f-P2A-nls-dTomato` virus at stock concentration.

### m00

#### *in vivo* Imaging

<embed type="text/html" src="../output/tdTomato/overlap/m00.html" width="1200" height="900"></embed>

#### Confocal

![](../data/2color_pilot_tdTomato/m00/confocal/sld3-x1-y0-img0/overlay.svg){width="900px"}

### cv03

#### *in vivo* Imaging

<embed type="text/html" src="../output/tdTomato/overlap/cv03.html" width="1200" height="900"></embed>

## Data: fosTTA-mCherry and syn-GCamp

All mice were wild type and injected with a cocktail of 3 virus: `AAV9-FosTTA`, `AAV9-TRE-mCherry`, `AAV1-syn-GCaMP6f`.
Initially there were no signal in the red channel.
We then take the animals off-dox for a few month and we have the following *in vivo* imaging.

### m03

#### *in vivo* Imaging

<embed type="text/html" src="../output/mCherry/overlap/m03.html" width="1200" height="900"></embed>

### m04

#### *in vivo* Imaging

<embed type="text/html" src="../output/mCherry/overlap/m04.html" width="1200" height="900"></embed>

## Data: flex-RCamp and CamKII-GCamp

Two GAD-Cre mice from collaborator lab are injected with `AAV1-CAG-FLEX-NEX-jRCaMP1a-WPRE-SV40` at $1.9 \times 10^{13} GC/mL$ for $250 nL/injection$ and `AAV1.CamKII.GCaMP6f.WPRE.SV40` at $1 \times 10^{13} vg/mL$ for $250 nL/injection$.

### gc01

#### Baseplating Imaging

Couldn't see any signals.
Unprocessed data can be found under data folder.

### gc02

#### Slice Imaging

![](../data/2color_pilot_tdTomato/gc02/thunder/Project_Gad-cre_3_10X_crop.png){width="900px"}
