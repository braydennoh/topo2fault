# Topo2fault

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21727937.svg)](https://doi.org/10.5281/zenodo.21727937)

## What is Topo2fault?

Topo2fault infers fault geometry at depth from river topography. Deviations of a river profile from its characteristic form map to dip changes along the fault, so topography constrains the relative, non-planar geometry.

River profiles are typically modeled with the stream-power equation ([Whipple & Tucker, 1999](https://doi.org/10.1029/1999JB900120)), assuming a spatially uniform rock-uplift rate $U$,

```math
\frac{\partial z}{\partial t}
=
U-KA^m\left|\frac{\partial z}{\partial x}\right|^n.
```

Topo2fault instead relates $U$ to the basal fault. In fault-bend folding ([Suppe, 1983](https://doi.org/10.2475/ajs.283.7.684)), rock moving over a fault segment with dip $\theta$ at slip rate $\dot{s}$ is uplifted at

```math
U(x)=\dot{s}\sin\theta(x).
```
These equations link fault geometry to topography, making a constrained inversion possible.

An interactive version of 2.2 Convergence partition is available [here](https://braydennohprojects.github.io/CoInterFaultFold/Version5/partition_sim/) (click to launch).

## Constraints

Topography alone leaves two degeneracies. At steady state, the stream-power equation gives

```math
\left|\frac{\partial z}{\partial x}\right|
=
\left(\frac{U}{K}\right)^{1/n}A^{-m/n},
```

so the profile constrains only the ratio $U/K$.

Since $U=\dot{s}\sin\theta$, only their product is identified. Any rescaling

```math
\left(\dot{s},\sin\theta\right)
\rightarrow
\left(c\dot{s},\frac{\sin\theta}{c}\right)
```

leaves $U$, and hence the topography, unchanged.

Priors on $\dot{s}$ and $\theta$ are informative. In the Himalaya, the Main Himalayan Thrust accommodates convergence at $\dot{s}\approx20\ \mathrm{mm\,yr^{-1}}$ ([Bilham et al., 2001](https://doi.org/10.1126/science.1062584)) and dips at $\theta\sim10^\circ$ ([Nábělek et al., 2009](https://doi.org/10.1126/science.1167719)). Given these priors, departures from characteristic stream-power concavity become diagnostic because a topographic bulge requires locally higher $U$, implying a steeper fault segment.

<p align="center">
  <img src="assets/comparison_morph.gif" width="100%" alt="Prior-to-posterior morph: ensembles of river profiles (h), rock uplift (v_z/s-dot), and fault geometry (z) tightening from scattered prior draws onto the true fault in three synthetic cases">
</p>

## Earthquake cycle

The uplift $U=\dot{s}\sin\theta$ above is the long-term, cycle-averaged field. Over a balanced cycle it splits as

```math
v_\mathrm{interseismic}=v_\mathrm{long\text{-}term}-v_\mathrm{coseismic},
```

so geometry recovered from topography also predicts the geodetic fields. At a fault bend, flexural slip folds the hanging wall across an axial surface; material crossing it changes velocity discontinuously, and that jump deforms the elastic half-space exactly as fault slip does, so each axial surface is carried as a dislocation ([Souter & Hager, 1997](https://doi.org/10.1029/97JB00209)) alongside the fault segments ([Freund & Barnett, 1976](https://doi.org/10.1785/BSSA0660030667)). Discretizing one listric fault more finely spreads the folds into a continuous distribution of slip density $\dot{s}\kappa$, and all three fields converge to the smooth non-planar limit. Section 2.3 of the paper; derived from scratch in `earthquake_cycle.ipynb`.

## Code

This repository archives the inversion applied to 15 Himalayan rivers (Beas to Dudh Koshi).

```
synthetic_test.ipynb     tutorial: build a synthetic profile, then invert it
earthquake_cycle.ipynb   tutorial: the earthquake-cycle dislocation model from scratch
run_all.py               driver (snap + 15 inversions)
run_inversion.py         MCMC inversion for one river (--seg N --config F3)
thrust_fault_model.py    forward model (fault-bend folding + stream power)
snap_rivers.py           snaps picked polylines to the DEM flow network
data/
  normals.csv            MFT-normal transects 1-15
  rivers/river_NN.csv          snapped channel profiles (inversion input)
  rivers/river_NN_orig.csv     picked polylines (pre-snap)
```

`python run_inversion.py --seg 7 --config F3` inverts one river; `python run_all.py` runs all 15 (about 7.5 hr). Each run ends in `results/river_NN/mcmc_results.npz` with the full posterior.

`synthetic_test.ipynb` is a self-contained worked example that generates a river profile from a prescribed flat-ramp-flat fault and recovers the geometry from it, in about 90 seconds and with no external data.

`earthquake_cycle.ipynb` derives the uplift forward model of the section above from scratch — free-surface edge dislocations, axial surfaces as dislocations, and the balanced cycle — reproducing the figure there. Only `numpy` and `matplotlib`; runtime a few seconds.

Python dependencies are listed in `requirements.txt`. Snapping and channel-elevation sampling additionally require the Copernicus GLO-30 DEM and topotoolbox, whose local paths are set at the top of `snap_rivers.py` and `run_inversion.py`.
