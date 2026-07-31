# Topo2fault

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

## Code

This repository archives the inversion applied to 15 Himalayan rivers (Beas to Dudh Koshi).

```
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

Python dependencies are listed in `requirements.txt`. Snapping and channel-elevation sampling additionally require the Copernicus GLO-30 DEM and topotoolbox, whose local paths are set at the top of `snap_rivers.py` and `run_inversion.py`.
