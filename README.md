# Complex DEM simulation using Taichi DEM
DEM simulation in Taichi HPC Framework incorporating partcle orientation, rotation and bonding.

![](Demos/cube_911_particles_impact/cube_911_particles_impact.gif)

>Visualized using BIMBase with secondary development. BIMBase is a graphical platform aimed for BIM by Glory PKPM. https://app.pkpm.cn/pbims

## Authors
Denver Pilphis (Complex DEM mechanism implementation)

MuGdxy (GPU HPC optimization)

## Introducion
A bonded agglomerate with cubed shape hitting on a flat surface is performed as the initial demo. The bonds within the agglomerate will fail while the agglomerate is hitting the surface. Then the agglomerate will break into fragments, flying to the surrounding space.

To run the demo:

```bash
$ python dem.py
```

## Features
Compared to the initial Taichi DEM commit, this instance has added the following features:

1. 2D DEM to 3D DEM;
2. Particle orientation and rotation are fully considered and implemented, in which the possibility for modeling nonspherical particles is reserved;
3. Wall (geometry in DEM) element is implemented, particle-wall contact is solved;
4. Complex DEM contact model is implemented, including a bond model (Edinburgh Bond Particle Model, EBPM) and a granular contact model (Hertz-Mindlin Model);
5. As a bond model is implemented, nonspherical particles can be simulated with bonded agglomerates;
6. As a bond model is implemented, particle breakage can be simulated.

## Future work
The development will persist in which more interesting demos will come out!
