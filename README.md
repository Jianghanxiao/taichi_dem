# Engineering quantitative DEM simulation using Taichi DEM
A complete implementation of DEM in Taichi Lang from engineering perspective.

![](Demos/carom/carom.gif)
![](Demos/cube_911_particles_impact/cube_911_particles_impact.gif)
![](Demos/cube_18112_particles_impact/cube_18112_particles_impact.gif)

>Visualized using BIMBase with secondary development. BIMBase is a graphical platform aimed for BIM by Glory PKPM. https://app.pkpm.cn/pbims

## Authors
Denver Pilphis (Di Peng) - DEM theory and implementation

MuGdxy (Xinyu Lu) - Performance optimization

## Introducion
This instance provides a complete implementation of discrete element method (DEM) for simulation.
Complex DEM mechanics are considered and the result is engineering quantitative.
The efficiency of computation is guaranteed by Taichi, along with proper arrangements of data and algorithms.

To run the demo:

```bash
$ python dem.py
```

You may need to modify parameters before you run. See the comments in `dem.py`.

## Features
Compared with initial version, this instance has added the following features:

1. 2D DEM to 3D DEM;
2. Particle orientation and rotation are fully considered and implemented, in which the possibility for modeling nonspherical particles is reserved;
3. Wall (geometry in DEM) element is implemented, particle-wall contact is solved;
4. Complex DEM contact model is implemented, including a bond model (Edinburgh Bonded Particle Model, EBPM) and a granular contact model (Hertz-Mindlin Contact Model);
5. As a bond model is implemented, nonspherical particles can be simulated with bonded agglomerates;
6. As a bond model is implemented, particle breakage can be simulated.

## Demos
### Carom billiards
This demo performs the first stage of carom billiards. The white ball goes towards other balls and collision
occurs soon. Then the balls scatter. Although there is energy loss, all the balls will never stop as they
enter the state of pure rotation and no rolling resistance is available to dissipate the rotational kinematic
energy. This could be a good example of validating Hertz-Mindlin model.

![](Demos/carom/carom.gif)

### Cube with 911 particles impact on a flat surface
This demo performs a bonded agglomerate with cubed shape hitting on a flat surface.
The bonds within the agglomerate will fail while the agglomerate is hitting the surface.
Then the agglomerate will break into fragments, flying to the surrounding space.
This could be a good example of validating EBPM.

![](Demos/cube_911_particles_impact/cube_911_particles_impact.gif)

### Cube with 18112 particles impact on a flat surface
This demo is similar to the one above, with the only difference of particle number.
This could be a good example of benchmark on large system simulation.

![](Demos/cube_18112_particles_impact/cube_18112_particles_impact.gif)

## Acknowledgements
Dr. Xizhong Chen from Department of Chemical and Biological Engineering,
The University of Sheffield is acknowledged.
