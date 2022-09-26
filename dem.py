# Complex DEM simulation using Taichi DEM
# 
# Authors:
# Denver Pilphis (Complex DEM mechanism implementation)
# MuGouDanx (GPU HPC optimization)
# 
# Introducion
# This example performs a bonded agglomerate with cubed shape hitting on a flat surface.
# The bonds within the agglomerate will fail while the agglomerate is hitting the surface.
# Then the agglomerate will break into fragments, flying to the surrounding space.
#
# Features
# Compared with initial version, this example has added the following features:
# 1. 2D DEM to 3D DEM;
# 2. Particle orientation and rotation are fully considered and implemented, in which the possibility
#    for modeling nonspherical particles is reserved;
# 3. Wall (geometry in DEM) element is implemented, particle-wall contact is solved;
# 4. Complex DEM contact model is implemented, including a bond model (Edinburgh Bond Particle Model)
#    and a granular contact model (Hertz-Mindlin Model).
# 
# TODO List
# 1. Matrix (vector) multiplying calculations were not checked, especially the 12x12 matrix multiplying 12x1 vector in EBPM,
#    as well as the combination of three Vector3ds into a 12x1 vector;
# 2. Particle / wall / particle-particle contact / particle-wall contact lists are NOT initialized properly
#    (must ensure some flags are zeros initially);
# 3. HPC optimization has NOT been implemented for particle-wall contacts.

import taichi as ti
import math
import os

# Taichi base properties
ti.init(arch=ti.gpu, default_ip = ti.i32, default_fp = ti.f64);
Vector3d = ti.math.vec3;
Vector4d = ti.math.vec4;
DEMMatrix = ti.math.mat3;
EBPMStiffnessMatrix = ti.types.matrix(12, 12, ti.f64);
EBPMForceDisplacementVector = ti.types.matrix(12, 1, ti.f64);

SAVE_FRAMES = True;

window_size = 1024  # Number of pixels of the window

# DEM simulation deck properties
nDims : ti.i32 = 3; # Number of dimensions in DEM; as 2D DEM is deprecated now, only 3D DEM is considered - Di Peng

n : ti.i32 = 0; # n = 8192; # Number of particles; will be loaded at the first stage

# Di Peng: density should be linked to a material by which a particle is assigned
# so we deprecate density as global parameter
#density = 100.0
# Di Peng: stiffness should be linked to a material by which a particle is assigned
# so we deprecate stiffness as global parameter
#stiffness = 8e3
# Di Peng: coefficient of restitution should be linked to a material by which a particle is assigned
# so we deprecate coefficient of restitution as global parameter
# restitution_coef = 0.001

# Gravity, a global parameter
# Di Peng: in this example, we assign no gravity
gravity : Vector3d = Vector3d(0.0, 0.0, 0.0); # -9.81;
# Time step, a global parameter
dt : ti.f64 = 1e-7;  # Larger dt might lead to unstable results.
target_time : ti.f64 = 0.1;
# No. of steps for run, a global parameter
nsteps : ti.i32 = int(target_time / dt);
saving_interval_time : ti.f64 = 0.001;
saving_interval_steps : ti.i32 = int(saving_interval_time / dt);

# Particle in DEM
# Di Peng: keep spherical shape first, added particle attributes to make the particle kinematically complete
@ti.dataclass
class Grain:
    # Material attributes
    # At this stage, the material attributes are written here
    density: ti.f64;  # Density, double
    mass: ti.f64;  # Mass, double
    radius: ti.f64;  # Radius, double
    elasticModulus: ti.f64;  # Elastic modulus, double
    poissonRatio: ti.f64; # Poisson's ratio, double
    # Translational attributes
    position: Vector3d;  # Position in GLOBAL coordinates, Vector3d
    velocity: Vector3d;  # Velocity in GLOBAL coordinates, Vector3d
    acceleration: Vector3d;  # Acceleration in GLOBAL coordinates, Vector3d
    force: Vector3d;  # Force in GLOBAL coordinates, Vector3d
    # Rotational attributes
    quaternion: Vector4d;  # Quaternion in GLOBAL coordinates, Vector4d, order in [w, x, y, z]
    omega: Vector3d;  # Angular velocity in GLOBAL coordinates, Vector3d
    omega_dot: Vector3d;  # Angular acceleration in GLOBAL coordinates, Vector3d
    inertia: DEMMatrix; # Moment of inertia tensor in LOCAL coordinates, 3 * 3 matrix with double
    moment: Vector3d; # Total moment/torque in GLOBAL coordinates, Vector3d

# Initialize grain field
gf = Grain.field();

# Wall in DEM
# Only premitive wall is implemented
@ti.dataclass
class Wall:
    # Wall equation: Ax + By + Cz - D = 0
    # Reference: Peng and Hanley (2019) Contact detection between convex polyhedra and superquadrics in discrete element codes.
    # https://doi.org/10.1016/j.powtec.2019.07.082
    # Eq. (8)
    normal: Vector3d; # Outer normal vector of the wall, [A, B, C]
    distance: ti.f64; # Distance between origin and the wall, D
    # Material properties
    density: ti.f64; # Density of the wall
    elasticModulus: ti.f64; # Elastic modulus of the wall
    poissonRatio: ti.f64; # Poisson's ratio of the wall

wf = Wall.field();

# Contact in DEM
# In this example, the Edinburgh Bond Particle Model (EBPM), along with Hertz-Mindlin model, is implemented
# Reference: Brown et al. (2014) A bond model for DEM simulation of cementitious materials and deformable structures.
# https://doi.org/10.1007/s10035-014-0494-4
# Reference: Mindlin and Deresiewicz (1953) Elastic spheres in contact under varying oblique forces.
# https://doi.org/10.1115/1.4010702
@ti.dataclass
class Contact:
    # Contact status
    isActive : ti.i8; # Contact exists: 1 - exist; 0 - not exist
    isBonded : ti.i8; # Contact is bonded: 1 - bonded, use EBPM; 0 - unbonded, use Hertz-Mindlin
    # Common Parameters
    rotationMatrix : DEMMatrix; # Rotation matrix from global to local system of the contact
    # EBPM parameters
    radius_ratio : ti.f64; # Section radius ratio
    # radius : ti.f64; # Section radius: r = rratio * min(r1, r2), temporarily calculated in evaluation
    length : ti.f64; # Length of the bond
    elasticModulus : ti.f64; # Elastic modulus of the bond
    poissonRatio : ti.f64; # Poission's ratio of the bond
    compressiveStrength: ti.f64; # Compressive strength of the bond
    tensileStrength: ti.f64; # Tensile strength of the bond
    shearStrength: ti.f64; # Shear strength of the bond
    force_a : Vector3d; # Contact force at side a in LOCAL coordinate
    moment_a : Vector3d; # Contact moment/torque at side a in LOCAL coordinate
    force_b : Vector3d; # Contact force at side b in LOCAL coordinate
    moment_b : Vector3d; # Contact moment/torque at side b in LOCAL coordinate
    # Hertz-Mindlin parameters
    position : Vector3d; # Position of contact point in GLOBAL coordinate
    # normalStiffness: ti.f64; # Normal stiffness, middleware parameter only
    # shearStiffness: ti.f64; # Shear stiffness, middleware parameter only
    coefficientFriction: ti.f64; # Friction coefficient, double
    coefficientRestitution: ti.f64; # Coefficient of resitution, double
    coefficientRollingResistance: ti.f64; # Coefficient of rolling resistance, double
    shear_displacement: Vector3d; # Shear displacement stored in the contact

# Initialize contact field for resolving contacts
cf = Contact.field();
wcf = Contact.field();

# GPU grid setup
# TODO: lmm
grid_n = 128
grid_size = 1.0 / grid_n  # Simulation domain of size [0, 1]
print(f"Grid size: {grid_n}x{grid_n}")

grain_r_min = 0.002
grain_r_max = 0.003

assert grain_r_max * 2 < grid_size

# Add a math function: quaternion to rotation matrix
# References:
# https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
# Lines 511-534, https://github.com/CFDEMproject/LIGGGHTS-PUBLIC/blob/master/src/math_extra_liggghts_nonspherical.h
@ti.kernel
def quat2RotMatrix(quat : Vector4d):
    # w i j k
    # 0 1 2 3
    w2 : ti.f64 = quat[0] * quat[0];
    i2 : ti.f64 = quat[1] * quat[1];
    j2 : ti.f64 = quat[2] * quat[2];
    k2 : ti.f64 = quat[3] * quat[3];
    
    twoij : ti.f64 = 2.0 * quat[1] * quat[2];
    twoik : ti.f64 = 2.0 * quat[1] * quat[3];
    twojk : ti.f64 = 2.0 * quat[2] * quat[3];
    twoiw : ti.f64 = 2.0 * quat[1] * quat[0];
    twojw : ti.f64 = 2.0 * quat[2] * quat[0];
    twokw : ti.f64 = 2.0 * quat[3] * quat[0];

    result : DEMMatrix = ti.Matrix.diag(3, 3, 0.0);
    result[0, 0] = w2 + i2 - j2 - k2;
    result[0, 1] = twoij - twokw;
    result[0, 2] = twojw + twoik;
    result[1, 0] = twoij + twokw;
    result[1, 1] = w2 - i2 + j2 - k2;
    result[1, 2] = twojk - twoiw;
    result[2, 0] = twoik - twojw;
    result[2, 1] = twojk + twoiw;
    result[2, 2] = w2 - i2 - j2 + k2;

    return result;

@ti.kernel
def quatNormalize(quat : Vector4d):
    # w i j k
    # 0 1 2 3
    w2 : ti.f64 = quat[0] * quat[0];
    i2 : ti.f64 = quat[1] * quat[1];
    j2 : ti.f64 = quat[2] * quat[2];
    k2 : ti.f64 = quat[3] * quat[3];
    length : ti.f64 = ti.math.sqrt(w2 + i2 + j2 + k2);
    
    result : Vector4d;
    result[0] = quat[0] / length;
    result[1] = quat[1] / length;
    result[2] = quat[2] / length;
    result[3] = quat[3] / length;
    return result;

# Initialization - read particle dataset
# Di Peng
@ti.kernel
def init():
    fp = open("input.p4p", encoding="UTF-8");
    line : str = fp.readline(); # "TIMESTEP  PARTICLES" line
    line = fp.readline(); # "0 18112" line
    global n;
    n = ti.i32(line.split(' ')[1]);
    # Initialize particles
    global gf;
    gf = Grain.field(Grain, shape=(n,));
    global cf;
    cf = Contact.field(Contact, shape=(n, n));
    cf.fill(Contact()) # TODO: fill all NULLs
    line = fp.readline(); # "ID  GROUP  VOLUME  MASS  PX  PY  PZ  VX  VY  VZ" line
    # Processing particles
    while (True):
        line = fp.readline();
        if (not line): break;
        i : ti.f64 = ti.f64(line.split(' ')[0]);
        # GROUP omitted
        volume : ti.f64 = ti.f64(line.split(' ')[2]);
        mass : ti.f64 = ti.f64(line.split(' ')[3]);
        px : ti.f64 = ti.f64(line.split(' ')[4]);
        py : ti.f64 = ti.f64(line.split(' ')[5]);
        pz : ti.f64 = ti.f64(line.split(' ')[6]);
        vx : ti.f64 = ti.f64(line.split(' ')[7]);
        vy : ti.f64 = ti.f64(line.split(' ')[8]);
        vz : ti.f64 = ti.f64(line.split(' ')[9]);
        density : ti.f64 = mass / volume;
        radius : ti.f64 = ti.math.pow(volume * 3.0 / 4.0 / ti.math.pi, 1.0 / 3.0);
        inertia : ti.f64 = 2.0 / 5.0 * mass * radius * radius;
        gf[i].density = density;
        gf[i].mass = mass;
        gf[i].radius = radius;
        gf[i].elasticModulus = 7e10; # Di Peng: hard coding; need to be modified in the future
        gf[i].poissonRatio = 0.25; # Di Peng: hard coding; need to be modified in the future
        gf[i].position = Vector3d(px, py, pz);
        gf[i].velocity = Vector3d(vx, vy, vz);
        gf[i].acceleration = Vector3d(0.0, 0.0, 0.0);
        gf[i].force = Vector3d(0.0, 0.0, 0.0);
        gf[i].quaternion = Vector4d(1.0, 0.0, 0.0, 0.0);
        gf[i].omega = Vector3d(0.0, 0.0, 0.0);
        gf[i].omega_dot = Vector3d(0.0, 0.0, 0.0);
        gf[i].moment = Vector3d(0.0, 0.0, 0.0);
        gf[i].inertia = inertia * ti.Matrix.diag(3, 1.0);
    fp.close();
    # Input wall
    # Di Peng: hard coding: need to be modified in the future
    wf = Wall.field(Wall, shape = (1,));
    wf[0].normal = Vector3d(1.0, 0.0, 0.0); # Outer normal vector of the wall, [A, B, C]
    wf[0].distance = 0.1; # Distance between origin and the wall, D
    # Material properties
    wf[0].density = 7800; # Density of the wall
    wf[0].elasticModulus = 2e11; # Elastic modulus of the wall
    wf[0].poissonRatio = 0.25; # Poisson's ratio of the wall

    wcf = Contact.field(Contact, shape = (1,n,));

# NVE integrator
@ti.kernel
def update():
    for i in gf:
        # Translational
        # Velocity Verlet integrator is adopted
        # Reference: https://www.algorithm-archive.org/contents/verlet_integration/verlet_integration.html
        gf[i].acceleration = gf[i].force / gf[i].mass;
        gf[i].position += gf[i].velocity * dt + 0.5 * gf[i].acceleration * dt ** 2;
        gf[i].velocity += gf[i].acceleration * dt;
        # Rotational
        # Angular acceleration should be calculated via Euler's equation for rigid body
        # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
        # https://doi.org/10.1002/nme.6568
        # Eqs. (5)-(16)
        rotational_matrix = quat2RotMatrix(gf[i].quaternion);
        moment_local = rotational_matrix @ gf[i].moment;
        omega_local = rotational_matrix @ gf[i].omega;
        omega_dot_local = ti.Matrix.inverse(gf[i].inertia) @ (moment_local - omega_local.cross(gf[i].inertia @ omega_local));
        gf[i].omega_dot = ti.Matrix.inverse(rotational_matrix) @ omega_dot_local;
        # Update particle orientation
        # Reference: Lu et al. (2015) Discrete element models for non-spherical particle systems: From theoretical developments to applications.
        # http://dx.doi.org/10.1016/j.ces.2014.11.050
        # Eq. (6)
        # Originally from Langston et al. (2004) Distinct element modelling of non-spherical frictionless particle flow.
        # https://doi.org/10.1016/j.ces.2003.10.008
        dq0 = - 0.5 * (gf[i].quaternion[1] * gf[i].omega[0] + gf[i].quaternion[2] * gf[i].omega[1] + gf[i].quaternion[3] * gf[i].omega[2]);
        dq1 = + 0.5 * (gf[i].quaternion[0] * gf[i].omega[0] - gf[i].quaternion[3] * gf[i].omega[1] + gf[i].quaternion[2] * gf[i].omega[2]);
        dq2 = + 0.5 * (gf[i].quaternion[3] * gf[i].omega[0] + gf[i].quaternion[0] * gf[i].omega[1] + gf[i].quaternion[1] * gf[i].omega[2]);
        dq3 = + 0.5 * (-gf[i].quaternion[2] * gf[i].omega[0] + gf[i].quaternion[1] * gf[i].omega[1] + gf[i].quaternion[0] * gf[i].omega[2]);
        gf[i].quaternion[0] += dq0;
        gf[i].quaternion[1] += dq1;
        gf[i].quaternion[2] += dq2;
        gf[i].quaternion[3] += dq3;
        gf[i].quaternion = quatNormalize(gf[i].quaternion);
        # Update angular velocity
        gf[i].omega += gf[i].omega_dot * dt;

# Add GLOBAL damping
# for EBPM, GLOBAL damping is assigned to particles
@ti.kernel
def apply_bc():
    t_d : ti.f64 = 0.0; # Di Peng: hard coding - should be modified in the future
    for i in gf:
        damp_force : Vector3d = Vector3d(0.0, 0.0, 0.0);
        damp_moment : Vector3d = Vector3d(0.0, 0.0, 0.0);
        for j in range(3):
            damp_force[j] = -t_d * ti.abs(gf[i].force[j]) * ti.math.sign(gf[i].velocity[j]);
            damp_moment[j] = -t_d * ti.abs(gf[i].moment[j]) * ti.math.sign(gf[i].omega[j]);
        gf[i].force += damp_force;
        gf[i].moment += damp_moment;

# Contact resolution and evaluation
# Contact model is implemented here
@ti.func
def evaluate(i : ti.i32, j : ti.i32):
    # Contact resolution
    # Find out rotation matrix
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    a : Vector3d = ti.math.normalize(gf[j].position - gf[i].position);
    b : Vector3d = Vector3d(1.0, 0.0, 0.0); # Local x coordinate
    v : Vector3d = ti.math.cross(a, b);
    s : ti.f64 = ti.math.length(v);
    c : ti.f64 = ti.math.dot(a, b);
    vx : DEMMatrix = [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]];
    cf[i, j].rotationMatrix = ti.Matrix.diag(3, 1.0) + vx + ((1.0 - c) / s**2) * ti.math.dot(vx, vx);

    cf[i, j].length = ti.math.length(gf[j].position - gf[i].position);
    # Contact evaluation (with contact model)
    if (cf[i, j].isbonded): # Bonded, use EBPM
        disp_a: Vector3d = cf[i, j].rotationMatrix @ gf[i].velocity * dt;
        disp_b: Vector3d = cf[i, j].rotationMatrix @ gf[j].velocity * dt;
        rot_a: Vector3d = cf[i, j].rotationMatrix @ gf[i].omega * dt;
        rot_b: Vector3d = cf[i, j].rotationMatrix @ gf[j].omega * dt;
        dispVector : EBPMForceDisplacementVector = [disp_a, rot_a, disp_b, rot_b];
        r_b : ti.f64 = cf[i, j].radius_ratio * ti.math.min(gf[i].radius, gf[j].radius);
        L_b : ti.f64 = cf[i, j].length;
        E_b : ti.f64 = cf[i, j].elasticModulus;
        nu: ti.f64 = cf[i, j].poissionRatio;
        I_b : ti.f64 = (r_b ** 4) * ti.math.pi / 4.0;
        phi : ti.f64 = 20.0 / 3.0 * (r_b ** 2) / (L_b ** 2) * (1.0 + nu);
        A_b : ti.f64 = ti.math.pi * (r_b ** 2);
        k1 : ti.f64 = E_b * A_b / L_b;
        k2 : ti.f64 = 12.0 * E_b * I_b / (L_b ** 3) / (1.0 + phi);
        k3 : ti.f64 = 6.0 * E_b * I_b / (L_b ** 2) / (1.0 + phi);
        k4 : ti.f64 = E_b * I_b / L_b / (1.0 + phi);
        k5 : ti.f64 = E_b * I_b * (4.0 + phi) / L_b / (1.0 + phi);
        k6 : ti.f64 = E_b * I_b * (2.0 - phi) / L_b / (1.0 + phi);
        K : EBPMStiffnessMatrix = [
            [  k1,   0,   0,   0,   0,   0, -k1,   0,   0,   0,   0,   0],
            [   0,  k2,   0,   0,   0,  k3,   0, -k2,   0,   0,   0,  k3],
            [   0,   0,  k2,   0, -k3,   0,   0,   0, -k2,   0, -k3,   0],
            [   0,   0,   0,  k4,   0,   0,   0,   0,   0, -k4,   0,   0],
            [   0,   0, -k3,   0,   k5,  0,   0,   0,  k3,   0,  k6,   0],
            [   0,   k3,  0,   0,   0,  k5,   0, -k3,   0,   0,   0,  k6],
            [ -k1,   0,   0,   0,   0,   0,  k1,   0,   0,   0,   0,   0],
            [   0, -k2,   0,   0,   0,  k3,   0,  k2,   0,   0,   0, -k3],
            [   0,   0, -k2,   0,  k3,   0,   0,   0,  k2,   0,  k3,   0],
            [   0,   0,   0, -k4,   0,   0,   0,   0,   0,  k4,   0,   0],
            [   0,   0, -k3,   0,  k6,   0,   0,   0,  k3,   0,  k5,   0],
            [   0,  k3,   0,   0,   0,  k6,   0, -k3,   0,   0,   0,  k5]
        ];
        forceVector : EBPMForceDisplacementVector = K @ dispVector;
        cf[i, j].force_a = Vector3d(forceVector[0, 0], forceVector[1, 0], forceVector[2, 0]);
        cf[i, j].moment_a = Vector3d(forceVector[3, 0], forceVector[4, 0], forceVector[5, 0]);
        cf[i, j].force_b = Vector3d(forceVector[6, 0], forceVector[7, 0], forceVector[8, 0]);
        cf[i, j].moment_b = Vector3d(forceVector[9, 0], forceVector[10, 0], forceVector[11, 0]);
        
        # Check whether the bond fails
        sigma_c_a : ti.f64 = cf[i, j].force_b[0] / A_b - r_b / I_b * ti.math.sqrt(cf[i, j].moment_a[1] ** 2 + cf[i, j].moment_a[2] ** 2);
        sigma_c_b : ti.f64 = cf[i, j].force_b[0] / A_b - r_b / I_b * ti.math.sqrt(cf[i, j].moment_b[1] ** 2 + cf[i, j].moment_b[2] ** 2);
        sigma_c_max : ti.f64 = -ti.math.min(sigma_c_a, sigma_c_b);
        sigma_t_a : ti.f64 = sigma_c_a;
        sigma_t_b : ti.f64 = sigma_c_b;
        sigma_t_max : ti.f64 = max(sigma_t_a, sigma_t_b);
        tau_max : ti.f64 = ti.abs(cf[i, j].moment_a[0]) * r_b / 2.0 / I_b + 4.0 / 3.0 / A_b * ti.math.sqrt(cf[i, j].force_a[1] ** 2 + cf[i, j].force_a[2] ** 2);
        if (sigma_c_max >= cf[i, j].compressiveStrength): # Compressive failure
            cf[i, j].isBonded = 0;
            cf[i, j].isActive = 0;
        elif (sigma_t_max >= cf[i, j].tensileStrength): # Tensile failure
            cf[i, j].isBonded = 0;
            cf[i, j].isActive = 0;
        elif (tau_max >= cf[i, j].shearStrength): # Shear failure
            cf[i, j].isBonded = 0;
            cf[i, j].isActive = 0;
        else: # Intact bond, need to conduct force to particles
            # Notice the inverse of signs due to Newton's third law
            # and LOCAL to GLOBAL coordinates
            gf[i].force += ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-cf[i, j].force_a);
            gf[j].force += ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-cf[i, j].force_b);
            gf[i].moment += ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-cf[i, j].moment_a);
            gf[j].moment += ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-cf[i, j].moment_b);
    else: # Non-bonded, use Hertz-Mindlin
        # Calculation relative translational and rotational displacements
        # Need to include the impact of particle rotation in contact relative translational displacement
        # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
        # https://doi.org/10.1002/nme.6568
        # Eqs. (1)-(2)
        # Implementation reference: https://github.com/CFDEMproject/LIGGGHTS-PUBLIC/blob/master/src/surface_model_default.h
        # Lines 140-189
        gap : ti.f64 = cf[i, j].length - gf[i].radius - gf[j].radius; # gap must be negative to ensure an intact contact
        delta_n : ti.f64 = ti.abs(gap); # For parameter calculation only
        cf[i, j].position = gf[i].position + ti.math.normalize(gf[j].position - gf[i].position) * (gf[i].radius - delta_n);
        r_i : Vector3d = cf[i, j].position - gf[i];
        r_j : Vector3d = cf[i, j].position - gf[j];
        # Velocity of a point on the surface of a rigid body
        v_c_i : Vector3d = ti.math.cross(gf[i].omega, r_i) + gf[i].velocity;
        v_c_j : Vector3d = ti.math.cross(gf[j].omega, r_j) + gf[j].velocity;
        v_c : Vector3d = cf[i, j].rotationMatrix @ (v_c_j - v_c_i); # LOCAL coordinate
        # Parameter calculation
        # Reference: https://www.cfdem.com/media/DEM/docu/gran_model_hertz.html
        Y_star : ti.f64 = 1.0 / ((1.0 - gf[i].poissonRatio ** 2) / gf[i].elasticModulus + (1.0 - gf[j].poissonRatio ** 2) / gf[j].elasticModulus);
        G_star : ti.f64 = 1.0 / (2.0 * (2.0 - gf[i].poissonRatio) * (1.0 + gf[i].poissonRatio) / gf[i].elasticModulus + 2.0 * (2.0 - gf[j].poissonRatio) * (1.0 + gf[j].poissonRatio) / gf[j].elasticModulus);
        R_star : ti.f64 = 1.0 / (1.0 / gf[i].radius + 1.0 / gf[j].radius);
        m_star : ti.f64 = 1.0 / (1.0 / gf[i].mass + 1.0 / gf[j].mass);
        beta : ti.f64 = ti.math.log(cf[i, j].coefficientRestitution) / ti.math.sqrt(ti.math.log(cf[i, j].coefficientRestitution) ** 2 + ti.math.pi ** 2);
        S_n : ti.f64 = 2.0 * Y_star * ti.math.sqrt(R_star * delta_n);
        S_t : ti.f64 = 8.0 * G_star * ti.math.sqrt(R_star * delta_n);
        k_n : ti.f64 = 4.0 / 3.0 * Y_star * ti.math.sqrt(R_star * delta_n);
        gamma_n : ti.f64 = - 2.0 * beta * ti.math.sqrt(5.0 / 6.0 * S_n * m_star); # Check whether gamma_n >= 0
        k_t : ti.f64 = 8.0 * G_star * ti.math.sqrt(R_star * delta_n);
        gamma_t : ti.f64 = - 2.0 * beta * ti.math.sqrt(5.0 / 6.0 * S_t * m_star); # Check whether gamma_t >= 0

        # Shear displacement increments
        shear_increment : Vector3d = v_c * dt;
        shear_increment[0] = 0.0; # Remove the normal direction
        cf[i, j].shear_displacement += shear_increment;
        # Normal direction - LOCAL - the force towards particle i
        F : Vector3d = Vector3d(0.0, 0.0, 0.0);
        F[0] = k_n * gap - gamma_n * v_c[0];
        # Shear direction - LOCAL - the force towards particle i
        try_shear_force : Vector3d = k_t * cf[i, j].shear_displacement;
        if (ti.math.length(try_shear_force) >= cf[i, j].coefficientFriction * F[0]): # Sliding
            ratio : ti.f64 = cf[i, j].coefficientFriction * F[0] / ti.math.length(try_shear_force);
            F[1] = try_shear_force[1] / ratio;
            F[2] = try_shear_force[2] / ratio;
            cf[i, j].shear_displacement[1] = F[1] / k_t;
            cf[i, j].shear_displacement[2] = F[2] / k_t;
        else: # No sliding
            F[1] = k_t * v_c[1] * dt - gamma_t * v_c[1];
            F[2] = k_t * v_c[2] * dt - gamma_t * v_c[2];
        
        # No moment is conducted in Hertz-Mindlin model
        
        # Assigning contact force to particles
        # Notice the inverse of signs due to Newton's third law
        # and LOCAL to GLOBAL coordinates
        gf[i].force += ti.Matrix.inverse(cf[i, j].rotationMatrix) @ F;
        gf[j].force += ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-F);
        # As the force is at contact position
        # additional moments will be assigned to particles
        # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
        # https://doi.org/10.1002/nme.6568
        # Eqs. (3)-(4)
        gf[i].moment += ti.math.cross(r_i, ti.Matrix.inverse(cf[i, j].rotationMatrix) @ F);
        gf[j].moment += ti.math.cross(r_j, ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-F));  

# Particle-particle contact detection
@ti.func
def resolve(i : ti.i32, j : ti.i32):
    global gf;
    global cf;
    # Particle-particle contacts
    if (cf[i, j].isActive): # Existing contact
        if (cf[i, j].isBonded): # Bonded contact
            evaluate(i, j); # Bonded contact must exist. Go to evaluation and if bond fails, the contact state will change thereby.
        else: # Non-bonded contact, should check whether two particles are still in contact
            if (- gf[i].radius - gf[j].radius + ti.math.length(gf[j].position - gf[i].position) < 0): # Use PFC's gap < 0 criterion
                evaluate(i, j);
            else:
                cf[i, j].isActive = 0;
    else:
        if (- gf[i].radius - gf[j].radius + ti.math.length(gf[j].position - gf[i].position) < 0): # Use PFC's gap < 0 criterion
            cf[i, j] = Contact( # Hertz-Mindlin model
                isActive = 1,
                isBonded = 0,
                coefficientFriction = 0.3, # Di Peng: hard coding; need to be modified in the future
                coefficientRestitution = 0.9, # Di Peng: hard coding; need to be modified in the future
                coefficientRollingResistance = 0.01, # Di Peng: hard coding; need to be modified in the future
                shear_displacement = Vector3d(0.0, 0.0, 0.0)
            );
            evaluate(i, j); # Send to evaluation using Hertz-Mindlin contact model

# Particle-wall contact evaluation
# Contact model is Hertz-Mindlin
@ti.func
def evaluate_wall(i : ti.i32, j : ti.i32): # i is particle, j is wall
    # Contact resolution
    # Find out rotation matrix
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    a : Vector3d = wf[j].normal;
    b : Vector3d = Vector3d(1.0, 0.0, 0.0); # Local x coordinate
    v : Vector3d = ti.math.cross(a, b);
    s : ti.f64 = ti.math.length(v);
    c : ti.f64 = ti.math.dot(a, b);
    vx : DEMMatrix = [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]];
    wcf[i, j].rotationMatrix = ti.Matrix.diag(3, 1.0) + vx + ((1.0 - c) / s**2) * ti.math.dot(vx, vx);

    # Calculation relative translational and rotational displacements
    distance : ti.f64 = ti.math.dot(gf[i].position, wf[j].normal) - wf[j].distance; # Distance < 0 means that particle is beneath the plane
    gap : ti.f64 = ti.abs(distance) - gf[i].radius; # gap must be negative
    delta_n : ti.f64 = ti.abs(gap); # For parameter calculation only
    r_i : Vector3d = - distance * wf[j].normal / ti.abs(distance) * (ti.abs(distance) + delta_n / 2.0);
    wcf[i, j].position = gf[i].position + r_i;
    # Velocity of a point on the surface of a rigid body
    v_c_i : Vector3d = ti.math.cross(gf[i].omega, r_i) + gf[i].velocity;
    v_c : Vector3d = - v_c_i; # LOCAL coordinate
    # Parameter calculation
    # Reference: https://www.cfdem.com/media/DEM/docu/gran_model_hertz.html
    Y_star : ti.f64 = 1.0 / ((1.0 - gf[i].poissonRatio ** 2) / gf[i].elasticModulus + (1.0 - wf[j].poissonRatio ** 2) / wf[j].elasticModulus);
    G_star : ti.f64 = 1.0 / (2.0 * (2.0 - gf[i].poissonRatio) * (1.0 + gf[i].poissonRatio) / gf[i].elasticModulus + 2.0 * (2.0 - wf[j].poissonRatio) * (1.0 + wf[j].poissonRatio) / wf[j].elasticModulus);
    R_star : ti.f64 = gf[i].radius;
    m_star : ti.f64 = gf[i].mass;
    beta : ti.f64 = ti.math.log(wcf[i, j].coefficientRestitution) / ti.math.sqrt(ti.math.log(wcf[i, j].coefficientRestitution) ** 2 + ti.math.pi ** 2);
    S_n : ti.f64 = 2.0 * Y_star * ti.math.sqrt(R_star * delta_n);
    S_t : ti.f64 = 8.0 * G_star * ti.math.sqrt(R_star * delta_n);
    k_n : ti.f64 = 4.0 / 3.0 * Y_star * ti.math.sqrt(R_star * delta_n);
    gamma_n : ti.f64 = - 2.0 * beta * ti.math.sqrt(5.0 / 6.0 * S_n * m_star); # Check whether gamma_n >= 0
    k_t : ti.f64 = 8.0 * G_star * ti.math.sqrt(R_star * delta_n);
    gamma_t : ti.f64 = - 2.0 * beta * ti.math.sqrt(5.0 / 6.0 * S_t * m_star); # Check whether gamma_t >= 0

    # Shear displacement increments
    shear_increment : Vector3d = v_c * dt;
    shear_increment[0] = 0.0; # Remove the normal direction
    wcf[i, j].shear_displacement += shear_increment;
    # Normal direction - LOCAL - the force towards particle i
    F : Vector3d = Vector3d(0.0, 0.0, 0.0);
    F[0] = k_n * gap - gamma_n * v_c[0];
    # Shear direction - LOCAL - the force towards particle i
    try_shear_force : Vector3d = k_t * wcf[i, j].shear_displacement;
    if (ti.math.length(try_shear_force) >= wcf[i, j].coefficientFriction * F[0]): # Sliding
        ratio : ti.f64 = wcf[i, j].coefficientFriction * F[0] / ti.math.length(try_shear_force);
        F[1] = try_shear_force[1] / ratio;
        F[2] = try_shear_force[2] / ratio;
        wcf[i, j].shear_displacement[1] = F[1] / k_t;
        wcf[i, j].shear_displacement[2] = F[2] / k_t;
    else: # No sliding
        F[1] = k_t * v_c[1] * dt - gamma_t * v_c[1];
        F[2] = k_t * v_c[2] * dt - gamma_t * v_c[2];
        
    # No moment is conducted in Hertz-Mindlin model
        
    # Assigning contact force to particles
    # Notice the inverse of signs due to Newton's third law
    # and LOCAL to GLOBAL coordinates
    # As the force is at contact position
    # additional moments will be assigned to particles
    gf[i].force += ti.Matrix.inverse(cf[i, j].rotationMatrix) @ F;
    # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
    # https://doi.org/10.1002/nme.6568
    # Eqs. (3)-(4)
    gf[i].moment += ti.math.cross(r_i, ti.Matrix.inverse(cf[i, j].rotationMatrix) @ F);
    
# Particle-wall contact detection
@ti.func
def resolve_wall():
    global gf;
    global wf;
    global wcf;
    # Di Peng: particle-wall neighboring search has not been implemented
    # and thus all particles will be contact detection with the wall
    for i in gf:
        for j in wf:
            # Particle-wall contacts
            if (wcf[i, j].isActive): # Existing contact
                if (ti.abs(ti.math.dot(gf[i].position, wf[j].normal) - wf[j].distance) >= gf[i].radius): # Non-contact
                    wcf[i, j].isActive = 0;
                else: # Contact
                    evaluate_wall(i, j);
            else:
                if (ti.abs(ti.math.dot(gf[i].position, wf[j].normal) - wf[j].distance) < gf[i].radius): # Contact
                    wcf[i, j] = Contact( # Hertz-Mindlin model
                        isActive = 1,
                        isBonded = 0,
                        coefficientFriction = 0.35, # Di Peng: hard coding; need to be modified in the future
                        coefficientRestitution = 0.7, # Di Peng: hard coding; need to be modified in the future
                        coefficientRollingResistance = 0.01, # Di Peng: hard coding; need to be modified in the future
                        shear_displacement = Vector3d(0.0, 0.0, 0.0)
                    );
                    evaluate_wall(i, j);

# Bonding
# Similar to contact, but runs only once at the beginning
@ti.kernel
def bond(gf: ti.template()):
    '''
    Handle the collision between grains.
    '''
    # for i in gf:
    #    gf[i].f = vec(0., gravity * gf[i].m)  # Apply gravity.

    grain_count.fill(0)

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, ti.i32)
        grain_count[grid_idx] += 1

    for i in range(grid_n):
        sum = 0
        for j in range(grid_n):
            sum += grain_count[i, j]
        column_sum[i] = sum

    prefix_sum[0, 0] = 0

    ti.loop_config(serialize=True)
    for i in range(1, grid_n):
        prefix_sum[i, 0] = prefix_sum[i - 1, 0] + column_sum[i - 1]

    for i in range(grid_n):
        for j in range(grid_n):
            if j == 0:
                prefix_sum[i, j] += grain_count[i, j]
            else:
                prefix_sum[i, j] = prefix_sum[i, j - 1] + grain_count[i, j]

            linear_idx = i * grid_n + j

            list_head[linear_idx] = prefix_sum[i, j] - grain_count[i, j]
            list_cur[linear_idx] = list_head[linear_idx]
            list_tail[linear_idx] = prefix_sum[i, j]

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, ti.i32)
        linear_idx = grid_idx[0] * grid_n + grid_idx[1]
        grain_location = ti.atomic_add(list_cur[linear_idx], 1)
        particle_id[grain_location] = i

    # Brute-force collision detection
    '''
    for i in range(n):
        for j in range(i + 1, n):
            resolve(i, j)
    '''

    # Fast collision detection
    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, ti.i32)
        x_begin = max(grid_idx[0] - 1, 0)
        x_end = min(grid_idx[0] + 2, grid_n)

        y_begin = max(grid_idx[1] - 1, 0)
        y_end = min(grid_idx[1] + 2, grid_n)

        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                neigh_linear_idx = neigh_i * grid_n + neigh_j
                for p_idx in range(list_head[neigh_linear_idx],
                                   list_tail[neigh_linear_idx]):
                    j = particle_id[p_idx]
                    if i < j:
                        # Contact detection is performed here
                        # Using CONTACT RADIUS of the spheres
                        # To determine whether a bond is assigned between two particles
                        contact_radius_i : ti.f64 = gf[i].radius * 1.1; # Di Peng: hard coding; need to be modified in the future
                        contact_radius_j : ti.f64 = gf[j].radius * 1.1; # Di Peng: hard coding; need to be modified in the future
                        if (ti.math.length(gf[j].position - gf[i].position) - contact_radius_i - contact_radius_j < 0.0):
                            cf[i, j] = Contact( # Forced to bond contact
                                isActive = 1,
                                isBonded = 1,
                                # EBPM parameters
                                # Di Peng: hard coding; need to be modified in the future
                                radius_ratio = 0.5,
                                elasticModulus = 28e9,
                                poissonRatio = 0.2,
                                compressiveStrength = 3e8,
                                tensileStrength = 6e7,
                                shearStrength = 6e7,
                                force_a = Vector3d(0.0, 0.0, 0.0),
                                moment_a = Vector3d(0.0, 0.0, 0.0),
                                force_b = Vector3d(0.0, 0.0, 0.0),
                                moment_b = Vector3d(0.0, 0.0, 0.0),
                            );

# Neighboring search
@ti.kernel
def contact(gf: ti.template()):
    '''
    Handle the collision between grains.
    '''
    # for i in gf:
    #    gf[i].f = vec(0., gravity * gf[i].m)  # Apply gravity.

    grain_count.fill(0)

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, ti.i32)
        grain_count[grid_idx] += 1

    for i in range(grid_n):
        sum = 0
        for j in range(grid_n):
            sum += grain_count[i, j]
        column_sum[i] = sum

    prefix_sum[0, 0] = 0

    ti.loop_config(serialize=True)
    for i in range(1, grid_n):
        prefix_sum[i, 0] = prefix_sum[i - 1, 0] + column_sum[i - 1]

    for i in range(grid_n):
        for j in range(grid_n):
            if j == 0:
                prefix_sum[i, j] += grain_count[i, j]
            else:
                prefix_sum[i, j] = prefix_sum[i, j - 1] + grain_count[i, j]

            linear_idx = i * grid_n + j

            list_head[linear_idx] = prefix_sum[i, j] - grain_count[i, j]
            list_cur[linear_idx] = list_head[linear_idx]
            list_tail[linear_idx] = prefix_sum[i, j]

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, ti.i32)
        linear_idx = grid_idx[0] * grid_n + grid_idx[1]
        grain_location = ti.atomic_add(list_cur[linear_idx], 1)
        particle_id[grain_location] = i

    # Brute-force collision detection
    '''
    for i in range(n):
        for j in range(i + 1, n):
            resolve(i, j)
    '''

    # Fast collision detection
    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, ti.i32)
        x_begin = max(grid_idx[0] - 1, 0)
        x_end = min(grid_idx[0] + 2, grid_n)

        y_begin = max(grid_idx[1] - 1, 0)
        y_end = min(grid_idx[1] + 2, grid_n)

        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                neigh_linear_idx = neigh_i * grid_n + neigh_j
                for p_idx in range(list_head[neigh_linear_idx],
                                   list_tail[neigh_linear_idx]):
                    j = particle_id[p_idx]
                    if i < j:    
                        resolve(i, j);

# Running script
init();
bond();
gui = ti.GUI('Taichi DEM', (window_size, window_size))
# step = 0

# GPU grid parameters
list_head = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_cur = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_tail = ti.field(dtype=ti.i32, shape=grid_n * grid_n)

grain_count = ti.field(dtype=ti.i32,
                       shape=(grid_n, grid_n),
                       name="grain_count")
column_sum = ti.field(dtype=ti.i32, shape=grid_n, name="column_sum")
prefix_sum = ti.field(dtype=ti.i32, shape=(grid_n, grid_n), name="prefix_sum")
particle_id = ti.field(dtype=ti.i32, shape=n, name="particle_id")

if SAVE_FRAMES:
    os.makedirs('output', exist_ok=True)

while gui.running:
    for step in range(nsteps):
        for i in gf:
            gf[i].force = Vector3d(0.0, 0.0, 0.0);
            gf[i].moment = Vector3d(0.0, 0.0, 0.0);
        # Particle-particle
        contact(gf); # Neighboring search
        resolve(); # Contact detection, resolution and evaluation
        # Particle-wall
        resolve_wall();
        apply_bc(); # Add GLOBAL damping
        update(); # Time integration
    
    # TODO: consider saving when step mod saving_interval_steps == 0
    pos = gf.p.to_numpy()
    r = gf.r.to_numpy() * window_size
    gui.circles(pos, radius=r)
    if SAVE_FRAMES:
        gui.show(f'output/{step:06d}.png')
    else:
        gui.show()
    # step += 1
