# Complex DEM simulation using Taichi DEM
# 
# Authors:
# Denver Pilphis (Complex DEM mechanism implementation)
# MuGdxy (GPU HPC optimization)
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
# TODO List:
# 1. cf MUST BE OPTIMIZED, otherwise large scale calculations cannot be performed.
# 2. HPC optimization has NOT been implemented for particle-wall contacts.

import taichi as ti
import taichi.math as tm
import os
import numpy as np
#=====================================
# Hard Code For Dev Test
#=====================================
# WallDistance:float = 0.1 # Denver Pilphis: The wall used in this example is a specific wall
# Wall distance is wall property and should not be modified in global scenario
MaxParticleCount:int = 10000000; # Limit number of particles for efficiency benchmark

#=====================================
# Environmental Variables
#=====================================
DoublePrecisionTolerance: float = 1e-12; # Boundary between zeros and non-zeros

# init taichi context
ti.init(arch=ti.gpu)
#=====================================
# Type Definition
#=====================================
Real = ti.f64
Integer = ti.i32
Vector2 = ti.types.vector(2, Real)
Vector3 = ti.types.vector(3, Real)
Vector4 = ti.types.vector(4, Real)
Matrix3x3 = ti.types.matrix(3, 3, Real)

DEMMatrix = Matrix3x3
EBPMStiffnessMatrix = ti.types.matrix(12, 12, Real)
EBPMForceDisplacementVector = ti.types.vector(12, Real)


#=====================================
# Utils
#=====================================
@ti.func
def Zero3x3() -> Matrix3x3:
    return Matrix3x3([[0,0,0],[0,0,0],[0,0,0]])


# Add a math function: quaternion to rotation matrix
# References:
# https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
# Lines 511-534, https://github.com/CFDEMproject/LIGGGHTS-PUBLIC/blob/master/src/math_extra_liggghts_nonspherical.h
@ti.func
def quat2RotMatrix(quat : Vector4) -> Matrix3x3:
    # w i j k
    # 0 1 2 3
    w2 = quat[0] * quat[0]
    i2 = quat[1] * quat[1]
    j2 = quat[2] * quat[2]
    k2 = quat[3] * quat[3]
    
    twoij = 2.0 * quat[1] * quat[2]
    twoik = 2.0 * quat[1] * quat[3]
    twojk = 2.0 * quat[2] * quat[3]
    twoiw = 2.0 * quat[1] * quat[0]
    twojw = 2.0 * quat[2] * quat[0]
    twokw = 2.0 * quat[3] * quat[0]

    result = Zero3x3()
    result[0, 0] = w2 + i2 - j2 - k2
    result[0, 1] = twoij - twokw
    result[0, 2] = twojw + twoik
    result[1, 0] = twoij + twokw
    result[1, 1] = w2 - i2 + j2 - k2
    result[1, 2] = twojk - twoiw
    result[2, 0] = twoik - twojw
    result[2, 1] = twojk + twoiw
    result[2, 2] = w2 - i2 - j2 + k2

    return result


#======================================
# data class definition
#======================================
# Particle in DEM
# Denver Pilphis: keep spherical shape first, added particle attributes to make the particle kinematically complete
@ti.dataclass
class Grain:
    # Material attributes
    # At this stage, the material attributes are written here
    density: Real  # Density, double
    mass: Real # Mass, double
    radius: Real  # Radius, double
    elasticModulus: Real  # Elastic modulus, double
    poissonRatio: Real # Poisson's ratio, double
    # Translational attributes, all in GLOBAL coordinates
    position: Vector3  # Position, Vector3
    velocity: Vector3  # Velocity, Vector3
    acceleration: Vector3  # Acceleration, Vector3
    force: Vector3  # Force, Vector3
    # Rotational attributes, all in GLOBAL coordinates
    quaternion: Vector4  # Quaternion, Vector4, order in [w, x, y, z]
    omega: Vector3  # Angular velocity, Vector3
    omega_dot: Vector3  # Angular acceleration, Vector3
    inertia: DEMMatrix # Moment of inertia tensor, 3 * 3 matrix with double
    moment: Vector3 # Total moment (including torque), Vector3


# Wall in DEM
# Only premitive wall is implemented
@ti.dataclass
class Wall:
    # Wall equation: Ax + By + Cz - D = 0
    # Reference: Peng and Hanley (2019) Contact detection between convex polyhedra and superquadrics in discrete element codes.
    # https://doi.org/10.1016/j.powtec.2019.07.082
    # Eq. (8)
    normal: Vector3 # Outer normal vector of the wall, [A, B, C]
    distance: Real # Distance between origin and the wall, D
    # Material properties
    density: Real # Density of the wall
    elasticModulus: Real # Elastic modulus of the wall
    poissonRatio: Real # Poisson's ratio of the wall


# Contact in DEM
# In this example, the Edinburgh Bond Particle Model (EBPM), along with Hertz-Mindlin model, is implemented
# Reference: Brown et al. (2014) A bond model for DEM simulation of cementitious materials and deformable structures.
# https://doi.org/10.1007/s10035-014-0494-4
# Reference: Mindlin and Deresiewicz (1953) Elastic spheres in contact under varying oblique forces.
# https://doi.org/10.1115/1.4010702
@ti.dataclass
class Contact:
    # Contact status
    isActive : Integer # Contact exists: 1 - exist 0 - not exist
    isBonded : Integer # Contact is bonded: 1 - bonded, use EBPM 0 - unbonded, use Hertz-Mindlin
    # Common Parameters
    rotationMatrix : DEMMatrix # Rotation matrix from global to local system of the contact
    position : Vector3 # Position of contact point in GLOBAL coordinate
    # EBPM parameters
    radius_ratio : Real # Section radius ratio
    # radius : Real # Section radius: r = rratio * min(r1, r2), temporarily calculated in evaluation
    length : Real # Length of the bond
    elasticModulus : Real # Elastic modulus of the bond
    poissonRatio : Real # Possion's ratio of the bond
    compressiveStrength: Real # Compressive strength of the bond
    tensileStrength: Real # Tensile strength of the bond
    shearStrength: Real # Shear strength of the bond
    force_a : Vector3 # Contact force at side a in LOCAL coordinate
    moment_a : Vector3 # Contact moment/torque at side a in LOCAL coordinate
    force_b : Vector3 # Contact force at side b in LOCAL coordinate
    moment_b : Vector3 # Contact moment/torque at side b in LOCAL coordinate
    # Hertz-Mindlin parameters
    # normalStiffness: Real # Normal stiffness, middleware parameter only
    # shearStiffness: Real # Shear stiffness, middleware parameter only
    coefficientFriction: Real # Friction coefficient, double
    coefficientRestitution: Real # Coefficient of resitution, double
    coefficientRollingResistance: Real # Coefficient of rolling resistance, double
    shear_displacement: Vector3 # Shear displacement stored in the contact


class DEMSolverConfig:
    def __init__(self):
        # Gravity, a global parameter
        # Denver Pilphis: in this example, we assign no gravity
        self.gravity : Vector3 = Vector3(0.0, 0.0, 0.0)
        # Time step, a global parameter
        self.dt : Real = 1e-7  # Larger dt might lead to unstable results.
        self.target_time : Real = 0.1
        # No. of steps for run, a global parameter
        self.nsteps : Integer = int(self.target_time / self.dt)
        self.saving_interval_time : Real = 0.001
        self.saving_interval_steps : Real = int(self.saving_interval_time / self.dt)

@ti.data_oriented
class DEMSolver:
    def __init__(self, config:DEMSolverConfig):
        self.config:DEMSolverConfig = config
        # particle fields
        self.gf:ti.StructField
        self.cf:ti.StructField
        self.wf:ti.StructField
        self.wcf:ti.StructField
        self.particle_id:ti.StructField
        # grid fields
        self.list_head:ti.StructField
        self.list_cur:ti.StructField
        self.list_tail:ti.StructField
        self.grain_count:ti.StructField
        self.column_sum:ti.StructField
        self.prefix_sum:ti.StructField
    
    def save(self, file_name:str, time:float):
        # P4P file for particles
        fp = open(file_name + ".p4p", encoding="UTF-8",mode='w')
        n = self.gf.shape[0]
        fp.write("TIMESTEP  PARTICLES\n")
        fp.write(f'{time} {n}\n')
        fp.write("ID  GROUP  VOLUME  MASS  PX  PY  PZ  VX  VY  VZ\n")
        for i in range(n):
            # GROUP omitted
            group : int = 0
            volume : float = self.gf[i].density/self.gf[i].density
            mass : float = self.gf[i].mass
            p = self.gf[i].position
            v = self.gf[i].velocity
            px : float = p[0]
            py : float = p[1]
            pz : float = p[2]
            vx : float = v[0]
            vy : float = v[1]
            vz : float = v[2]
            fp.write(f'{i+1} {group} {volume} {mass} {px} {py} {pz} {vx} {vy} {vz}\n')
        fp.close()

        # P4C file for contacts
        ccache: list = ["P1  P2  CX  CY  CZ  FX  FY  FZ  CONTACT_IS_BONDED\n"];
        ncontact: int = 0;
        for i in range(n):
            for j in range(n):
                if (i >= j): continue;
                if (not solver.cf[i, j].isActive): continue;
                # GROUP omitted
                p1 : int = i + 1;
                p2 : int = j + 1;
                cx : float = solver.cf[i, j].position[0];
                cy : float = solver.cf[i, j].position[1];
                cz : float = solver.cf[i, j].position[2];
                fx : float = solver.cf[i, j].force_a[0]; # Denver Pilphis: the force is not stored in present version
                fy : float = solver.cf[i, j].force_a[1]; # Denver Pilphis: the force is not stored in present version
                fz : float = solver.cf[i, j].force_a[2]; # Denver Pilphis: the force is not stored in present version
                bonded : int = solver.cf[i, j].isBonded;
                ccache.append(f'{p1} {p2} {cx} {cy} {cz} {fx} {fy} {fz} {bonded}\n')
                ncontact += 1;

        fp = open(file_name + ".p4c", encoding="UTF-8",mode='w')
        # n = self.gf.shape[0]
        fp.write("TIMESTEP  CONTACTS\n")
        fp.write(f'{time} {ncontact}\n')
        for i in range (ncontact + 1): # Include the title line
            fp.write(ccache[i]);
        fp.close()
    
    def init_particle_fields(self, file_name:str="input.p4p"):
        fp = open(file_name, encoding="UTF-8")
        line : str = fp.readline() # "TIMESTEP  PARTICLES" line
        line = fp.readline().removesuffix('\n') # "0 18112" line
        n = int(line.split(' ')[1])
        n = min(n, MaxParticleCount)
        nwall = 1

        # Initialize particles
        self.gf = Grain.field(shape=(n))
        self.cf = Contact.field(shape=(n, n))
        self.wf = Wall.field(shape = nwall)
        self.wcf = Contact.field(shape = (n,nwall))
        self.particle_id = ti.field(dtype=Integer, shape=n, name="particle_id")
        # cf.fill(Contact()) # TODO: fill all NULLs
        
        line = fp.readline() # "ID  GROUP  VOLUME  MASS  PX  PY  PZ  VX  VY  VZ" line
        # Processing particles
        for _ in range(n):
            line = fp.readline()
            if (line==''): break
            tokens:list[str] = line.split(' ')
            i : Integer = int(tokens[0]) - 1
            # GROUP omitted
            volume : Real = float(tokens[2])
            mass : Real = float(tokens[3])
            px : Real = float(tokens[4])
            py : Real = float(tokens[5])
            pz : Real = float(tokens[6])
            vx : Real = float(tokens[7])
            vy : Real = float(tokens[8])
            vz : Real = float(tokens[9])
            density : Real = mass / volume
            radius : Real = tm.pow(volume * 3.0 / 4.0 / tm.pi, 1.0 / 3.0)
            inertia : Real = 2.0 / 5.0 * mass * radius * radius
            self.gf[i].density = density
            self.gf[i].mass = mass
            self.gf[i].radius = radius
            self.gf[i].elasticModulus = 7e10 # Denver Pilphis: hard coding need to be modified in the future
            self.gf[i].poissonRatio = 0.25 # Denver Pilphis: hard coding need to be modified in the future
            self.gf[i].position = Vector3(px, py, pz)
            self.gf[i].velocity = Vector3(vx, vy, vz)
            self.gf[i].acceleration = Vector3(0.0, 0.0, 0.0)
            self.gf[i].force = Vector3(0.0, 0.0, 0.0)
            self.gf[i].quaternion = Vector4(1.0, 0.0, 0.0, 0.0)
            self.gf[i].omega = Vector3(0.0, 0.0, 0.0)
            self.gf[i].omega_dot = Vector3(0.0, 0.0, 0.0)
            self.gf[i].moment = Vector3(0.0, 0.0, 0.0)
            self.gf[i].inertia = inertia * ti.Matrix.diag(3, 1.0)
        fp.close()
        # Input wall
        # Denver Pilphis: hard coding: need to be modified in the future
        for j in range(self.wf.shape[0]):
            self.wf[j].normal = Vector3(1.0, 0.0, 0.0) # Outer normal vector of the wall, [A, B, C]
            self.wf[j].distance = 0.01 # Distance between origin and the wall, D
            # Material properties
            self.wf[j].density = 7800.0 # Density of the wall
            self.wf[j].elasticModulus = 2e11 # Elastic modulus of the wall
            self.wf[j].poissonRatio = 0.25 # Poisson's ratio of the wall
    
    def init_grid_fields(self, grid_n:int):
        # GPU grid parameters
        self.list_head = ti.field(dtype=Integer, shape=grid_n * grid_n)
        self.list_cur = ti.field(dtype=Integer, shape=grid_n * grid_n)
        self.list_tail = ti.field(dtype=Integer, shape=grid_n * grid_n)

        self.grain_count = ti.field(dtype=Integer,
                            shape=(grid_n, grid_n),
                            name="grain_count")
        self.column_sum = ti.field(dtype=Integer, shape=grid_n, name="column_sum")
        self.prefix_sum = ti.field(dtype=Integer, shape=(grid_n, grid_n), name="prefix_sum")

    @ti.kernel
    def clear_state(self):
        #alias
        gf = ti.static(self.gf)
        
        for i in gf:
            gf[i].force = Vector3(0.0, 0.0, 0.0)
            gf[i].moment = Vector3(0.0, 0.0, 0.0)
    
    @ti.kernel
    def apply_body_force(self):
        #alias

        # Gravity
        gf = ti.static(self.gf)
        g = self.config.gravity
        for i in gf:
            gf[i].force += g * gf[i].mass
            gf[i].moment += Vector3(0.0, 0.0, 0.0)

        # GLOBAL damping
        '''
        Add GLOBAL damping for EBPM, GLOBAL damping is assigned to particles
        '''
        #alias
        # gf = ti.static(self.gf)
        t_d = 0.0 # Denver Pilphis: hard coding - should be modified in the future
        for i in gf:
            damp_force = Vector3(0.0, 0.0, 0.0)
            damp_moment = Vector3(0.0, 0.0, 0.0)
            for j in ti.static(range(3)):
                damp_force[j] = -t_d * ti.abs(gf[i].force[j]) * tm.sign(gf[i].velocity[j])
                damp_moment[j] = -t_d * ti.abs(gf[i].moment[j]) * tm.sign(gf[i].omega[j])
            gf[i].force += damp_force
            gf[i].moment += damp_moment

    # NVE integrator
    @ti.kernel
    def update(self):
        #alias
        gf = ti.static(self.gf)
        dt = self.config.dt
        
        for i in gf:
            # Translational
            # Velocity Verlet integrator is adopted
            # Reference: https://www.algorithm-archive.org/contents/verlet_integration/verlet_integration.html
            gf[i].acceleration = gf[i].force / gf[i].mass
            gf[i].position += gf[i].velocity * dt + 0.5 * gf[i].acceleration * dt ** 2
            gf[i].velocity += gf[i].acceleration * dt
            # Rotational
            # Angular acceleration should be calculated via Euler's equation for rigid body
            # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
            # https://doi.org/10.1002/nme.6568
            # Eqs. (5)-(16)
            rotational_matrix = quat2RotMatrix(gf[i].quaternion)
            moment_local = rotational_matrix @ gf[i].moment
            omega_local = rotational_matrix @ gf[i].omega
            omega_dot_local = ti.Matrix.inverse(gf[i].inertia) @ (moment_local - omega_local.cross(gf[i].inertia @ omega_local))
            gf[i].omega_dot = ti.Matrix.inverse(rotational_matrix) @ omega_dot_local
            # Update particle orientation
            # Reference: Lu et al. (2015) Discrete element models for non-spherical particle systems: From theoretical developments to applications.
            # http://dx.doi.org/10.1016/j.ces.2014.11.050
            # Eq. (6)
            # Originally from Langston et al. (2004) Distinct element modelling of non-spherical frictionless particle flow.
            # https://doi.org/10.1016/j.ces.2003.10.008
            dq0 = - 0.5 * (gf[i].quaternion[1] * gf[i].omega[0] + gf[i].quaternion[2] * gf[i].omega[1] + gf[i].quaternion[3] * gf[i].omega[2])
            dq1 = + 0.5 * (gf[i].quaternion[0] * gf[i].omega[0] - gf[i].quaternion[3] * gf[i].omega[1] + gf[i].quaternion[2] * gf[i].omega[2])
            dq2 = + 0.5 * (gf[i].quaternion[3] * gf[i].omega[0] + gf[i].quaternion[0] * gf[i].omega[1] + gf[i].quaternion[1] * gf[i].omega[2])
            dq3 = + 0.5 * (-gf[i].quaternion[2] * gf[i].omega[0] + gf[i].quaternion[1] * gf[i].omega[1] + gf[i].quaternion[0] * gf[i].omega[2])
            gf[i].quaternion[0] += dq0
            gf[i].quaternion[1] += dq1
            gf[i].quaternion[2] += dq2
            gf[i].quaternion[3] += dq3
            gf[i].quaternion = tm.normalize(gf[i].quaternion)
            # Update angular velocity
            gf[i].omega += gf[i].omega_dot * dt

    @ti.func
    def evaluate(self, i : Integer, j : Integer):
        '''
        Contact resolution and evaluation
        '''
        #alias
        gf = ti.static(self.gf)
        cf = ti.static(self.cf)
        
        dt = self.config.dt
        # Contact resolution
        # Find out rotation matrix
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        a = tm.normalize(gf[j].position - gf[i].position)
        b = Vector3(1.0, 0.0, 0.0) # Local x coordinate
        v = tm.cross(a, b)
        s = tm.length(v)
        c = tm.dot(a, b)
        if (s < DoublePrecisionTolerance):
            if (c > 0.0):
                cf[i, j].rotationMatrix = ti.Matrix.diag(3, 1.0);
            else:
                cf[i, j].rotationMatrix = DEMMatrix([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]);
        else:
            vx = DEMMatrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
            cf[i, j].rotationMatrix = ti.Matrix.diag(3, 1.0) + vx + ((1.0 - c) / s**2) * vx @ vx

        cf[i, j].length = tm.length(gf[j].position - gf[i].position)
        # Contact evaluation (with contact model)
        if (cf[i, j].isBonded): # Bonded, use EBPM
            cf[i, j].position = 0.5 * (gf[i].position + gf[j].position);
            disp_a = cf[i, j].rotationMatrix @ gf[i].velocity * dt
            disp_b = cf[i, j].rotationMatrix @ gf[j].velocity * dt
            rot_a = cf[i, j].rotationMatrix @ gf[i].omega * dt
            rot_b = cf[i, j].rotationMatrix @ gf[j].omega * dt
            dispVector = EBPMForceDisplacementVector([disp_a, rot_a, disp_b, rot_b])
            r_b = cf[i, j].radius_ratio * tm.min(gf[i].radius, gf[j].radius)
            L_b = cf[i, j].length
            E_b = cf[i, j].elasticModulus
            nu = cf[i, j].poissonRatio
            I_b = (r_b ** 4) * tm.pi / 4.0
            phi = 20.0 / 3.0 * (r_b ** 2) / (L_b ** 2) * (1.0 + nu)
            A_b = tm.pi * (r_b ** 2)
            k1 = E_b * A_b / L_b
            k2 = 12.0 * E_b * I_b / (L_b ** 3) / (1.0 + phi)
            k3 = 6.0 * E_b * I_b / (L_b ** 2) / (1.0 + phi)
            k4 = E_b * I_b / L_b / (1.0 + nu)
            k5 = E_b * I_b * (4.0 + phi) / L_b / (1.0 + phi)
            k6 = E_b * I_b * (2.0 - phi) / L_b / (1.0 + phi)
            K = EBPMStiffnessMatrix([
                [  k1,   0,   0,   0,   0,   0, -k1,   0,   0,   0,   0,   0],
                [   0,  k2,   0,   0,   0,  k3,   0, -k2,   0,   0,   0,  k3],
                [   0,   0,  k2,   0, -k3,   0,   0,   0, -k2,   0, -k3,   0],
                [   0,   0,   0,  k4,   0,   0,   0,   0,   0, -k4,   0,   0],
                [   0,   0, -k3,   0,   k5,  0,   0,   0,  k3,   0,  k6,   0],
                [   0,   k3,  0,   0,   0,  k5,   0, -k3,   0,   0,   0,  k6],
                [ -k1,   0,   0,   0,   0,   0,  k1,   0,   0,   0,   0,   0],
                # K[7, 5] is WRONG in original EBPM document
                # ΔFay + ΔFby is nonzero
                # which does not satisfy the equilibrium
                # Acknowledgement to Dr. Xizhong Chen in 
                # Department of Chemical and Biological Engineering,
                # The University of Sheffield
                # Reference: Chen et al. (2022) A comparative assessment and unification of bond models in DEM simulations.
                # https://doi.org/10.1007/s10035-021-01187-2
                [   0, -k2,   0,   0,   0, -k3,   0,  k2,   0,   0,   0, -k3],
                [   0,   0, -k2,   0,  k3,   0,   0,   0,  k2,   0,  k3,   0],
                [   0,   0,   0, -k4,   0,   0,   0,   0,   0,  k4,   0,   0],
                [   0,   0, -k3,   0,  k6,   0,   0,   0,  k3,   0,  k5,   0],
                [   0,  k3,   0,   0,   0,  k6,   0, -k3,   0,   0,   0,  k5]
            ])
            forceVector = K @ dispVector
            cf[i, j].force_a += Vector3(forceVector[0], forceVector[1], forceVector[2])
            cf[i, j].moment_a += Vector3(forceVector[3], forceVector[4], forceVector[5])
            cf[i, j].force_b += Vector3(forceVector[6], forceVector[7], forceVector[8])
            cf[i, j].moment_b += Vector3(forceVector[9], forceVector[10], forceVector[11])
            
            # Check whether the bond fails
            sigma_c_a = cf[i, j].force_b[0] / A_b - r_b / I_b * tm.sqrt(cf[i, j].moment_a[1] ** 2 + cf[i, j].moment_a[2] ** 2)
            sigma_c_b = cf[i, j].force_b[0] / A_b - r_b / I_b * tm.sqrt(cf[i, j].moment_b[1] ** 2 + cf[i, j].moment_b[2] ** 2)
            sigma_c_max = -tm.min(sigma_c_a, sigma_c_b)
            sigma_t_a = sigma_c_a
            sigma_t_b = sigma_c_b
            sigma_t_max = max(sigma_t_a, sigma_t_b)
            tau_max = ti.abs(cf[i, j].moment_a[0]) * r_b / 2.0 / I_b + 4.0 / 3.0 / A_b * tm.sqrt(cf[i, j].force_a[1] ** 2 + cf[i, j].force_a[2] ** 2)
            if (sigma_c_max >= cf[i, j].compressiveStrength): # Compressive failure
                cf[i, j].isBonded = 0
                cf[i, j].isActive = 0
            elif (sigma_t_max >= cf[i, j].tensileStrength): # Tensile failure
                cf[i, j].isBonded = 0
                cf[i, j].isActive = 0
            elif (tau_max >= cf[i, j].shearStrength): # Shear failure
                cf[i, j].isBonded = 0
                cf[i, j].isActive = 0
            else: # Intact bond, need to conduct force to particles
                # Notice the inverse of signs due to Newton's third law
                # and LOCAL to GLOBAL coordinates
                gf[i].force += ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-cf[i, j].force_a)
                gf[j].force += ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-cf[i, j].force_b)
                gf[i].moment += ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-cf[i, j].moment_a)
                gf[j].moment += ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-cf[i, j].moment_b)
        else: # Non-bonded, use Hertz-Mindlin
            # Calculation relative translational and rotational displacements
            # Need to include the impact of particle rotation in contact relative translational displacement
            # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
            # https://doi.org/10.1002/nme.6568
            # Eqs. (1)-(2)
            # Implementation reference: https://github.com/CFDEMproject/LIGGGHTS-PUBLIC/blob/master/src/surface_model_default.h
            # Lines 140-189
            gap = cf[i, j].length - gf[i].radius - gf[j].radius # gap must be negative to ensure an intact contact
            delta_n = ti.abs(gap) # For parameter calculation only
            cf[i, j].position = gf[i].position + tm.normalize(gf[j].position - gf[i].position) * (gf[i].radius - delta_n)
            r_i = cf[i, j].position - gf[i].position
            r_j = cf[i, j].position - gf[j].position
            # Velocity of a point on the surface of a rigid body
            v_c_i = tm.cross(gf[i].omega, r_i) + gf[i].velocity
            v_c_j = tm.cross(gf[j].omega, r_j) + gf[j].velocity
            v_c = cf[i, j].rotationMatrix @ (v_c_j - v_c_i) # LOCAL coordinate
            # Parameter calculation
            # Reference: https://www.cfdem.com/media/DEM/docu/gran_model_hertz.html
            Y_star = 1.0 / ((1.0 - gf[i].poissonRatio ** 2) / gf[i].elasticModulus + (1.0 - gf[j].poissonRatio ** 2) / gf[j].elasticModulus)
            G_star = 1.0 / (2.0 * (2.0 - gf[i].poissonRatio) * (1.0 + gf[i].poissonRatio) / gf[i].elasticModulus + 2.0 * (2.0 - gf[j].poissonRatio) * (1.0 + gf[j].poissonRatio) / gf[j].elasticModulus)
            R_star = 1.0 / (1.0 / gf[i].radius + 1.0 / gf[j].radius)
            m_star = 1.0 / (1.0 / gf[i].mass + 1.0 / gf[j].mass)
            beta  = tm.log(cf[i, j].coefficientRestitution) / tm.sqrt(tm.log(cf[i, j].coefficientRestitution) ** 2 + tm.pi ** 2)
            S_n  = 2.0 * Y_star * tm.sqrt(R_star * delta_n)
            S_t  = 8.0 * G_star * tm.sqrt(R_star * delta_n)
            k_n  = 4.0 / 3.0 * Y_star * tm.sqrt(R_star * delta_n)
            gamma_n  = - 2.0 * beta * tm.sqrt(5.0 / 6.0 * S_n * m_star) # Check whether gamma_n >= 0
            k_t  = 8.0 * G_star * tm.sqrt(R_star * delta_n)
            gamma_t  = - 2.0 * beta * tm.sqrt(5.0 / 6.0 * S_t * m_star) # Check whether gamma_t >= 0

            # Shear displacement increments
            shear_increment = v_c * dt
            shear_increment[0] = 0.0 # Remove the normal direction
            cf[i, j].shear_displacement += shear_increment
            # Normal direction - LOCAL - the force towards particle j
            F = Vector3(0.0, 0.0, 0.0)
            # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
            # https://doi.org/10.1002/nme.6568
            # Eq. (29)
            # Be aware of signs
            F[0] = - k_n * gap - gamma_n * v_c[0]
            # Shear direction - LOCAL - the force towards particle j
            try_shear_force = - k_t * cf[i, j].shear_displacement
            if (tm.length(try_shear_force) >= cf[i, j].coefficientFriction * F[0]): # Sliding
                ratio : Real = cf[i, j].coefficientFriction * F[0] / tm.length(try_shear_force)
                F[1] = try_shear_force[1] * ratio
                F[2] = try_shear_force[2] * ratio
                cf[i, j].shear_displacement[1] = F[1] / k_t
                cf[i, j].shear_displacement[2] = F[2] / k_t
            else: # No sliding
                F[1] = try_shear_force[1] * dt - gamma_t * v_c[1]
                F[2] = try_shear_force[2] * dt - gamma_t * v_c[2]
            
            # No moment is conducted in Hertz-Mindlin model
            
            # Assigning contact force to particles
            # Notice the inverse of signs due to Newton's third law
            # and LOCAL to GLOBAL coordinates
            F_i_global = ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-F)
            F_j_global = ti.Matrix.inverse(cf[i, j].rotationMatrix) @ F
            gf[i].force += F_i_global
            gf[j].force += F_j_global
            # As the force is at contact position
            # additional moments will be assigned to particles
            # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
            # https://doi.org/10.1002/nme.6568
            # Eqs. (3)-(4)
            gf[i].moment += tm.cross(r_i, F_i_global)
            gf[j].moment += tm.cross(r_j, F_j_global)  


    @ti.func
    def resolve(self, i : Integer, j : Integer):
        '''
        Particle-particle contact detection
        '''
        
        # alias
        gf = ti.static(self.gf)
        cf = ti.static(self.cf)
        # Particle-particle contacts
        if (cf[i, j].isActive): # Existing contact
            if (cf[i, j].isBonded): # Bonded contact
                self.evaluate(i, j) # Bonded contact must exist. Go to evaluation and if bond fails, the contact state will change thereby.
            else: # Non-bonded contact, should check whether two particles are still in contact
                if (- gf[i].radius - gf[j].radius + tm.length(gf[j].position - gf[i].position) < 0): # Use PFC's gap < 0 criterion
                    self.evaluate(i, j)
                else:
                    cf[i, j].isActive = 0
        else:
            if (- gf[i].radius - gf[j].radius + tm.length(gf[j].position - gf[i].position) < 0): # Use PFC's gap < 0 criterion
                cf[i, j] = Contact( # Hertz-Mindlin model
                    isActive = 1,
                    isBonded = 0,
                    coefficientFriction = 0.3, # Denver Pilphis: hard coding need to be modified in the future
                    coefficientRestitution = 0.9, # Denver Pilphis: hard coding need to be modified in the future
                    coefficientRollingResistance = 0.01, # Denver Pilphis: hard coding need to be modified in the future
                    shear_displacement = Vector3(0.0, 0.0, 0.0)
                )
                self.evaluate(i, j) # Send to evaluation using Hertz-Mindlin contact model


    @ti.func
    def evaluate_wall(self, i : Integer, j : Integer): # i is particle, j is wall
        '''
        Particle-wall contact evaluation
        Contact model is Hertz-Mindlin
        '''
        
        # alias
        gf = ti.static(self.gf)
        cf = ti.static(self.cf)
        wf = ti.static(self.wf)
        wcf = ti.static(self.wcf)
        
        dt = self.config.dt
        # Contact resolution
        # Find out rotation matrix
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        a = wf[j].normal
        b = Vector3(1.0, 0.0, 0.0) # Local x coordinate
        v = tm.cross(a, b)
        s = tm.length(v)
        c = tm.dot(a, b)
        if (s < DoublePrecisionTolerance):
            if (c > 0.0):
                wcf[i, j].rotationMatrix = ti.Matrix.diag(3, 1.0);
            else:
                wcf[i, j].rotationMatrix = DEMMatrix([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]);
        else:
            vx = DEMMatrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
            wcf[i, j].rotationMatrix = ti.Matrix.diag(3, 1.0) + vx + ((1.0 - c) / s**2) * vx @ vx

        # Calculation relative translational and rotational displacements
        distance = tm.dot(gf[i].position, wf[j].normal) - wf[j].distance # Distance < 0 means that particle is beneath the plane
        gap  = ti.abs(distance) - gf[i].radius # gap must be negative
        delta_n = ti.abs(gap) # For parameter calculation only
        r_i = - distance * wf[j].normal / ti.abs(distance) * (ti.abs(distance) + delta_n / 2.0)
        wcf[i, j].position = gf[i].position + r_i
        # Velocity of a point on the surface of a rigid body
        v_c_i = tm.cross(gf[i].omega, r_i) + gf[i].velocity
        v_c = wcf[i, j].rotationMatrix @ (- v_c_i) # LOCAL coordinate
        # Parameter calculation
        # Reference: https://www.cfdem.com/media/DEM/docu/gran_model_hertz.html
        Y_star = 1.0 / ((1.0 - gf[i].poissonRatio ** 2) / gf[i].elasticModulus + (1.0 - wf[j].poissonRatio ** 2) / wf[j].elasticModulus)
        G_star = 1.0 / (2.0 * (2.0 - gf[i].poissonRatio) * (1.0 + gf[i].poissonRatio) / gf[i].elasticModulus + 2.0 * (2.0 - wf[j].poissonRatio) * (1.0 + wf[j].poissonRatio) / wf[j].elasticModulus)
        R_star = gf[i].radius
        m_star = gf[i].mass
        beta = tm.log(wcf[i, j].coefficientRestitution) / tm.sqrt(tm.log(wcf[i, j].coefficientRestitution) ** 2 + tm.pi ** 2)
        S_n = 2.0 * Y_star * tm.sqrt(R_star * delta_n)
        S_t  = 8.0 * G_star * tm.sqrt(R_star * delta_n)
        k_n  = 4.0 / 3.0 * Y_star * tm.sqrt(R_star * delta_n)
        gamma_n = - 2.0 * beta * tm.sqrt(5.0 / 6.0 * S_n * m_star) # Check whether gamma_n >= 0
        k_t = 8.0 * G_star * tm.sqrt(R_star * delta_n)
        gamma_t = - 2.0 * beta * tm.sqrt(5.0 / 6.0 * S_t * m_star) # Check whether gamma_t >= 0

        # Shear displacement increments
        shear_increment  = v_c * dt
        shear_increment[0] = 0.0 # Remove the normal direction
        wcf[i, j].shear_displacement += shear_increment
        # Normal direction - LOCAL - the force towards the wall
        F = Vector3(0.0, 0.0, 0.0)
        # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
        # https://doi.org/10.1002/nme.6568
        # Eq. (29)
        # Be aware of signs
        F[0] = - k_n * gap - gamma_n * v_c[0]
        # Shear direction - LOCAL - the force towards the wall
        try_shear_force = - k_t * wcf[i, j].shear_displacement
        if (tm.length(try_shear_force) >= wcf[i, j].coefficientFriction * F[0]): # Sliding
            ratio = wcf[i, j].coefficientFriction * F[0] / tm.length(try_shear_force)
            F[1] = try_shear_force[1] * ratio
            F[2] = try_shear_force[2] * ratio
            wcf[i, j].shear_displacement[1] = F[1] / k_t
            wcf[i, j].shear_displacement[2] = F[2] / k_t
        else: # No sliding
            F[1] = try_shear_force[1] * dt - gamma_t * v_c[1]
            F[2] = try_shear_force[2] * dt - gamma_t * v_c[2]
            
        # No moment is conducted in Hertz-Mindlin model
            
        # Assigning contact force to particles
        # Notice the inverse of signs due to Newton's third law
        # and LOCAL to GLOBAL coordinates
        # As the force is at contact position
        # additional moments will be assigned to particles
        F_i_global = ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-F)
        gf[i].force += F_i_global
        # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
        # https://doi.org/10.1002/nme.6568
        # Eqs. (3)-(4)
        gf[i].moment += tm.cross(r_i, F_i_global)


    @ti.kernel
    def resolve_wall(self):
        '''
        Particle-wall contact detection
        '''
        
        #alias
        gf = ti.static(self.gf)
        wf = ti.static(self.wf)
        wcf = ti.static(self.wcf)
        # Denver Pilphis: particle-wall neighboring search has not been implemented
        # and thus all particles will be contact detection with the wall
        for i, j in ti.ndrange(gf.shape[0], wf.shape[0]):
                # Particle-wall contacts
                if (wcf[i, j].isActive): # Existing contact
                    if (ti.abs(tm.dot(gf[i].position, wf[j].normal) - wf[j].distance) >= gf[i].radius): # Non-contact
                        wcf[i, j].isActive = 0
                    else: # Contact
                        self.evaluate_wall(i, j)
                else:
                    if (ti.abs(tm.dot(gf[i].position, wf[j].normal) - wf[j].distance) < gf[i].radius): # Contact
                        wcf[i, j] = Contact( # Hertz-Mindlin model
                            isActive = 1,
                            isBonded = 0,
                            coefficientFriction = 0.35, # Denver Pilphis: hard coding need to be modified in the future
                            coefficientRestitution = 0.7, # Denver Pilphis: hard coding need to be modified in the future
                            coefficientRollingResistance = 0.01, # Denver Pilphis: hard coding need to be modified in the future
                            shear_displacement = Vector3(0.0, 0.0, 0.0)
                        )
                        self.evaluate_wall(i, j)


    @ti.func 
    def bond_detect(self, i:Integer, j:Integer):
        '''
        Using CONTACT RADIUS of the spheres
        To determine whether a bond is assigned between two particles
        '''
        #alias
        gf = ti.static(self.gf)
        cf = ti.static(self.cf)
        
        contact_radius_i = gf[i].radius * 1.1 # Denver Pilphis: hard coding need to be modified in the future
        contact_radius_j = gf[j].radius * 1.1 # Denver Pilphis: hard coding need to be modified in the future
        if (tm.length(gf[j].position - gf[i].position) - contact_radius_i - contact_radius_j < 0.0):
            cf[i, j] = Contact( # Forced to bond contact
                isActive = 1,
                isBonded = 1,
                # EBPM parameters
                # Denver Pilphis: hard coding need to be modified in the future
                radius_ratio = 0.5,
                elasticModulus = 28e9,
                poissonRatio = 0.2,
                compressiveStrength = 3e8,
                tensileStrength = 6e7,
                shearStrength = 6e7,
                force_a = Vector3(0.0, 0.0, 0.0),
                moment_a = Vector3(0.0, 0.0, 0.0),
                force_b = Vector3(0.0, 0.0, 0.0),
                moment_b = Vector3(0.0, 0.0, 0.0),
            )
    

    @ti.kernel
    def bond(self):
        '''
        Similar to contact, but runs only once at the beginning
        '''
        # alias
        gf = ti.static(self.gf)
        cf = ti.static(self.cf)
        particle_id = ti.static(self.particle_id)
        
        list_head = ti.static(self.list_head)
        list_cur = ti.static(self.list_cur)
        list_tail = ti.static(self.list_cur)
        
        grain_count = ti.static(self.grain_count)
        column_sum = ti.static(self.column_sum)
        prefix_sum = ti.static(self.prefix_sum)

        n = gf.shape[0]

        # MuGdxy TODO:
        # Brute-force collision detection
        for i in range(n):
            for j in range(i + 1, n):
                self.bond_detect(i, j)


    # Neighboring search
    @ti.kernel
    def contact(self):
        '''
        Handle the collision between grains.
        '''
        # alias
        gf = ti.static(self.gf)
        cf = ti.static(self.cf)
        particle_id = ti.static(self.particle_id)
        
        list_head = ti.static(self.list_head)
        list_cur = ti.static(self.list_cur)
        list_tail = ti.static(self.list_cur)
        
        grain_count = ti.static(self.grain_count)
        column_sum = ti.static(self.column_sum)
        prefix_sum = ti.static(self.prefix_sum)
        
        n = gf.shape[0]
        # MuGdxy TODO:
        # Brute-force collision detection
        for i in range(n):
            for j in range(i + 1, n):
                self.resolve(i, j)


    def run_simulation(self):
            self.clear_state()

            
            # Particle-particle
            self.contact() # Neighboring search + Contact detection, resolution and evaluation
            
            # Particle-wall
            self.resolve_wall()

            # Particle body force
            self.apply_body_force()

            self.update() # Time integration
    
    def init_simulation(self):
        self.bond()

#======================================================================
# basic setup
#======================================================================
SAVE_FRAMES = False
window_size = 1024  # Number of pixels of the window
grid_n = 128
#=======================================================================
# entrance
#=======================================================================
if __name__ == '__main__':
    print(f"Grid size: {grid_n}x{grid_n}")
    
    config = DEMSolverConfig()
    solver = DEMSolver(config)
    solver.init_particle_fields("input.p4p")
    solver.init_grid_fields(grid_n)
    
    step = 0
    if SAVE_FRAMES:
        os.makedirs('output', exist_ok=True)

    gui = ti.GUI('Taichi DEM', (window_size, window_size))
    
    solver.init_simulation()
    while gui.running:
        # run simulation 100 times
        for _ in range(solver.config.saving_interval_steps): solver.run_simulation()
        step+=solver.config.saving_interval_steps;
        # visualize
        # TODO: consider saving when step mod saving_interval_steps == 0
        solver.save(f'output_data/{step}', solver.config.dt * step) # Denver Pilphis: problematic - for variant dt this does not work
        pos = solver.gf.position.to_numpy()
        r = solver.gf.radius.to_numpy() * window_size
        gui.circles(pos[:,(0,1)] + np.array([0.5,0.55]), radius=r)
        gui.circles(pos[:,(0,2)] + np.array([0.5,0.45]), radius=r)
        gui.line(np.array([solver.wf[0].distance, 0.3]) + 0.5, np.array([solver.wf[0].distance, -0.3]) + 0.5) # Denver Pilphis: hard coding - only one wall in this example
        if SAVE_FRAMES:
            gui.show(f'output_png/{step:06d}.png')
        else:
            gui.show()
        
