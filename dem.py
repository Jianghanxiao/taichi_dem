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
# 4. Complex DEM contact model is implemented, including a bond model (Edinburgh Bond Particle Model, EBPM)
#    and a granular contact model (Hertz-Mindlin Model);
# 5. As a bond model is implemented, nonspherical particles can be simulated with bonded agglomerates;
# 6. As a bond model is implemented, particle breakage can be simulated.

import taichi as ti
import taichi.math as tm
import os
import numpy as np
import time

#=====================================
# Type Definition
#=====================================
Real = ti.f64
Integer = ti.i32
Vector2 = ti.types.vector(2, Real)
Vector3 = ti.types.vector(3, Real)
Vector4 = ti.types.vector(4, Real)
Vector3i = ti.types.vector(3, Integer)
Vector2i = ti.types.vector(2, Integer)
Matrix3x3 = ti.types.matrix(3, 3, Real)

DEMMatrix = Matrix3x3
EBPMStiffnessMatrix = ti.types.matrix(12, 12, Real)
EBPMForceDisplacementVector = ti.types.vector(12, Real)

#=====================================
# DEM Problem Configuration
#=====================================
set_domain_min: Vector3 = Vector3(-5,-5,-5)
set_domain_max: Vector3 = Vector3(0.1,5,5)
set_init_particles: str = "Resources/cube_911_particles_impact.p4p"

class DEMSolverConfig:
    def __init__(self):
        # Gravity, a global parameter
        # Denver Pilphis: in this example, we assign no gravity
        self.gravity : Vector3 = Vector3(0.0, 0.0, 0.0)
        # Time step, a global parameter
        self.dt : Real = 1e-7  # Larger dt might lead to unstable results.
        self.target_time : Real = 0.001
        # No. of steps for run, a global parameter
        self.nsteps : Integer = int(self.target_time / self.dt)
        self.saving_interval_time : Real = 1e-5
        self.saving_interval_steps : Real = int(self.saving_interval_time / self.dt)

#=====================================
# Environmental Variables
#=====================================
DoublePrecisionTolerance: float = 1e-12; # Boundary between zeros and non-zeros
MaxParticleCount: int = 10000000;

# init taichi context
ti.init(arch=ti.gpu)

#=====================================
# Utils
#=====================================
@ti.func
def Zero3x3() -> Matrix3x3:
    return Matrix3x3([[0,0,0],[0,0,0],[0,0,0]])

def next_pow2(x:ti.i32):
    x -= 1
    x |= (x >> 1)
    x |= (x >> 2)
    x |= (x >> 4)
    x |= (x >> 8)
    x |= (x >> 16)
    return x + 1

def round32(n:ti.i32):
    if(n % 32 == 0): return n
    else: return ((n >> 5) + 1) << 5

@ti.func
def round32d(n:ti.i32):
    if(n % 32 == 0): return n
    else: return ((n >> 5) + 1) << 5

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
# Broad Phase Collision Detection
#======================================
@ti.data_oriented
class PrefixSumExecutor:
    @ti.kernel
    def serial(self, output:ti.template(), input:ti.template()):
        n = input.shape[0]
        output[0] = 0
        ti.loop_config(serialize=True)
        for i in range(1, n): 
            output[i] = output[i - 1] + input[i - 1]

    @ti.kernel
    def _down(self, d:Integer, 
                    n:Integer,
                    offset:ti.template(),
                    output:ti.template()):
            for i in range(n):
                if(i < d):
                    ai = offset*(2*i+1)-1
                    bi = offset*(2*i+2)-1
                    output[bi] += output[ai]
    @ti.kernel
    def _up(self,
            d:Integer, 
            n:Integer,
            offset:ti.template(),
            output:ti.template()):
        for i in range(n):
            if(i < d):
                ai = offset*(2*i+1)-1
                bi = offset*(2*i+2)-1
                tmp = output[ai]
                output[ai] = output[bi]
                output[bi] += tmp
    @ti.kernel
    def _copy(self, n:Integer,
              output:ti.template(),
              input:ti.template()):
        for i in range(n): output[i] = input[i]

    def parallel(self,output,input):
        n:ti.i32 = input.shape[0]
        d = n >> 1
        self._copy(n, output,input)
        offset = 1
        while(d > 0):
            self._down(d,n,offset,output)
            offset <<= 1
            d >>= 1
        
        output[n-1] = 0
        d = 1
        while(d < n):
            offset >>= 1
            self._up(d,n,offset,output)
            d <<= 1


@ti.data_oriented
class BPCD:
    '''
    Broad Phase Collision Detection
    '''
    @ti.dataclass
    class HashCell:
        offset : Integer
        count : Integer
        current : Integer

    def __init__(self, particle_count:Integer,hash_table_size:Integer, max_radius:Real, domain_min:Vector3):
        self.cell_size = max_radius * 4
        self.domain_min = domain_min
        self.hash_table = BPCD.HashCell.field(shape=hash_table_size)
        self.particle_id = ti.field(Integer, particle_count)
        self.collision_count = ti.field(Integer, shape = ())
        self.pse = PrefixSumExecutor()
    
    def create(particle_count:Integer, max_radius:Real, domain_min:Vector3, domain_max:Vector3):
        v = (domain_max - domain_min) / (4 * max_radius)
        size : ti.i32 = int(v[0] * v[1] * v[2])
        size = next_pow2(size)
        size = min(size, 1 << 20)
        return BPCD(particle_count,size,max_radius,domain_min)
    
    @ti.func
    def _count_particles(self, position:Vector3):
        ht = ti.static(self.hash_table)
        ti.atomic_add(ht[self.hash_codef(position)].count, 1)
    
    @ti.func
    def _put_particles(self, position:Vector3, id:Integer):
        ht = ti.static(self.hash_table)
        pid = ti.static(self.particle_id)
        hash_cell = self.hash_codef(position)
        loc = ti.atomic_add(ht[hash_cell].current, 1)
        offset = ht[hash_cell].offset
        pid[offset + loc] = id
    
    @ti.func
    def _fill_hash_cell(self, i:Integer):
        ht = ti.static(self.hash_table)
        ht[i].offset = 0
        ht[i].current = 0

    @ti.func
    def _clear_hash_cell(self, i:Integer):
        ht = ti.static(self.hash_table)
        ht[i].offset = 0
        ht[i].current = 0
        ht[i].count = 0

    def resolve_collision(self, 
                          positions, 
                          bounding_sphere_radius, 
                          collision_resolve_callback):
        '''
        positions: field of Vector3
        bounding_sphere_radius: field of Real
        collision_resolve_callback: func(i:ti.i32, j:ti.i32) -> None
        '''
        self._setup_collision(positions)
        self.pse.parallel(self.hash_table.offset, self.hash_table.count)
        # self.pse.serial(self.hash_table.offset, self.hash_table.count)
        self._solve_collision(positions, bounding_sphere_radius, collision_resolve_callback)

    @ti.kernel
    def _setup_collision(self, positions:ti.template()):
        ht = ti.static(self.hash_table)
        self.collision_count.fill(0)
        for i in ht: 
            self._clear_hash_cell(i)
        for i in positions: 
            self._count_particles(positions[i])
        for i in ht: 
            self._fill_hash_cell(i)
    
    @ti.kernel
    def _solve_collision(self, 
                          positions:ti.template(),
                          bounding_sphere_radius:ti.template(), 
                          collision_resolve_callback:ti.template()):
        ht = ti.static(self.hash_table)
        radius = ti.static(bounding_sphere_radius)

        for i in positions:
            self._put_particles(positions[i], i)

        for i in positions:
            o = positions[i]
            r = radius[i]
            ijk = self.cell(o)
            xyz = self.cell_center(ijk)
            Zero = Vector3i(0,0,0)
            dxyz = Zero

            for k in ti.static(range(3)):
                d = o[k] - xyz[k]
                if(d > 0): dxyz[k] = 1
                else: dxyz[k] = -1

            cells = [ ijk,
                      ijk + Vector3i(dxyz[0],   0      ,    0), 
                      ijk + Vector3i(0,         dxyz[1],    0), 
                      ijk + Vector3i(0,         0,          dxyz[2]),
                      
                      ijk + Vector3i(0,         dxyz[1],    dxyz[2]), 
                      ijk + Vector3i(dxyz[0],   0,          dxyz[2]), 
                      ijk + Vector3i(dxyz[0],   dxyz[1],    0), 
                      ijk + dxyz 
                    ]
            
            for k in ti.static(range(len(cells))):
                hash_cell = ht[self.hash_code(cells[k])]
                for idx in range(hash_cell.offset, hash_cell.offset + hash_cell.count):
                    pid = self.particle_id[idx]
                    other_o = positions[pid]
                    other_r = radius[pid]
                    if(pid > i and tm.distance(o,other_o) <= r + other_r):
                        collision_resolve_callback(i, pid)
    
    @ti.kernel
    def brute_resolve_collision(self,
                                positions:ti.template(), 
                                bounding_sphere_radius:ti.template(), 
                                collision_resolve_callback:ti.template()):
        '''
        positions: field of Vector3
        bounding_sphere_radius: field of Real
        collision_resolve_callback: func(i:ti.i32, j:ti.i32) -> None
        '''
        for i in range(positions.shape[0]):
            o = positions[i]
            r = bounding_sphere_radius[i]
            for j in range(i+1, positions.shape[0]):
                other_o = positions[j]
                other_r = bounding_sphere_radius[j]
                if(tm.distance(o,other_o) <= r + other_r):
                    collision_resolve_callback(i, j)
        
    
    # https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
    @ti.func
    def morton3d32(x:Integer,y:Integer,z:Integer) -> Integer:
        answer = 0
        x &= 0x3ff
        x = (x | x << 16) & 0x30000ff
        x = (x | x << 8) & 0x300f00f
        x = (x | x << 4) & 0x30c30c3
        x = (x | x << 2) & 0x9249249
        y &= 0x3ff
        y = (y | y << 16) & 0x30000ff
        y = (y | y << 8) & 0x300f00f
        y = (y | y << 4) & 0x30c30c3
        y = (y | y << 2) & 0x9249249
        z &= 0x3ff
        z = (z | z << 16) & 0x30000ff
        z = (z | z << 8) & 0x300f00f
        z = (z | z << 4) & 0x30c30c3
        z = (z | z << 2) & 0x9249249
        answer |= x | y << 1 | z << 2
        return answer
    
    @ti.func
    def hash_codef(self, xyz:Vector3): 
        return self.hash_code(self.cell(xyz))
    
    @ti.func
    def hash_code(self, ijk:Vector3i): 
        return BPCD.morton3d32(ijk[0],ijk[1],ijk[2]) % self.hash_table.shape[0]

    @ti.func
    def cell(self, xyz:Vector3):
        ijk = ti.floor((xyz - self.domain_min) / self.cell_size, Integer)
        return ijk

    @ti.func
    def coord(self, ijk:Vector3i):
        return ijk * self.cell_size + self.domain_min

    @ti.func
    def cell_center(self, ijk:Vector3i):
        ret = Vector3(0,0,0)
        for i in ti.static(range(3)):
            ret[i] = (ijk[i] + 0.5) * self.cell_size + self.domain_min[i]
        return ret

#======================================
# Data Class Definition
#======================================
# Particle in DEM
# Denver Pilphis: keep spherical shape first, added particle attributes to make the particle kinematically complete
@ti.dataclass
class Grain:
    # Material attributes
    # At this stage, the material attributes are written here
    ID: Integer # Record Grain ID
    density: Real  # Density, double
    mass: Real # Mass, double
    radius: Real  # Radius, double
    contactRadius: Real
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

@ti.dataclass
class IOContact:
    '''
    Contact data for IO
    '''
    i:Integer
    j:Integer
    position:Vector3
    force_a:Vector3
    isBonded:Integer
    isActive:Integer

@ti.data_oriented
class DEMSolver:
    def __init__(self, config:DEMSolverConfig):
        self.config:DEMSolverConfig = config
        # broad phase collisoin detection
        self.bpcd:BPCD
        # particle fields
        self.gf:ti.StructField
        self.cf:ti.StructField
        self.cfn:ti.SNode
        
        self.wf:ti.StructField
        self.wcf:ti.StructField
        
        self.collision_pairs:ti.StructField
        self.collision_node:ti.SNode
        
        self.cp:ti.StructField
        self.cn:ti.SNode
        # grid fields
        # self.list_head:ti.StructField
        # self.list_cur:ti.StructField
        # self.list_tail:ti.StructField
        # self.grain_count:ti.StructField
        # self.column_sum:ti.StructField
        # self.prefix_sum:ti.StructField
        # self.particle_id:ti.StructField


    @ti.kernel
    def fill_dense_cf(self, dense_cf:ti.template()):
        k = 0
        for i,j in self.cf:
            if(self.cf[i,j].isActive):
                p = ti.atomic_add(k, 1)
                dense_cf[p].position = self.cf[i, j].position
                dense_cf[p].force_a = self.cf[i, j].force_a
                dense_cf[p].isBonded = self.cf[i,j].isBonded
                dense_cf[p].isActive = self.cf[i,j].isActive
                dense_cf[p].i = i
                dense_cf[p].j = j
    
    @ti.kernel
    def dense_cf_size(self)->Integer:
        k = 0
        for i,j in self.cf: ti.atomic_add(k, 1)
        return k


    def save(self, file_name:str, time:float):
        '''
        save the solved data at <time> to file .p4p and .p4c
        '''
        # P4P file for particles
        p4p = open(file_name + ".p4p", encoding="UTF-8",mode='a')
        p4c = open(file_name + ".p4c", encoding="UTF-8",mode='a')
        self.save_single(p4p, p4c, time)
        p4p.close()
        p4c.close()



    def save_single(self, p4pfile, p4cfile, t:float):
        '''
        save the solved data at <time> to <p4pfile> and <p4cfile>
        '''
        tk1 = time.time()
        # P4P file for particles
        n = self.gf.shape[0]
        ccache = ["TIMESTEP  PARTICLES\n",
                  f"{t} {n}\n",
                  "ID  GROUP  VOLUME  MASS  PX  PY  PZ  VX  VY  VZ\n"
                  ]
        np_ID = self.gf.ID.to_numpy()
        np_mass = self.gf.mass.to_numpy()
        np_density = self.gf.density.to_numpy()
        np_position = self.gf.position.to_numpy()
        np_velocity = self.gf.velocity.to_numpy()
        for i in range(n):
            # GROUP omitted
            group : int = 0
            ID : int = np_ID[i]
            volume : float = np_mass[i]/np_density[i]
            mass : float = np_mass[i]
            px : float = np_position[i][0]
            py : float = np_position[i][1]
            pz : float = np_position[i][2]
            vx : float = np_velocity[i][0]
            vy : float = np_velocity[i][1]
            vz : float = np_velocity[i][2]
            ccache.append(f'{ID} {group} {volume} {mass} {px} {py} {pz} {vx} {vy} {vz}\n')

        for line in ccache: # Include the title line
            p4pfile.write(line);
        
        # P4C file for contacts
        ncontact = self.dense_cf_size()
        fb = ti.FieldsBuilder()
        dense_cf = IOContact.field()
        fb.dense(ti.i, ncontact).place(dense_cf)
        snode_tree = fb.finalize()  # Finalizes the FieldsBuilder and returns a SNodeTree
        self.fill_dense_cf(dense_cf)
        
        np_i = dense_cf.i.to_numpy()
        np_j = dense_cf.j.to_numpy()
        np_position = dense_cf.position.to_numpy()
        np_force_a = dense_cf.force_a.to_numpy()
        np_bonded = dense_cf.isBonded.to_numpy()
        np_active = dense_cf.isActive.to_numpy()
        ncontact = dense_cf.shape[0]
        snode_tree.destroy()
        
        ccache: list = ["TIMESTEP  CONTACTS\n",
                        f"{t} {ncontact}\n",
                        "P1  P2  CX  CY  CZ  FX  FY  FZ  CONTACT_IS_BONDED\n"];
        
        for k in range(dense_cf.shape[0]):
            # GROUP omitted
            p1 : int = np_ID[np_i[k]];
            p2 : int = np_ID[np_j[k]];
            cx : float = np_position[k][0];
            cy : float = np_position[k][1];
            cz : float = np_position[k][2];
            fx : float = np_force_a[k][0]; # Denver Pilphis: the force is not stored in present version
            fy : float = np_force_a[k][1]; # Denver Pilphis: the force is not stored in present version
            fz : float = np_force_a[k][2]; # Denver Pilphis: the force is not stored in present version
            bonded : int = np_bonded[k];
            ccache.append(f'{p1} {p2} {cx} {cy} {cz} {fx} {fy} {fz} {bonded}\n')

        for line in ccache: # Include the title line
            p4cfile.write(line);
        tk2 = time.time()
        print(f"save time cost = {tk2 - tk1}")

    # def save_single(self, p4pfile, p4cfile, time:float):
    #     '''
    #     save the solved data at <time> to <p4pfile> and <p4cfile>
    #     '''
    #     # P4P file for particles
    #     n = self.gf.shape[0]
    #     p4pfile.write("TIMESTEP  PARTICLES\n")
    #     p4pfile.write(f'{time} {n}\n')
    #     p4pfile.write("ID  GROUP  VOLUME  MASS  PX  PY  PZ  VX  VY  VZ\n")
    #     np_ID = self.gf.ID.to_numpy()
    #     np_mass = self.gf.mass.to_numpy()
    #     np_density = self.gf.density.to_numpy()
    #     np_position = self.gf.position.to_numpy()
    #     np_velocity = self.gf.velocity.to_numpy()
    #     for i in range(n):
    #         # GROUP omitted
    #         group : int = 0
    #         ID : int = np_ID[i]
    #         volume : float = np_mass[i]/np_density[i]
    #         mass : float = np_mass[i]
    #         px : float = np_position[i][0]
    #         py : float = np_position[i][1]
    #         pz : float = np_position[i][2]
    #         vx : float = np_velocity[i][0]
    #         vy : float = np_velocity[i][1]
    #         vz : float = np_velocity[i][2]
    #         p4pfile.write(f'{ID} {group} {volume} {mass} {px} {py} {pz} {vx} {vy} {vz}\n')

    #     # P4C file for contacts
    #     ccache: list = ["P1  P2  CX  CY  CZ  FX  FY  FZ  CONTACT_IS_BONDED\n"];
    #     ncontact: int = 0;
    #     np_ID = self.gf.ID.to_numpy()
        
        
    #     np_active = self.cf.isActive.to_numpy()
    #     np_position = self.cf.position.to_numpy()
    #     np_force_a = self.cf.force_a.to_numpy()
    #     np_bonded = self.cf.isBonded.to_numpy()
    #     for i in range(n):
    #         for j in range(n):
    #             if (i >= j): continue;
    #             if (not np_active[i][j]): continue;
    #             # GROUP omitted
    #             p1 : int = np_ID[i];
    #             p2 : int = np_ID[j];
    #             cx : float = np_position[i, j][0];
    #             cy : float = np_position[i, j][1];
    #             cz : float = np_position[i, j][2];
    #             fx : float = np_force_a[i, j][0]; # Denver Pilphis: the force is not stored in present version
    #             fy : float = np_force_a[i, j][1]; # Denver Pilphis: the force is not stored in present version
    #             fz : float = np_force_a[i, j][2]; # Denver Pilphis: the force is not stored in present version
    #             bonded : int = np_bonded[i, j];
    #             ccache.append(f'{p1} {p2} {cx} {cy} {cz} {fx} {fy} {fz} {bonded}\n')
    #             ncontact += 1;

    #     # n = self.gf.shape[0]
    #     p4cfile.write("TIMESTEP  CONTACTS\n")
    #     p4cfile.write(f'{time} {ncontact}\n')
    #     for i in range (ncontact + 1): # Include the title line
    #         p4cfile.write(ccache[i]);


    def init_particle_fields(self, file_name:str, domain_min:Vector3, domain_max:Vector3):
        fp = open(file_name, encoding="UTF-8")
        line : str = fp.readline() # "TIMESTEP  PARTICLES" line
        line = fp.readline().removesuffix('\n') # "0 18112" line
        n = int(line.split(' ')[1])
        n = min(n, MaxParticleCount)
        nwall = 1

        # Initialize particles
        self.gf = Grain.field(shape=(n))
        self.wf = Wall.field(shape = nwall)
        self.wcf = Contact.field(shape = (n,nwall))
        # self.particle_id = ti.field(dtype=Integer, shape=n, name="particle_id")
        # cf.fill(Contact()) # TODO: fill all NULLs
        
        line = fp.readline() # "ID  GROUP  VOLUME  MASS  PX  PY  PZ  VX  VY  VZ" line
        # Processing particles
        max_radius = 0.0
        np_ID = np.zeros(n, int)
        np_density = np.zeros(n, float)
        np_mass = np.zeros(n, float)
        np_radius = np.zeros(n, float)
        np_position = np.zeros((n,3))
        np_velocity = np.zeros((n,3))
        np_mass = np.zeros(n, float)
        np_inertia = np.zeros((n,3,3))
        for _ in range(n):
            line = fp.readline()
            if (line==''): break
            tokens:list[str] = line.split(' ')
            id : Integer = int(tokens[0])
            i = id - 1
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
            np_ID[i] = id
            # self.gf[i].density = density
            np_density[i] = density
            # self.gf[i].mass = mass
            np_mass[i] = mass
            # self.gf[i].radius = radius
            np_radius[i] = radius
            if(radius > max_radius): max_radius = radius
            # self.gf[i].position = Vector3(px, py, pz)
            np_position[i] = Vector3(px,py,pz)
            # self.gf[i].velocity = Vector3(vx, vy, vz)
            np_velocity[i] = Vector3(vx,vy,vz)
            # self.gf[i].inertia = inertia * ti.Matrix.diag(3, 1.0)
            np_inertia[i] = inertia * ti.Matrix.diag(3, 1.0)
        fp.close()
        self.gf.ID.from_numpy(np_ID)
        self.gf.density.from_numpy(np_density)
        self.gf.mass.from_numpy(np_mass)
        self.gf.radius.from_numpy(np_radius)
        self.gf.contactRadius.from_numpy(np_radius * 1.1)
        self.gf.position.from_numpy(np_position)
        self.gf.velocity.from_numpy(np_velocity)
        self.gf.inertia.from_numpy(np_inertia)
        
        self.gf.acceleration.fill(Vector3(0.0, 0.0, 0.0))
        self.gf.force.fill(Vector3(0.0, 0.0, 0.0))
        self.gf.quaternion.fill(Vector4(1.0, 0.0, 0.0, 0.0))
        self.gf.omega.fill((0.0, 0.0, 0.0))
        self.gf.omega_dot.fill(Vector3(0.0, 0.0, 0.0))
        self.gf.moment.fill(Vector3(0.0, 0.0, 0.0))
        self.gf.elasticModulus.fill(7e10) # Denver Pilphis: hard coding need to be modified in the future
        self.gf.poissonRatio.fill(0.25) # Denver Pilphis: hard coding need to be modified in the future
        # Input wall
        # Denver Pilphis: hard coding: need to be modified in the future
        for j in range(self.wf.shape[0]):
            self.wf[j].normal = Vector3(1.0, 0.0, 0.0) # Outer normal vector of the wall, [A, B, C]
            self.wf[j].distance = 0.01 # Distance between origin and the wall, D
            # Material properties
            self.wf[j].density = 7800.0 # Density of the wall
            self.wf[j].elasticModulus = 2e11 # Elastic modulus of the wall
            self.wf[j].poissonRatio = 0.25 # Poisson's ratio of the wall
        
        self.bpcd = BPCD.create(n, max_radius * 1.1, domain_min, domain_max)
        
        u1 = ti.types.quant.int(1, False)
        self.cp = ti.field(u1)
        self.cn = ti.root.dense(ti.i, round32(n * n)//32).quant_array(ti.i, dimensions=32, max_num_bits=32).place(self.cp)
        
        self.cf = Contact.field()
        self.cfn = ti.root.pointer(ti.ij, (n,n)).place(self.cf)
        
        # self.collision_pairs = ti.field(ti.i8)
        # self.collision_node = ti.root.pointer(ti.ij,(n,n))
        # self.collision_node.place(self.collision_pairs)


    @ti.func
    def set_collision_bit(self, i:ti.i32, j:ti.i32):
        n = self.gf.shape[0]
        idx = i * n + j
        self.cp[idx] = 1


    @ti.func
    def get_collision_bit(self, i:ti.i32, j:ti.i32):
        n = self.gf.shape[0]
        idx = i * n + j
        return self.cp[idx]


    # def init_grid_fields(self, grid_n:int):
    #     # GPU grid parameters
    #     self.list_head = ti.field(dtype=Integer, shape=grid_n * grid_n)
    #     self.list_cur = ti.field(dtype=Integer, shape=grid_n * grid_n)
    #     self.list_tail = ti.field(dtype=Integer, shape=grid_n * grid_n)

    #     self.grain_count = ti.field(dtype=Integer,
    #                         shape=(grid_n, grid_n),
    #                         name="grain_count")
    #     self.column_sum = ti.field(dtype=Integer, shape=grid_n, name="column_sum")
    #     self.prefix_sum = ti.field(dtype=Integer, shape=(grid_n, grid_n), name="prefix_sum")


    @ti.kernel
    def clear_state(self):
        #alias
        gf = ti.static(self.gf)
        
        for i in gf:
            gf[i].force = Vector3(0.0, 0.0, 0.0)
            gf[i].moment = Vector3(0.0, 0.0, 0.0)


    @ti.kernel
    def apply_body_force(self):
        # alias

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
        # alias
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

        # kinematic_energy : Real = 0.0;
        
        for i in gf:
            # Translational
            # Velocity Verlet integrator is adopted
            # Reference: https://www.algorithm-archive.org/contents/verlet_integration/verlet_integration.html
            gf[i].acceleration = gf[i].force / gf[i].mass
            # print(f"{gf[i].ID}.force = {gf[i].force[0]}")
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

            # ti.atomic_add(kinematic_energy, gf[i].mass / 2.0 * tm.dot(gf[i].velocity, gf[i].velocity));
        
        # print(f"{kinematic_energy}");


    @ti.kernel
    def resolve_collision(self):
        # for i,j in self.collision_pairs:
        #     self.resolve(i,j)
        size = self.gf.shape[0]
        for i,j in ti.ndrange(size, size):
            if(self.get_collision_bit(i,j)): self.resolve(i, j)
    
    
    @ti.func
    def resolve(self, i : Integer, j : Integer):
        '''
        Particle-particle contact detection
        '''
        # alias
        gf = ti.static(self.gf)
        cf = ti.static(self.cf)
        
        eval = False
        # Particle-particle contacts
        
        if (ti.is_active(self.cfn, [i,j]) and self.cf[i,j].isActive): # Existing contact
            if (cf[i, j].isBonded): # Bonded contact
                eval = True # Bonded contact must exist. Go to evaluation and if bond fails, the contact state will change thereby.
            else: # Non-bonded contact, should check whether two particles are still in contact
                if (- gf[i].radius - gf[j].radius + tm.length(gf[j].position - gf[i].position) < 0): # Use PFC's gap < 0 criterion
                    eval = True
                else:
                    self.cf[i,j].isActive = 0
                    ti.deactivate(self.cfn, [i,j])
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
                eval = True # Send to evaluation using Hertz-Mindlin contact model
        
        if(eval):
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
                
                # For debug only
                # Check equilibrium
                if (tm.length(cf[i, j].force_a + cf[i, j].force_b) > DoublePrecisionTolerance):
                    print("Equilibrium error.")

                # Check whether the bond fails
                sigma_c_a = cf[i, j].force_b[0] / A_b - r_b / I_b * tm.sqrt(cf[i, j].moment_a[1] ** 2 + cf[i, j].moment_a[2] ** 2)
                sigma_c_b = cf[i, j].force_b[0] / A_b - r_b / I_b * tm.sqrt(cf[i, j].moment_b[1] ** 2 + cf[i, j].moment_b[2] ** 2)
                sigma_c_max = -tm.min(sigma_c_a, sigma_c_b)
                sigma_t_a = sigma_c_a
                sigma_t_b = sigma_c_b
                sigma_t_max = tm.max(sigma_t_a, sigma_t_b)
                tau_max = ti.abs(cf[i, j].moment_a[0]) * r_b / 2.0 / I_b + 4.0 / 3.0 / A_b * tm.sqrt(cf[i, j].force_a[1] ** 2 + cf[i, j].force_a[2] ** 2)
                if (sigma_c_max >= cf[i, j].compressiveStrength): # Compressive failure
                    cf[i, j].isBonded = 0
                    cf[i, j].isActive = 0
                    ti.deactivate(self.cfn, [i,j])
                    # print(f"Bond compressive failure at: {i}, {j}");
                elif (sigma_t_max >= cf[i, j].tensileStrength): # Tensile failure
                    cf[i, j].isBonded = 0
                    cf[i, j].isActive = 0
                    ti.deactivate(self.cfn, [i,j])
                    # print(f"Bond tensile failure at: {i}, {j}\n");
                elif (tau_max >= cf[i, j].shearStrength): # Shear failure
                    cf[i, j].isBonded = 0
                    cf[i, j].isActive = 0
                    ti.deactivate(self.cfn, [i,j])
                    # print(f"Bond shear failure at: {i}, {j}\n");
                else: # Intact bond, need to conduct force to particles
                    # Notice the inverse of signs due to Newton's third law
                    # and LOCAL to GLOBAL coordinates
                    ti.atomic_add(gf[i].force, ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-cf[i, j].force_a))
                    ti.atomic_add(gf[j].force, ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-cf[i, j].force_b))
                    ti.atomic_add(gf[i].moment, ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-cf[i, j].moment_a))
                    ti.atomic_add(gf[j].moment, ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-cf[i, j].moment_b))
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

                # For debug only
                # if (delta_n > 0.05 * ti.min(gf[i].radius, gf[j].radius)):
                #     print("WARNING: Overlap particle-particle exceeds 0.05");
                
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
                    F[1] = try_shear_force[1] - gamma_t * v_c[1]
                    F[2] = try_shear_force[2] - gamma_t * v_c[2]
                
                # No moment is conducted in Hertz-Mindlin model
                
                # For P4C output
                cf[i, j].force_a = F;
                cf[i, j].force_b = -F;
                # Assigning contact force to particles
                # Notice the inverse of signs due to Newton's third law
                # and LOCAL to GLOBAL coordinates
                F_i_global = ti.Matrix.inverse(cf[i, j].rotationMatrix) @ (-F)
                F_j_global = ti.Matrix.inverse(cf[i, j].rotationMatrix) @ F
                ti.atomic_add(gf[i].force, F_i_global)
                ti.atomic_add(gf[j].force, F_j_global)
                # As the force is at contact position
                # additional moments will be assigned to particles
                # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
                # https://doi.org/10.1002/nme.6568
                # Eqs. (3)-(4)
                ti.atomic_add(gf[i].moment, tm.cross(r_i, F_i_global))
                ti.atomic_add(gf[j].moment, tm.cross(r_j, F_j_global))


    @ti.func
    def on_collision(self, i : Integer, j : Integer):
        # self.collision_pairs[i,j] = 1
        self.set_collision_bit(i,j)


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

        # For debug only
        # if (delta_n > 0.05 * gf[i].radius):
        #     print("WARNING: Overlap particle-wall exceeds 0.05");

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
            F[1] = try_shear_force[1] - gamma_t * v_c[1]
            F[2] = try_shear_force[2] - gamma_t * v_c[2]
            
        # No moment is conducted in Hertz-Mindlin model
        
        # For P4C output
        wcf[i, j].force_a = F;
        wcf[i, j].force_b = -F;
        # Assigning contact force to particles
        # Notice the inverse of signs due to Newton's third law
        # and LOCAL to GLOBAL coordinates
        # As the force is at contact position
        # additional moments will be assigned to particles
        F_i_global = ti.Matrix.inverse(wcf[i, j].rotationMatrix) @ (-F)
        
        ti.atomic_add(gf[i].force, F_i_global)
        # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
        # https://doi.org/10.1002/nme.6568
        # Eqs. (3)-(4)
        ti.atomic_add(gf[i].moment, tm.cross(r_i, F_i_global))


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


    def bond(self):
        '''
        Similar to contact, but runs only once at the beginning
        '''
        # In example 911, brute detection has better efficiency
        self.bpcd.resolve_collision(self.gf.position, self.gf.contactRadius, self.bond_detect)
        # self.bpcd.brute_resolve_collision(self.gf.position, self.gf.contactRadius, self.bond_detect)


    # Neighboring search
    def contact(self):
        '''
        Handle the collision between grains.
        '''
        # self.collision_node.deactivate_all()
        self.cp.fill(0)
        # In example 911, brute detection has better efficiency
        self.bpcd.resolve_collision(self.gf.position, self.gf.contactRadius, self.on_collision)
        # self.bpcd.brute_resolve_collision(self.gf.position, self.gf.contactRadius, self.on_collision)
        self.resolve_collision()


    def run_simulation(self):
        self.clear_state()
        # Particle-particle 
        # Neighboring search + Contact detection, resolution and evaluation
        self.contact()
        # Particle-wall
        self.resolve_wall()
        # Particle body force
        self.apply_body_force() 
        # Time integration
        self.update()


    def init_simulation(self):
        self.bond()


#======================================================================
# basic setup
#======================================================================
SAVE_FRAMES = True
VISUALIZE = False
window_size = 1024  # Number of pixels of the window
#=======================================================================
# entrance
#=======================================================================
if __name__ == '__main__':
    config = DEMSolverConfig()
    solver = DEMSolver(config)
    domain_min = set_domain_min
    domain_max = set_domain_max
    solver.init_particle_fields(set_init_particles,domain_min,domain_max)
    print(f"hash table size = {solver.bpcd.hash_table.shape[0]}")
    
    step = 0
    elapsed_time = 0.0
    solver.init_simulation()
    if VISUALIZE:
        if SAVE_FRAMES: os.makedirs('output', exist_ok=True)
        gui = ti.GUI('Taichi DEM', (window_size, window_size))
        while gui.running and step < config.nsteps:
            for _ in range(100):
                step+=1 
                solver.run_simulation()
            pos = solver.gf.position.to_numpy()
            r = solver.gf.radius.to_numpy() * window_size
            gui.circles(pos[:,(0,1)] + np.array([0.5,0.55]), radius=r)
            # gui.circles(pos[:,(0,2)] + np.array([0.5,0.45]), radius=r)
            gui.line(np.array([solver.wf[0].distance, 0.3]) + 0.5, np.array([solver.wf[0].distance, -0.3]) + 0.5) # Denver Pilphis: hard coding - only one wall in this example
            if(SAVE_FRAMES):
                gui.show(f'output/{step:07d}.png')
            else:
                gui.show()
    else: # offline
        solver.save('output', 0)
        #p4p = open('output',encoding="UTF-8",mode='w')
        #p4c = open('output',encoding="UTF-8",mode='w')
        #solver.save_single(p4p,p4c,solver.config.dt * step)
        while step < config.nsteps:
            for _ in range(config.saving_interval_steps): 
                step += 1
                elapsed_time += config.dt
                solver.run_simulation()
            #solver.save_single(p4p,p4c,solver.config.dt * step)
            solver.save('output', elapsed_time)
            print(f"solved steps: {step}")
        #p4p.close()
        #p4c.close()