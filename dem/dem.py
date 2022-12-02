# Engineering quantitative DEM simulation using Taichi DEM
# Taichi Hackathon 2022 Edition
# 
# Authors:
# Denver Pilphis (Di Peng) - DEM theory and implementation, particle-wall bonding
# MuGdxy (Xinyu Lu) - HPC implementation, time sequence implementation


from math import pi
import taichi as ti
import taichi.math as tm
import os
import numpy as np
import time
# define the data type
from TypeDefine import *
# Data Class
from DataClass import *
# dem config file
from DemConfig import *
# util function
from Utils import *
# statistic tool
from DEMSolverStatistics import *
# broad phase collision detection
from BroadPhaseCollisionDetection import *
# time sequence
from TimeSequence import *

# Init taichi context
# Device memory size is recommended to be 75% of your GPU VRAM
ti.init(arch=ti.gpu, debug=False)

@ti.data_oriented
class DEMSolver:
    def __init__(self, config:DEMSolverConfig, statistics:DEMSolverStatistics = None, workload = WorkloadType.Auto):
        self.config:DEMSolverConfig = config
        # Broad phase collisoin detection
        self.bpcd:BPCD
        # Material mapping
        self.mf:ti.StructField # Material types, n*1 field: 0 - particles 1 - walls
        self.surf:ti.StructField # Surface types, n*n field: [0, 0] - particle-particle [0, 1] == [1, 0] - particle-wall [1, 1] - wall-wall (insensible)
        # Particle fields
        self.gf:ti.StructField
        self.grainID2Index = None
        
        self.cf:ti.StructField # neighbors for every particle
        self.cfn:ti.StructField # neighbor counter for every particle
        
        self.wf:ti.StructField
        self.wcf:ti.StructField

        self.cp:ti.StructField # collision pairs
        self.cn:ti.SNode # collision pair node
        
        self.statistics:DEMSolverStatistics = statistics
        self.workload_type = workload
        # Taichi Hackathon 2022 append
        self.timeSquence = TimeSequence(set_time_sequnce_folder)
        # Enclose step count and time elapsed to handle the time sequence
        # Step count in DEM
        self.frame = 0
        # Time elapsed in DEM
        self.elapsedTime = 0.0

    def save(self, file_name:str, time:float):
        '''
        save the solved data at <time> to file .p4p and .p4c
        '''
        # P4P file for particles
        p4p = open(file_name + ".p4p", encoding="UTF-8", mode = 'w')
        p4c = open(file_name + ".p4c", encoding="UTF-8", mode = 'w')
        self.save_single(p4p, p4c, time)
        p4p.close()
        p4c.close()


    def save_single(self, p4pfile, p4cfile, t:float):
        '''
        save the solved data at <time> to <p4pfile> and <p4cfile>
        usage:
            p4p = open('output.p4p',encoding="UTF-8",mode='w')
            p4c = open('output.p4c',encoding="UTF-8",mode='w')
            while(True):
                solver.save_single(p4p, p4c, elapsed_time)
        '''
        tk1 = time.time()
        # P4P file for particles
        n = self.gf.shape[0]
        ccache = ["TIMESTEP  PARTICLES\n",
                  f"{t} {n}\n",
                  "ID  GROUP  VOLUME  MASS  PX  PY  PZ  VX  VY  VZ\n"
                  ]
        np_ID = self.gf.ID.to_numpy()
        np_materialType = self.gf.materialType.to_numpy()
        np_radius = self.gf.radius.to_numpy()
        #np_mass = self.gf.mass.to_numpy()
        #np_density = self.gf.density.to_numpy()
        np_position = self.gf.position.to_numpy()
        np_velocity = self.gf.velocity.to_numpy()

        np_density = self.mf.density.to_numpy()
        for i in range(n):
            # GROUP omitted
            group : int = 0
            ID : int = np_ID[i]
            volume : float = 4.0 / 3.0 * pi * np_radius[i] ** 3
            mass : float = volume * np_density[np_materialType[i]]
            px : float = np_position[i][0]
            py : float = np_position[i][1]
            pz : float = np_position[i][2]
            vx : float = np_velocity[i][0]
            vy : float = np_velocity[i][1]
            vz : float = np_velocity[i][2]
            ccache.append(f'{ID} {group} {volume} {mass} {px} {py} {pz} {vx} {vy} {vz}\n')

        for line in ccache: # Include the title line
            p4pfile.write(line)
        
        # P4C file for contacts

        np_i = self.cf.i.to_numpy()
        np_j = self.cf.j.to_numpy()
        np_position = self.cf.position.to_numpy()
        np_force_a = self.cf.force_a.to_numpy()
        np_bonded = self.cf.isBonded.to_numpy()
        np_active = self.cf.isActive.to_numpy()
        ncontact = 0
        
        ccache: list = []
        
        for k in range(self.cf.shape[0]):
            # GROUP omitted
            if(np_active[k]):
                p1 : int = np_ID[np_i[k]]
                p2 : int = np_ID[np_j[k]]
                cx : float = np_position[k][0]
                cy : float = np_position[k][1]
                cz : float = np_position[k][2]
                fx : float = np_force_a[k][0]
                fy : float = np_force_a[k][1]
                fz : float = np_force_a[k][2]
                bonded : int = np_bonded[k]
                ncontact+=1
                ccache.append(f'{p1} {p2} {cx} {cy} {cz} {fx} {fy} {fz} {bonded}\n')
        head: list = ["TIMESTEP  CONTACTS\n",
                        f"{t} {ncontact}\n",
                        "P1  P2  CX  CY  CZ  FX  FY  FZ  CONTACT_IS_BONDED\n"]
        for line in head:
            p4cfile.write(line)
        for line in ccache: # Include the title line
            p4cfile.write(line)
        tk2 = time.time()
        print(f"save time cost = {tk2 - tk1}")

    
    def check_workload(self, n):
        if(self.workload_type == WorkloadType.Auto):
            if(n < 1000): self.workload_type = WorkloadType.Light
            elif(n < 20000): self.workload_type = WorkloadType.Midium
            else: self.workload_type = WorkloadType.Heavy


    def init_particle_fields(self, file_name:str, domain_min:Vector3, domain_max:Vector3):
        fp = open(file_name, encoding="UTF-8")
        line : str = fp.readline() # "TIMESTEP  PARTICLES" line
        line = fp.readline().removesuffix('\n') # "0 18112" line
        n = int(line.split(' ')[1])
        n = min(n, MaxParticleCount)
        self.check_workload(n)
        # Denver Pilphis: hard-coding
        nwall = 1

        # Initialize particles
        self.gf = Grain.field(shape=(n))
        self.wf = Wall.field(shape = nwall)
        self.wcf = Contact.field(shape = (n,nwall))
        self.grainID2Index = ti.field(dtype=ti.i32, shape = n)
        
        # Denver Pilphis: hard-coding
        self.mf = Material.field(shape = 2)
        self.surf = Surface.field(shape = (2,2))
        
        line = fp.readline() # "ID  GROUP  VOLUME  MASS  PX  PY  PZ  VX  VY  VZ" line
        # Processing particles
        max_radius = 0.0
        np_ID = np.zeros(n, int)
        np_grainID2Index = np.zeros(n, int)
        np_density = np.zeros(n, float)
        np_mass = np.zeros(n, float)
        np_radius = np.zeros(n, float)
        np_position = np.zeros((n,3))
        np_velocity = np.zeros((n,3))
        np_mass = np.zeros(n, float)
        np_inertia = np.zeros((n,3,3))

        # Extract density, hard coding
        material_density: float = 0.0
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
            # Hard coding
            material_density = density
            radius : Real = tm.pow(volume * 3.0 / 4.0 / tm.pi, 1.0 / 3.0)
            inertia : Real = 2.0 / 5.0 * mass * radius * radius
            np_ID[i] = id
            np_grainID2Index[id-1] = i
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
        self.gf.materialType.fill(0) # Denver Pilphis: hard coding
        self.gf.radius.from_numpy(np_radius)
        self.gf.contactRadius.from_numpy(np_radius * set_particle_contact_radius_multiplier)
        self.gf.position.from_numpy(np_position)
        self.gf.velocity.from_numpy(np_velocity)
        self.grainID2Index.from_numpy(np_grainID2Index)
        self.gf.acceleration.fill(Vector3(0.0, 0.0, 0.0))
        self.gf.force.fill(Vector3(0.0, 0.0, 0.0))        
        self.gf.quaternion.fill(Vector4(1.0, 0.0, 0.0, 0.0))
        self.gf.omega.fill((0.0, 0.0, 0.0))
        self.gf.omega_dot.fill(Vector3(0.0, 0.0, 0.0))
        self.gf.inertia.from_numpy(np_inertia)
        self.gf.moment.fill(Vector3(0.0, 0.0, 0.0))
        # Input wall
        # Denver Pilphis: hard coding - need to be modified in the future
        for j in range(self.wf.shape[0]):
            self.wf[j].normal = set_wall_normal # Outer normal vector of the wall, [A, B, C]
            self.wf[j].distance = set_wall_distance # Distance between origin and the wall, D
            # Material property
            self.wf[j].materialType = 1 # Hard coding

        # Material
        # Particle
        self.mf[0].density = material_density
        self.mf[0].elasticModulus = set_particle_elastic_modulus
        self.mf[0].poissonRatio = set_particle_poisson_ratio
        # Wall
        self.mf[1].density = set_wall_density
        self.mf[1].elasticModulus = set_wall_elastic_modulus
        self.mf[1].poissionRatio = set_wall_poisson_ratio

        # Surface
        # Particle-particle, including EBPM and Hertz-Mindlin model parameters
        # HM
        self.surf[0, 0].coefficientFriction = set_pp_coefficient_friction
        self.surf[0, 0].coefficientRestitution = set_pp_coefficient_restitution
        self.surf[0, 0].coefficientRollingResistance = set_pp_coefficient_rolling_resistance
        # EBPM
        self.surf[0, 0].radius_ratio = set_bond_radius_ratio
        self.surf[0, 0].elasticModulus = set_bond_elastic_modulus
        self.surf[0, 0].poissonRatio = set_bond_poission_ratio
        self.surf[0, 0].compressiveStrength = set_bond_compressive_strength
        self.surf[0, 0].tensileStrength = set_bond_tensile_strength
        self.surf[0, 0].shearStrength = set_bond_shear_strength
        # Particle-wall, only Hertz-Mindlin model parameters
        self.surf[0, 1].coefficientFriction = set_pw_coefficient_friction;
        self.surf[0, 1].coefficientRestitution = set_pw_coefficient_restitution;
        self.surf[0, 1].coefficientRollingResistance = set_pw_coefficient_rolling_resistance;
        
        # Symmetric matrix for surf
        self.surf[1, 0] = self.surf[0, 1]

        # surf[1, 1] is insensible
        if(self.workload_type ==  WorkloadType.Heavy):
            r = cal_neighbor_search_radius(max_radius)
            self.bpcd = BPCD.create(n, r, domain_min, domain_max, BPCD.ExplicitCollisionPair)
            self.bpcd.statistics = self.statistics
        
        self.cf = Contact.field(shape=set_max_coordinate_number * n)
        self.cfn = ti.field(Integer, shape=n)


# >>> contact field utils
    @ti.func
    def append_contact_offset(self, i):
        ret = -1
        offset = ti.atomic_add(self.cfn[i], 1)
        # print(f'i={i}, offset={offset}')
        if(offset < set_max_coordinate_number):
            ret =  i * set_max_coordinate_number + offset
        return ret

    @ti.func
    def search_active_contact_offset(self, i, j):
        ret = -1
        for offset in range(self.cfn[i]):
            if (self.cf[i * set_max_coordinate_number + offset].j == j 
                and self.cf[i * set_max_coordinate_number + offset].isActive):
                    ret = i * set_max_coordinate_number + offset
                    break
        return ret
    
    @ti.func 
    def remove_inactive_contact(self, i):
        active_count = 0
        for j in range(self.cfn[i]):
            if(self.cf[i * set_max_coordinate_number + j].isActive): active_count+=1
        offset = 0
        for j in range(self.cfn[i]):
            if(self.cf[i * set_max_coordinate_number + j].isActive):
                self.cf[i * set_max_coordinate_number + offset] = self.cf[i * set_max_coordinate_number + j]
                offset += 1
                if(offset >= active_count): break
        for j in range(active_count, self.cfn[i]):
            self.cf[i * set_max_coordinate_number + j].isActive = False
        self.cfn[i] = active_count
    
# <<< contact field utils


# >>> collision pair list utils
    @ti.kernel
    def cp_list_resolve_collision(self, cp_range:ti.template(), cp_list:ti.template()):
        for i in cp_range:
            for k in range(cp_range[i].count):
                j = cp_list[cp_range[i].offset + k]
                self.resolve_unbonded(i, j)
# <<< collision pair list utils
    @ti.kernel
    def clear_state(self):
        #alias
        gf = ti.static(self.gf)
        
        for i in gf:
            gf[i].force = Vector3(0.0, 0.0, 0.0)
            gf[i].moment = Vector3(0.0, 0.0, 0.0)


    @ti.kernel
    def late_clear_state(self):
        gf = ti.static(self.gf)
        # remove inactive contact and do compress
        for i in gf:
            self.remove_inactive_contact(i)


    @ti.kernel
    def apply_body_force(self):
        # alias

        # Gravity
        gf = ti.static(self.gf)
        mf = ti.static(self.mf)
        g = self.config.gravity
        for i in gf:
            type_i = gf[i].materialType
            gf[i].force += mf[type_i].density * 4.0 / 3.0 * tm.pi * gf[i].radius ** 3 * g
            gf[i].moment += Vector3(0.0, 0.0, 0.0)

        # GLOBAL damping
        '''
        Add GLOBAL damping for EBPM, GLOBAL damping is assigned to particles
        '''
        # alias
        # gf = ti.static(self.gf)
        t_d = config.global_damping
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
        mf = ti.static(self.mf)
        dt = self.config.dt

        # kinematic_energy : Real = 0.0
        
        for i in gf:
            # Translational
            # Velocity Verlet integrator is adopted
            # Reference: https://www.algorithm-archive.org/contents/verlet_integration/verlet_integration.html
            type_i = gf[i].materialType
            gf[i].acceleration = gf[i].force / (mf[type_i].density * 4.0 / 3.0 * tm.pi * gf[i].radius ** 3)
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
            # ti.atomic_add(kinematic_energy, gf[i].mass / 2.0 * tm.dot(gf[i].velocity, gf[i].velocity))
        # print(f"{kinematic_energy}")


    @ti.kernel
    def resolve_bonded(self):
        for i in self.cfn:
            for k in range(self.cfn[i]):
                offset = i * set_max_coordinate_number + k
                self.evaluate_bonded(offset)


    @ti.func
    def evaluate_bonded(self, offset: Integer):
        # Bonded, use EBPM
        dt = self.config.dt
        gf = ti.static(self.gf)
        cf = ti.static(self.cf)
        mf = ti.static(self.mf)
        surf = ti.static(self.surf)
        
        i = cf[offset].i
        j = cf[offset].j
        
        length = tm.length(gf[j].position - gf[i].position)
        # Contact resolution
        # Find out rotation matrix
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        a = tm.normalize(gf[j].position - gf[i].position)
        b = Vector3(1.0, 0.0, 0.0) # Local x coordinate
        v = tm.cross(a, b)
        s = tm.length(v)
        c = tm.dot(a, b)
        rotationMatrix = Zero3x3()
        if (s < DoublePrecisionTolerance):
            if (c > 0.0):
                rotationMatrix = ti.Matrix.diag(3, 1.0)
            else:
                rotationMatrix = Matrix3x3([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        else:
            vx = Matrix3x3([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
            rotationMatrix = ti.Matrix.diag(3, 1.0) + vx + ((1.0 - c) / s**2) * vx @ vx
        
        if (cf[offset].isActive and cf[offset].isBonded):
            cf[offset].position = 0.5 * (gf[i].position + gf[j].position)
            disp_a = rotationMatrix @ gf[i].velocity * dt
            disp_b = rotationMatrix @ gf[j].velocity * dt
            rot_a = rotationMatrix @ gf[i].omega * dt
            rot_b = rotationMatrix @ gf[j].omega * dt

            type_i: Integer = gf[i].materialType
            type_j: Integer = gf[j].materialType
            r_b = surf[type_i, type_j].radius_ratio * tm.min(gf[i].radius, gf[j].radius)
            L_b = length
            E_b = surf[type_i, type_j].elasticModulus
            nu = surf[type_i, type_j].poissonRatio
            I_b = (r_b ** 4) * tm.pi / 4.0
            phi = 20.0 / 3.0 * (r_b ** 2) / (L_b ** 2) * (1.0 + nu)
            A_b = tm.pi * (r_b ** 2)
            k1 = E_b * A_b / L_b
            k2 = 12.0 * E_b * I_b / (L_b ** 3) / (1.0 + phi)
            k3 = 6.0 * E_b * I_b / (L_b ** 2) / (1.0 + phi)
            k4 = E_b * I_b / L_b / (1.0 + nu)
            k5 = E_b * I_b * (4.0 + phi) / L_b / (1.0 + phi)
            k6 = E_b * I_b * (2.0 - phi) / L_b / (1.0 + phi)
            inc_force_a = Vector3(
                k1 * (disp_a[0] - disp_b[0]),
                k2 * (disp_a[1] - disp_b[1]) + k3 * (rot_a[2] + rot_b[2]),
                k2 * (disp_a[2] - disp_b[2]) - k3 * (rot_a[1] + rot_b[1])
            )
            inc_moment_a = Vector3(
                k4 * (rot_a[0] - rot_b[0]),
                k3 * (disp_b[2] - disp_a[2]) + k5 * rot_a[1] + k6 * rot_b[1],
                k3 * (disp_a[1] - disp_b[1]) + k5 * rot_a[2] + k6 * rot_b[2]
            )
            # No need to calculate inc_force_b as inc_force_b == - inc_force_a and force_b == - force_a
            inc_moment_b = Vector3(
                k4 * (rot_b[0] - rot_a[0]),
                k3 * (disp_b[2] - disp_a[2]) + k6 * rot_a[1] + k5 * rot_b[1],
                k3 * (disp_a[1] - disp_b[1]) + k6 * rot_a[2] + k5 * rot_b[2]
            )

            # forceVector = K @ dispVector
            cf[offset].force_a += inc_force_a
            # cf[offset].force_a += Vector3(forceVector[0], forceVector[1], forceVector[2])
            cf[offset].moment_a += inc_moment_a
            # cf[offset].moment_a += Vector3(forceVector[3], forceVector[4], forceVector[5])
            force_b = - cf[offset].force_a
            # cf[offset].force_b += Vector3(forceVector[6], forceVector[7], forceVector[8])
            cf[offset].moment_b += inc_moment_b
            # cf[offset].moment_b += Vector3(forceVector[9], forceVector[10], forceVector[11])
            
            # For debug only
            # Check equilibrium
            # if (tm.length(cf[offset].force_a + cf[offset].force_b) > DoublePrecisionTolerance):
            #     print("Equilibrium error.")

            # Check whether the bond fails
            sigma_c_a = force_b[0] / A_b - r_b / I_b * tm.sqrt(cf[offset].moment_a[1] ** 2 + cf[offset].moment_a[2] ** 2)
            sigma_c_b = force_b[0] / A_b - r_b / I_b * tm.sqrt(cf[offset].moment_b[1] ** 2 + cf[offset].moment_b[2] ** 2)
            sigma_c_max = -tm.min(sigma_c_a, sigma_c_b)
            sigma_t_a = sigma_c_a
            sigma_t_b = sigma_c_b
            sigma_t_max = tm.max(sigma_t_a, sigma_t_b)
            tau_max = ti.abs(cf[offset].moment_a[0]) * r_b / 2.0 / I_b + 4.0 / 3.0 / A_b * tm.sqrt(cf[offset].force_a[1] ** 2 + cf[offset].force_a[2] ** 2)
            if (sigma_c_max >= surf[type_i, type_j].compressiveStrength): # Compressive failure
                cf[offset].isBonded = 0
                cf[offset].isActive = 0
                # print(f"Bond compressive failure at: {i}, {j}")
            elif (sigma_t_max >= surf[type_i, type_j].tensileStrength): # Tensile failure
                cf[offset].isBonded = 0
                cf[offset].isActive = 0
                # print(f"Bond tensile failure at: {i}, {j}\n")
            elif (tau_max >= surf[type_i, type_j].shearStrength): # Shear failure
                cf[offset].isBonded = 0
                cf[offset].isActive = 0
                # print(f"Bond shear failure at: {i}, {j}\n")
            else: # Intact bond, need to conduct force to particles
                # Notice the inverse of signs due to Newton's third law
                # and LOCAL to GLOBAL coordinates
                ti.atomic_add(gf[i].force, ti.Matrix.inverse(rotationMatrix) @ (-cf[offset].force_a))
                ti.atomic_add(gf[j].force, ti.Matrix.inverse(rotationMatrix) @ (-force_b))
                ti.atomic_add(gf[i].moment, ti.Matrix.inverse(rotationMatrix) @ (-cf[offset].moment_a))
                ti.atomic_add(gf[j].moment, ti.Matrix.inverse(rotationMatrix) @ (-cf[offset].moment_b))


    @ti.func
    def evaluate_unbonded(self, offset: Integer):
        dt = self.config.dt
        gf = ti.static(self.gf)
        cf = ti.static(self.cf)
        mf = ti.static(self.mf)
        surf = ti.static(self.surf)
        
        i = cf[offset].i
        j = cf[offset].j
        
        # Contact resolution
        # Find out rotation matrix
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        a = tm.normalize(gf[j].position - gf[i].position)
        b = Vector3(1.0, 0.0, 0.0) # Local x coordinate
        v = tm.cross(a, b)
        s = tm.length(v)
        c = tm.dot(a, b)
        rotationMatrix = Zero3x3()
        if (s < DoublePrecisionTolerance):
            if (c > 0.0):
                rotationMatrix = ti.Matrix.diag(3, 1.0)
            else:
                rotationMatrix = Matrix3x3([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        else:
            vx = Matrix3x3([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
            rotationMatrix = ti.Matrix.diag(3, 1.0) + vx + ((1.0 - c) / s**2) * vx @ vx
            
        length = tm.length(gf[j].position - gf[i].position)
        offset = i * set_max_coordinate_number + j
        # Contact evaluation (with contact model)
        
        # Non-bonded, use Hertz-Mindlin
        # Calculation relative translational and rotational displacements
        # Need to include the impact of particle rotation in contact relative translational displacement
        # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
        # https://doi.org/10.1002/nme.6568
        # Eqs. (1)-(2)
        # Implementation reference: https://github.com/CFDEMproject/LIGGGHTS-PUBLIC/blob/master/src/surface_model_default.h
        # Lines 140-189
        gap = length - gf[i].radius - gf[j].radius # gap must be negative to ensure an intact contact
        delta_n = ti.abs(gap) # For parameter calculation only

        # For debug only
        # if (delta_n > 0.05 * ti.min(gf[i].radius, gf[j].radius)):
        #     print("WARNING: Overlap particle-particle exceeds 0.05")
        
        cf[offset].position = gf[i].position + tm.normalize(gf[j].position - gf[i].position) * (gf[i].radius - delta_n)
        r_i = cf[offset].position - gf[i].position
        r_j = cf[offset].position - gf[j].position
        # Velocity of a point on the surface of a rigid body
        v_c_i = tm.cross(gf[i].omega, r_i) + gf[i].velocity
        v_c_j = tm.cross(gf[j].omega, r_j) + gf[j].velocity
        v_c = rotationMatrix @ (v_c_j - v_c_i) # LOCAL coordinate
        # Parameter calculation
        # Reference: https://www.cfdem.com/media/DEM/docu/gran_model_hertz.html
        type_i: Integer = gf[i].materialType
        type_j: Integer = gf[j].materialType
        Y_star = 1.0 / ((1.0 - mf[type_i].poissonRatio ** 2) / mf[type_i].elasticModulus + (1.0 - mf[type_j].poissonRatio ** 2) / mf[type_j].elasticModulus)
        G_star = 1.0 / (2.0 * (2.0 - mf[type_i].poissonRatio) * (1.0 + mf[type_i].poissonRatio) / mf[type_i].elasticModulus + 2.0 * (2.0 - mf[type_j].poissonRatio) * (1.0 + mf[type_j].poissonRatio) / mf[type_j].elasticModulus)
        R_star = 1.0 / (1.0 / gf[i].radius + 1.0 / gf[j].radius)
        m_star = 1.0 / (1.0 / (mf[type_i].density * 4.0 / 3.0 * tm.pi * gf[i].radius ** 3) + 1.0 / (mf[type_j].density * 4.0 / 3.0 * tm.pi * gf[j].radius ** 3))
        beta  = tm.log(surf[type_i, type_j].coefficientRestitution) / tm.sqrt(tm.log(surf[type_i, type_j].coefficientRestitution) ** 2 + tm.pi ** 2)
        S_n  = 2.0 * Y_star * tm.sqrt(R_star * delta_n)
        S_t  = 8.0 * G_star * tm.sqrt(R_star * delta_n)
        k_n  = 4.0 / 3.0 * Y_star * tm.sqrt(R_star * delta_n)
        gamma_n  = - 2.0 * beta * tm.sqrt(5.0 / 6.0 * S_n * m_star) # Check whether gamma_n >= 0
        k_t  = 8.0 * G_star * tm.sqrt(R_star * delta_n)
        gamma_t  = - 2.0 * beta * tm.sqrt(5.0 / 6.0 * S_t * m_star) # Check whether gamma_t >= 0

        # Shear displacement increments
        shear_increment = v_c * dt
        shear_increment[0] = 0.0 # Remove the normal direction
        cf[offset].shear_displacement += shear_increment
        # Normal direction - LOCAL - the force towards particle j
        F = Vector3(0.0, 0.0, 0.0)
        # Reference: Peng et al. (2021) Critical time step for discrete element method simulations of convex particles with central symmetry.
        # https://doi.org/10.1002/nme.6568
        # Eq. (29)
        # Be aware of signs
        F[0] = - k_n * gap - gamma_n * v_c[0]
        # Shear direction - LOCAL - the force towards particle j
        try_shear_force = - k_t * cf[offset].shear_displacement
        if (tm.length(try_shear_force) >= surf[type_i, type_j].coefficientFriction * F[0]): # Sliding
            ratio : Real = surf[type_i, type_j].coefficientFriction * F[0] / tm.length(try_shear_force)
            F[1] = try_shear_force[1] * ratio
            F[2] = try_shear_force[2] * ratio
            cf[offset].shear_displacement[1] = F[1] / k_t
            cf[offset].shear_displacement[2] = F[2] / k_t
        else: # No sliding
            F[1] = try_shear_force[1] - gamma_t * v_c[1]
            F[2] = try_shear_force[2] - gamma_t * v_c[2]
        
        # No moment is conducted in Hertz-Mindlin model
        
        # For P4C output
        cf[offset].force_a = F
        # cf[offset].force_b = -F
        # Assigning contact force to particles
        # Notice the inverse of signs due to Newton's third law
        # and LOCAL to GLOBAL coordinates
        F_i_global = ti.Matrix.inverse(rotationMatrix) @ (-F)
        F_j_global = ti.Matrix.inverse(rotationMatrix) @ F
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
    def resolve_unbonded(self, i : Integer, j : Integer):
        '''
        Particle-particle contact detection
        '''
        # alias
        gf = ti.static(self.gf)
        cf = ti.static(self.cf)

        mf = ti.static(self.mf)
        surf = ti.static(self.surf)
        
        eval = False
        # Particle-particle contacts
        offset = self.search_active_contact_offset(i, j)
        
        if (offset >= 0): # Existing contact
            if (cf[offset].isBonded): # Bonded contact
                eval = False # Bonded contact must exist. which is already solved in previous step: resolve_bonded()
            else: # Non-bonded contact, should check whether two particles are still in contact
                if (- gf[i].radius - gf[j].radius + tm.length(gf[j].position - gf[i].position) < 0): # Use PFC's gap < 0 criterion
                    eval = True
                else:
                    cf[offset].isActive = 0
        else:
            if (- gf[i].radius - gf[j].radius + tm.length(gf[j].position - gf[i].position) < 0): # Use PFC's gap < 0 criterion
                offset = self.append_contact_offset(i)
                if(offset < 0): print(f"ERROR: coordinate number > set_max_coordinate_number({set_max_coordinate_number})")
                cf[offset] = Contact( # Hertz-Mindlin model
                    i = i,
                    j = j,
                    isActive = 1,
                    isBonded = 0,
                    materialType_i = 0,
                    materialType_j = 0,
                    force_a = Vector3(0.0, 0.0, 0.0),
                    moment_a = Vector3(0.0, 0.0, 0.0),
                    moment_b = Vector3(0.0, 0.0, 0.0),
                    shear_displacement = Vector3(0.0, 0.0, 0.0)
                )
                eval = True # Send to evaluation using Hertz-Mindlin contact model
        
        if(eval): self.evaluate_unbonded(offset)


    @ti.func
    def evaluate_wall(self, i : Integer, j : Integer): # i is particle, j is wall
        '''
        Particle-wall contact evaluation
        Contact model is Hertz-Mindlin
        '''
        
        # alias
        gf = ti.static(self.gf)
        wf = ti.static(self.wf)
        wcf = ti.static(self.wcf)

        mf = ti.static(self.mf)
        surf = ti.static(self.surf)
        
        dt = self.config.dt
        # Contact resolution
        # Find out rotation matrix
        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        a = wf[j].normal
        b = Vector3(1.0, 0.0, 0.0) # Local x coordinate
        v = tm.cross(a, b)
        s = tm.length(v)
        c = tm.dot(a, b)
        rotationMatrix = Zero3x3();
        if (s < DoublePrecisionTolerance):
            if (c > 0.0):
                rotationMatrix = ti.Matrix.diag(3, 1.0);
            else:
                rotationMatrix = Matrix3x3([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]);
        else:
            vx = Matrix3x3([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
            rotationMatrix = ti.Matrix.diag(3, 1.0) + vx + ((1.0 - c) / s**2) * vx @ vx

        # Calculation relative translational and rotational displacements
        distance = tm.dot(gf[i].position, wf[j].normal) - wf[j].distance # Distance < 0 means that particle is beneath the plane
        gap = ti.abs(distance) - gf[i].radius # gap must be negative
        delta_n = ti.abs(gap) # For parameter calculation only

        # For debug only
        # if (delta_n > 0.05 * gf[i].radius):
        #     print("WARNING: Overlap particle-wall exceeds 0.05");

        r_i = - distance * wf[j].normal / ti.abs(distance) * (ti.abs(distance) + delta_n / 2.0)
        wcf[i, j].position = gf[i].position + r_i
        # Velocity of a point on the surface of a rigid body
        v_c_i = tm.cross(gf[i].omega, r_i) + gf[i].velocity
        v_c = rotationMatrix @ (- v_c_i) # LOCAL coordinate
        # Parameter calculation
        # Reference: https://www.cfdem.com/media/DEM/docu/gran_model_hertz.html
        type_i: Integer = gf[i].materialType;
        type_j: Integer = wf[j].materialType;
        Y_star = 1.0 / ((1.0 - mf[type_i].poissonRatio ** 2) / mf[type_i].elasticModulus + (1.0 - mf[type_j].poissonRatio ** 2) / mf[type_j].elasticModulus)
        G_star = 1.0 / (2.0 * (2.0 - mf[type_i].poissonRatio) * (1.0 + mf[type_i].poissonRatio) / mf[type_i].elasticModulus + 2.0 * (2.0 - mf[type_j].poissonRatio) * (1.0 + mf[type_j].poissonRatio) / mf[type_j].elasticModulus)
        R_star = gf[i].radius
        m_star = mf[type_i].density * 4.0 / 3.0 * tm.pi * gf[i].radius ** 3;
        beta = tm.log(surf[type_i, type_j].coefficientRestitution) / tm.sqrt(tm.log(surf[type_i, type_j].coefficientRestitution) ** 2 + tm.pi ** 2)
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
        if (tm.length(try_shear_force) >= surf[type_i, type_j].coefficientFriction * F[0]): # Sliding
            ratio = surf[type_i, type_j].coefficientFriction * F[0] / tm.length(try_shear_force)
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
        # wcf[i, j].force_b = -F;
        # Assigning contact force to particles
        # Notice the inverse of signs due to Newton's third law
        # and LOCAL to GLOBAL coordinates
        # As the force is at contact position
        # additional moments will be assigned to particles
        F_i_global = ti.Matrix.inverse(rotationMatrix) @ (-F)
        
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
                            materialType_i = 0,
                            materialType_j = 1,
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
        
        contact_radius_i = gf[i].contactRadius
        contact_radius_j = gf[j].contactRadius
        if (tm.length(gf[j].position - gf[i].position) - contact_radius_i - contact_radius_j < 0.0):
            offset = self.append_contact_offset(i)
            if(offset >= 0):
                cf[offset] = Contact( # Forced to bond contact
                i = i,
                j = j,
                isActive = 1,
                isBonded = 1,
                materialType_i = 0,
                materialType_j = 0,
                force_a = Vector3(0.0, 0.0, 0.0),
                moment_a = Vector3(0.0, 0.0, 0.0),
                # force_b = Vector3(0.0, 0.0, 0.0),
                moment_b = Vector3(0.0, 0.0, 0.0),
                )
            else:
                print(f"ERROR: coordinate number > set_max_coordinate_number({set_max_coordinate_number})")


    def bond(self):
        '''
        Similar to contact, but runs only once at the beginning
        '''
        # In example 911, brute detection has better efficiency
        if(self.workload_type == WorkloadType.Heavy):
            # use detect_collision in implicit mode
            self.bpcd.detect_collision(self.gf.position, self.bond_detect)
        else:
            raise("In Taichi Hackathon 2022, only heavy workload is supported")
        # use detect_collision in implicit mode


    # Neighboring search
    def contact(self):
        '''
        Handle the collision between grains.
        '''
        
        if(self.workload_type == WorkloadType.Heavy):
            
            
            if(self.statistics!=None):self.statistics.BroadPhaseDetectionTime.tick()
            # No BPCD needed as no non-bonded particle-particle contact model exists - Denver Pilphis
            # self.bpcd.detect_collision(self.gf.position)
            if(self.statistics!=None):self.statistics.BroadPhaseDetectionTime.tick()
            
            if(self.statistics!=None):self.statistics.ContactResolveTime.tick()
            # As bond contacts are sorted separately
            # This should work - Denver Pilphis
            self.resolve_bonded()
            # No non-bonded particle-particle contact model exists - Denver Pilphis
            # self.cp_list_resolve_collision(self.bpcd.get_collision_pair_range(), self.bpcd.get_collision_pair_list())
            if(self.statistics!=None):self.statistics.ContactResolveTime.tick()
        else:
            raise("In Taichi Hackathon 2022 we only use heavy workload")


    def run_simulation(self):
        if(self.statistics!=None):self.statistics.SolveTime.tick()
        self.clear_state()
        # Particle-particle 
        # Neighboring search + Contact detection, resolution and evaluation
        if(self.statistics!=None):self.statistics.ContactTime.tick()
        self.contact()
        if(self.statistics!=None):self.statistics.ContactTime.tick()
        
        # Particle-wall
        if(self.statistics!=None):self.statistics.ResolveWallTime.tick()
        self.resolve_wall()
        if(self.statistics!=None):self.statistics.ResolveWallTime.tick()
        
        # Particle body force
        if(self.statistics!=None):self.statistics.ApplyForceTime.tick()
        self.apply_body_force() 
        if(self.statistics!=None):self.statistics.ApplyForceTime.tick()
        
        # Taichi Hackathon 2022 append
        # "Time sequence - particle range - body force" implementation
        # Similar to but much more than "fix setforce command" in LIGGGHTS:
        # https://www.cfdem.com/media/DEM/docu/fix_setforce.html
        self.timeSquence.add_force(self.frame, self.gf.force, self.grainID2Index)

        # Time integration
        if(self.statistics!=None):self.statistics.UpdateTime.tick()
        self.update()
        if(self.statistics!=None):self.statistics.UpdateTime.tick()
        
        # clear certain states at the end
        self.late_clear_state()
        if(self.statistics!=None):self.statistics.SolveTime.tick()

        # Update step count and time
        self.frame += 1
        self.elapsedTime += self.config.dt


    def init_simulation(self):
        self.bond()


#=======================================================================
# entrance
#=======================================================================
if __name__ == '__main__':
    config = DEMSolverConfig()
    # Disable simulation benchmark
    statistics = None
    # statistics = DEMSolverStatistics()
    solver = DEMSolver(config, statistics, WorkloadType.Heavy)
    domain_min = set_domain_min
    domain_max = set_domain_max
    solver.init_particle_fields(set_init_particles,domain_min,domain_max)
    print(f"hash table size = {solver.bpcd.hash_table.shape[0]}, cell_size = {solver.bpcd.cell_size}")
    
    tstart = time.time()
    
    # Deprecated as step count and time elapsed are enclosed in solver
    # step = 0
    # elapsed_time = 0.0
    solver.init_simulation()
    # solver.save('output', 0)
    if(not os.path.exists('output_data/')): os.mkdir('output_data/')
    p4p = open('output_data/output.p4p',encoding="UTF-8",mode='w')
    p4c = open('output_data/output.p4c',encoding="UTF-8",mode='w')
    solver.save_single(p4p,p4c,solver.config.dt * solver.frame)
    # solver.save(f'output_data/{step}', elapsed_time)
    while solver.frame < config.nsteps:
        t1 = time.time()
        for _ in range(config.saving_interval_steps): 
            # step += 1
            # elapsed_time += config.dt
            solver.run_simulation()
        t2 = time.time()
        #solver.save_single(p4p,p4c,solver.config.dt * step)
        print('>>>')
        print(f"solved steps: {solver.frame}/{config.nsteps}(simt={solver.elapsedTime}s) last-{config.saving_interval_steps}-sim: {t2-t1}s")
        # save all frames to the single file
        solver.save_single(p4p, p4c, solver.config.dt * solver.frame)
        # save one frame per file for debugging
        solver.save(f'output_data/{solver.frame//config.saving_interval_steps}', solver.config.dt * solver.frame)
        if(solver.statistics): solver.statistics.report()
        print('<<<')
    p4p.close()
    p4c.close()
    
    tend = time.time()
    
    print(f"total solve time cost = {tend - tstart}")