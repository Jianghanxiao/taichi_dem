from TypeDefine import *
from DemConfig import *

#=====================================
# Init Data Structure
#=====================================
class DEMSolverConfig:
    def __init__(self):
        # Gravity, a global parameter
        # Denver Pilphis: in this example, we assign no gravity
        self.gravity : Vector3 = set_gravity
        # Global damping coefficient
        self.global_damping = set_global_damping_coefficient
        # Time step, a global parameter
        self.dt : Real = set_time_step  # Larger dt might lead to unstable results.
        self.target_time : Real = set_target_time
        # No. of steps for run, a global parameter
        self.nsteps : Integer = int(self.target_time / self.dt)
        self.saving_interval_time : Real = set_saving_interval_time
        self.saving_interval_steps : Real = int(self.saving_interval_time / self.dt)
        

#======================================
# Data Class Definition
#======================================
# Material property
@ti.dataclass
class Material: # Size: 24B
    # Material attributes
    density: Real  # Density, double
    elasticModulus: Real  # Elastic modulus, double
    poissonRatio: Real # Poisson's ratio, double

# Surface interaction property
@ti.dataclass
class Surface: # Size: 72B
    # Hertz-Mindlin parameters
    coefficientFriction: Real # Friction coefficient, double
    coefficientRestitution: Real # Coefficient of resitution, double
    coefficientRollingResistance: Real # Coefficient of rolling resistance, double
    # EBPM parameters
    radius_ratio : Real # Section radius ratio
    elasticModulus : Real # Elastic modulus of the bond
    poissonRatio : Real # Possion's ratio of the bond
    compressiveStrength: Real # Compressive strengthrotationMatrixInte of the bond
    tensileStrength: Real # Tensile strength of the bond
    shearStrength: Real # Shear strength of the bond

# Particle in DEM
# Denver Pilphis: keep spherical shape at this stage, added particle attributes to make the particle kinematically complete
@ti.dataclass
class Grain: # Size: 296B
    ID: Integer # Record Grain ID
    materialType: Integer # Type number of material
    radius: Real  # Radius, double
    contactRadius: Real
    # Translational attributes, all in GLOBAL coordinates
    position: Vector3  # Position, Vector3
    velocity: Vector3  # Velocity, Vector3
    acceleration: Vector3  # Acceleration, Vector3
    force: Vector3  # Force, Vector3
    # Rotational attributes, all in GLOBAL coordinates
    quaternion: Vector4  # Quaternion, Vector4, order in [w, x, y, z]
    omega: Vector3  # Angular velocity, Vector3
    omega_dot: Vector3  # Angular acceleration, Vector3
    inertia: Matrix3x3 # Moment of inertia tensor, 3 * 3 matrix with double
    moment: Vector3 # Total moment (including torque), Vector3

# Wall in DEM
# Only premitive wall is implemented
@ti.dataclass
class Wall: # Size: 36B
    # Wall equation: Ax + By + Cz - D = 0
    # Reference: Peng and Hanley (2019) Contact detection between convex polyhedra and superquadrics in discrete element codes.
    # https://doi.org/10.1016/j.powtec.2019.07.082
    # Eq. (8)
    normal: Vector3 # Outer normal vector of the wall, [A, B, C]
    distance: Real # Distance between origin and the wall, D
    # Material properties
    materialType: Integer

# Contact in DEM
# In this example, the Edinburgh Bond Particle Model (EBPM), along with Hertz-Mindlin model, is implemented
# Reference: Brown et al. (2014) A bond model for DEM simulation of cementitious materials and deformable structures.
# https://doi.org/10.1007/s10035-014-0494-4
# Reference: Mindlin and Deresiewicz (1953) Elastic spheres in contact under varying oblique forces.
# https://doi.org/10.1115/1.4010702
@ti.dataclass
class Contact: # Size: 144B
    i:Integer
    j:Integer
    # Contact status
    isActive : Integer # Contact exists: 1 - exist 0 - not exist
    isBonded : Integer # Contact is bonded: 1 - bonded, use EBPM 0 - unbonded, use Hertz-Mindlin
    # Common Parameters
    materialType_i: Integer
    materialType_j: Integer
    # rotationMatrix : Matrix3x3 # Rotation matrix from global to local system of the contact
    position : Vector3 # Position of contact point in GLOBAL coordinate
    # radius : Real # Section radius: r = rratio * min(r1, r2), temporarily calculated in evaluation
    # length : Real # Length of the bond
    # EBPM parts
    force_a : Vector3 # Contact force at side a in LOCAL coordinate
    moment_a : Vector3 # Contact moment/torque at side a in LOCAL coordinate
    # force_b = - force_a due to equilibrium
    # force_b : Vector3 # Contact force at side b in LOCAL coordinate
    moment_b : Vector3 # Contact moment/torque at side b in LOCAL coordinate
    # Hertz-Mindlin parts
    shear_displacement: Vector3 # Shear displacement stored in the contact

@ti.dataclass
class IOContact: # Size: 64B
    '''
    Contact data for IO
    '''
    i:Integer
    j:Integer
    position:Vector3
    force_a:Vector3
    isBonded:Integer
    isActive:Integer

class WorkloadType:
    Auto = -1
    Light = 0
    Midium = 1
    Heavy = 2