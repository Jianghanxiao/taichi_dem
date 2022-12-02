# DEM Config
# All the settings relating to the simulation come from here.

from TypeDefine import *

# Data import

set_init_particles: str = "../input_data/input.p4p"
set_time_sequnce_folder: str = "../input_data/SGF"

#=====================================
# DEM Simulation Configuration
#=====================================

# Configuration

set_domain_min: Vector3 = Vector3(0.0, 0.0, -10.0)
set_domain_max: Vector3 = Vector3(200.0, 300.0, 10.0)

set_gravity: Vector3 = Vector3(0.0, 0.0, 0.0);
set_global_damping_coefficient: Real = 0.0;

set_time_step: Real = 1e-3;
set_target_time: Real = 4.0;
set_saving_interval_time: Real = 0.04;

# Parameters

set_particle_contact_radius_multiplier: Real = 1.1
set_neiboring_search_safety_factor: Real = 1.01
set_particle_elastic_modulus: Real = 7e10
set_particle_poisson_ratio: Real = 0.25

set_wall_normal: Vector3 = Vector3(0.0, -1.0, 0.0)
set_wall_distance: Real = 259.0
set_wall_density: Real = 7800.0
set_wall_elastic_modulus: Real = 2e11
set_wall_poisson_ratio: Real = 0.25

set_bond_radius_ratio: Real = 0.5
set_bond_elastic_modulus: Real = 1e7
set_bond_poission_ratio: Real = 0.25
set_bond_compressive_strength: Real = 1e8
set_bond_tensile_strength: Real = 1e8
set_bond_shear_strength: Real = 1e8

# Taichi Hackathon 2022 append
# Particle-wall bonds have very high strength
# to fix the specimen to the wall
set_pw_bond_radius_ratio: Real = 1.0
set_pw_bond_elastic_modulus: Real = 1e10
set_pw_bond_poission_ratio: Real = 0.25
set_pw_bond_compressive_strength: Real = 1e11
set_pw_bond_tensile_strength: Real = 1e11
set_pw_bond_shear_strength: Real = 1e11

set_pp_coefficient_friction: Real = 0.3
set_pp_coefficient_restitution: Real = 0.9
set_pp_coefficient_rolling_resistance: Real = 0.01

set_pw_coefficient_friction: Real = 0.35
set_pw_coefficient_restitution: Real = 0.7
set_pw_coefficient_rolling_resistance: Real = 0.01

set_max_coordinate_number: Integer = 64
# reserve collision pair count as (set_collision_pair_init_capacity_factor * n)
set_collision_pair_init_capacity_factor = 128

#=====================================
# Environmental Variables
#=====================================
DoublePrecisionTolerance: float = 1e-12 # Boundary between zeros and non-zeros
MaxParticleCount: int = 1000000000
# MaxParticleCount: int = 100