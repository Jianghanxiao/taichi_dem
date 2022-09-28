#pragma once
#include "pch.h"

//======================================================================
// basic setup
//======================================================================

const bool SAVE_FRAMES = false;
const int window_size = 1024; // Number of pixels of the window
const int grid_n = 128;

const double DoublePrecisionTolerance = 1e-12;

// Debug-convenience parameters
const double wall_position_x = 0.01;
const double time_increment = 1e-7;
const double total_time = 1e-3;
const double save_interval_time = 1e-5;