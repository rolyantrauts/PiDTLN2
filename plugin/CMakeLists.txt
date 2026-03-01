cmake_minimum_required(VERSION 3.16)
project(dtln_ladspa)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Hardware Optimization strictly for Raspberry Pi Zero 2 W ---
# The Pi Zero 2 W uses the Broadcom BCM2710A1 SoC featuring a Quad-core Cortex-A53.
# -mcpu=cortex-a53 & -mtune=cortex-a53: Tunes instruction scheduling for this exact pipeline.
# -march=armv8-a+crc+simd: Strictly enforces 64-bit ARMv8 architecture with NEON SIMD.
# -ffast-math & -ftree-vectorize: Forces the compiler to unroll loops into SIMD parallel math.
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -mcpu=cortex-a53 -march=armv8-a+crc+simd -mtune=cortex-a53 -ffast-math -ftree-vectorize -funsafe-math-optimizations")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mcpu=cortex-a53 -march=armv8-a+crc+simd -mtune=cortex-a53 -ffast-math -ftree-vectorize -funsafe-math-optimizations")

# --- TensorFlow Lite Hardware Delegates ---
# Force TensorFlow Lite to build with ARM NEON, XNNPACK, and RUY enabled
set(TFLITE_ENABLE_XNNPACK ON CACHE BOOL "Enable XNNPACK" FORCE)
set(TFLITE_ENABLE_RUY ON CACHE BOOL "Enable RUY" FORCE)
set(TFLITE_ENABLE_NEON ON CACHE BOOL "Enable NEON" FORCE)

# Find System FFTW3
find_library(FFTW3F_LIB fftw3f REQUIRED)

# Point to your downloaded TensorFlow source
set(TF_SOURCE_DIR "$ENV{HOME}/tensorflow")

# Add TFLite as a sub-project. 
# This automatically imports all headers and strictly links ALL dependencies (Abseil, Flatbuffers, RUY, XNNPACK, etc.)
add_subdirectory(
    "${TF_SOURCE_DIR}/tensorflow/lite" 
    "${CMAKE_CURRENT_BINARY_DIR}/tflite_build" 
    EXCLUDE_FROM_ALL
)

# Build the LADSPA shared object
add_library(dtln_ladspa SHARED dtln_plugin.cpp)

# 'tensorflow-lite' is the official CMake target name that contains everything
target_link_libraries(dtln_ladspa 
    ${FFTW3F_LIB} 
    tensorflow-lite
    pthread 
    dl 
    m
)

# LADSPA plugins must NOT have a "lib" prefix
set_target_properties(dtln_ladspa PROPERTIES PREFIX "")
