cmake_minimum_required(VERSION 3.16)
project(dtln_ladspa)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find System FFTW3
find_library(FFTW3F_LIB fftw3f REQUIRED)

# Point to your downloaded TensorFlow source
set(TF_SOURCE_DIR "$ENV{HOME}/tensorflow")

# Add TFLite as a sub-project. 
# This automatically imports all headers and strictly links ALL dependencies (Abseil, Flatbuffers, etc.)
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
