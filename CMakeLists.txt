cmake_minimum_required(VERSION 3.12)
project(pointnet_hrc_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(Torch_DIR "/home/internship/Downloads/libtorch/share/cmake/Torch")

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(ZED REQUIRED)
find_package(PCL REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Add executable with all source files
add_executable(${PROJECT_NAME} 
    main.cpp
    pointnet_inference.cpp
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${ZED_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${ZED_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${PCL_LIBRARIES}
    ${TORCH_LIBRARIES}
)

# Compiler definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE ${PCL_DEFINITIONS})

# Set install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Add compiler flags for optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(${PROJECT_NAME} PRIVATE -O3 -march=native)
endif()

# Print configuration info
message(STATUS "PCL version: ${PCL_VERSION}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "ZED SDK found: ${ZED_FOUND}")

# Copy model file to build directory
configure_file(${CMAKE_SOURCE_DIR}/DeployModel/laptop_classifier_traced.pt 
               ${CMAKE_BINARY_DIR}/DeployModel/laptop_classifier_traced.pt COPYONLY)
