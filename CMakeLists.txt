cmake_minimum_required(VERSION 3.12)
project(pointnet_hrc_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(ZED REQUIRED)
find_package(PCL REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${ZED_INCLUDE_DIRS}
  

)

# Create executable
add_executable(${PROJECT_NAME} main.cpp)

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    ${ZED_LIBRARIES}
    ${PCL_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

# Set torch compile flags
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)