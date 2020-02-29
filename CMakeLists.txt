cmake_minimum_required(VERSION 2.8.3)
project(ecnn_eval_flow)

find_package(catkin_simple REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

catkin_simple()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

# make the executable
cs_add_executable(ecnn_eval_flow
  src/ecnn_eval_flow.cpp
)

# link the executable to the necesarry libs
target_link_libraries(ecnn_eval_flow
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
)

cs_install()

