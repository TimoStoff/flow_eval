cmake_minimum_required(VERSION 2.8.3)
project(ecnn_eval_flow)

find_package(catkin_simple REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(HDFQL_ROOT "/usr/local/hdfql-2.1.0")
include_directories(${HDFQL_ROOT}/include)

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
   "${HDFQL_ROOT}/wrapper/cpp/libHDFql.so"
)

cs_install()

