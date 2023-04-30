cmake_minimum_required(VERSION 2.8)

Project(meterReader)

# set working directory
set(ROOT_DIR ${CMAKE_SOURCE_DIR})
set(SRC_DIR ${ROOT_DIR}/src)
set(INCLUDE_DIR ${ROOT_DIR}/include)
set(BINARY_DIR ${ROOT_DIR}/bin)
message(STATUS "Root directory: ${ROOT_DIR}")
message(STATUS "Source directory: ${SRC_DIR}")
message(STATUS "Include directory: ${INCLUDE_DIR}")



# set compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -O3")
set(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations -Wno-deprecated-declarations ")
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR})

# set opencv
find_package(OpenCV REQUIRED)

message(STATUS "Opnecv library status: ")
message(STATUS "> version: ${OpenCV_VERSION} ")
message(STATUS "> libraries: ${OpenCV_LIBS} ")
message(STATUS "> include: ${OpenCV_INCLUDE_DIRS}  ")

# set glog
find_package (glog 0.7.0 REQUIRED)

message(STATUS "Glog library status: ")
message(STATUS "> version: ${glog_VERSION} ")
message(STATUS "> libraries: ${glog_LIBRARIES} ")
message(STATUS "> include: ${glog_INCLUDE_DIRS}  ")

# set cuda
find_package(CUDA REQUIRED)
message(STATUS "CUDA library status: ")
message(STATUS "> version: ${CUDA_VERSION} ")
message(STATUS "> libraries: ${CUDA_LIBRARIES} ")
message(STATUS "> include: ${CUDA_INCLUDE_DIRS}  ")

# set tensorrt
set(TensorRT_INCLUDE_DIRS /usr/local/TensorRT-8.6.0.12/include)
set(TensorRT_SAMPLE_INCLUDE_DIRS /usr/local/TensorRT-8.6.0.12/samples/common)
set(TensorRT_LIB_DIRS /usr/local/TensorRT-8.6.0.12/lib)
set(TensorRT_TARGETS_LIB_DIR /usr/local/TensorRT-8.6.0.12/targets/x86_64-linux-gnu/lib)
link_directories(${TensorRT_LIB_DIRS} ${TensorRT_TARGETS_LIB_DIR})

include_directories(${OpenCV_INCLUDE_DIRS} ${INCLUDE_DIR} ${TensorRT_INCLUDE_DIRS} ${TensorRT_SAMPLE_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# set source file
aux_source_directory(${SRC_DIR} SRC_LIST)

# set executable file
add_executable(test ${SRC_LIST})

# set link library
target_link_libraries(test ${OpenCV_LIBS})
target_link_libraries(test ${CUDA_LIBRARIES})
target_link_libraries(test glog::glog  )
target_link_libraries(test "nvinfer" "nvinfer_plugin")