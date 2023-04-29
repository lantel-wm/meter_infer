cmake_minimum_required(VERSION 2.8)

Project(meterReader)

set(ROOT_DIR ${CMAKE_SOURCE_DIR})
set(SRC_DIR ${ROOT_DIR}/src)
set(INC_DIR ${ROOT_DIR}/include)
set(BINARY_DIR ${ROOT_DIR}/bin)

message(STATUS "Root directory: ${ROOT_DIR}")
message(STATUS "Source directory: ${SRC_DIR}")
message(STATUS "Include directory: ${INC_DIR}")

find_package(OpenCV REQUIRED)

message(STATUS "Opnecv library status: ")
message(STATUS "> version: ${OpenCV_VERSION} ")
message(STATUS "libraries: ${OpenCV_LIBS} ")
message(STATUS "> include: ${OpenCV_INCLUDE_DIRS}  ")

find_package (glog 0.7.0 REQUIRED)

message(STATUS "Glog library status: ")
message(STATUS "> version: ${glog_VERSION} ")
message(STATUS "libraries: ${glog_LIBS} ")
message(STATUS "> include: ${glog_INCLUDE_DIRS}  ")

include_directories(${OpenCV_INCLUDE_DIRS} ${INC_DIR}) 

aux_source_directory(${SRC_DIR} SRC_LIST)

add_executable(test ${SRC_LIST})

target_link_libraries(test ${OpenCV_LIBS} glog::glog)