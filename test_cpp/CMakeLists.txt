cmake_minimum_required(VERSION 3.5)

set(PROJECT_VERSION "1.0")

##################################################

project(u2netp_demo)

add_compile_options(
    -std=c++11
)

# set(TOOLCHAIN_PATH_RK3588 "/mnt/CodeDisk/gcc_tools/rv3588-aarch64--glibc--stable-final")
# set(CMAKE_C_COMPILER ${TOOLCHAIN_PATH_RK3588}/bin/aarch64-buildroot-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PATH_RK3588}/bin/aarch64-buildroot-linux-gnu-g++)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/3rd/opencv/include
    ${PROJECT_SOURCE_DIR}/3rd/rknn_api/include
)

link_directories(
    ${PROJECT_SOURCE_DIR}/3rd/rknn_api/lib
    ${PROJECT_SOURCE_DIR}/3rd/opencv/lib
)

# 公共库
link_libraries(
    rknn_api
    opencv_imgcodecs opencv_imgproc opencv_core
)

add_executable(
    u2netp_demo
    ${CMAKE_CURRENT_SOURCE_DIR}/src/u2netp_demo.cpp
)
