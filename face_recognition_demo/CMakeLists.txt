cmake_minimum_required(VERSION 3.4.1)

get_filename_component(APP_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

project(${APP_NAME})

message(STATUS "APP_NAME = ${APP_NAME}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================
# Toolchain
# =============================
set(TOOLCHAIN_DIR /opt/atk-dlrk356x-toolchain)

set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/usr/bin/aarch64-linux-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/usr/bin/aarch64-linux-g++)

set(CMAKE_SYSROOT ${TOOLCHAIN_DIR}/aarch64-buildroot-linux-gnu/sysroot)
set(SYSROOT ${CMAKE_SYSROOT}/usr/include)

# =============================
# Runtime RPATH
# executable will search ./lib
# =============================
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "lib")

# =============================
# RKNN
# =============================
set(RKNN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(RKNN_RT_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/librknnrt.so)

include_directories(${RKNN_INCLUDE_DIR})

# =============================
# OpenCV
# use OpenCV headers/libs from toolchain sysroot
# =============================
include_directories(${SYSROOT})
include_directories(${SYSROOT}/opencv4)

set(OPENCV_LIBS
    opencv_core
    opencv_flann
    opencv_videoio
    opencv_video
    opencv_highgui
    opencv_imgcodecs
    opencv_imgproc
    opencv_calib3d
)

# =============================
# Source
# =============================
add_executable(${APP_NAME}
    src/main.cc
    src/yolos.cc
    src/face.cc
)

target_link_libraries(${APP_NAME}
    ${RKNN_RT_LIB}
    ${OPENCV_LIBS}
    dl
    pthread
)

# =============================
# Install package
# =============================
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/${APP_NAME})

install(TARGETS ${APP_NAME} DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(DIRECTORY model DESTINATION ./)
install(DIRECTORY gallery DESTINATION ./ OPTIONAL)