cmake_minimum_required(VERSION 2.8)
project(ECN_caffe_tools)

# Select a default build configuration if none was chosen
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

include_directories ("/usr/src/caffe-git/caffe/include/")
include_directories ("/usr/src/caffe-git/caffe/.build_release/src/")
include_directories ("/usr/local/cuda-7.5/targets/x86_64-linux/include/")
include_directories ("/usr/include/hdf5/serial/")
link_directories ("/usr/src/caffe-git/caffe/build/lib/")

set (EXTRA_LIBS ${EXTRA_LIBS} libcaffe.so -lprotobuf -lglog -lboost_system -lopencv_core -lopencv_highgui -lopencv_imgproc)

ADD_EXECUTABLE(eval_siw13 eval_siw13.cpp)
ADD_EXECUTABLE(convert_imageset_ECN convert_imageset_ECN.cpp)
ADD_EXECUTABLE(compute_image_mean_ECN compute_image_mean_ECN.cpp)

TARGET_LINK_LIBRARIES(eval_siw13 ${EXTRA_LIBS})
TARGET_LINK_LIBRARIES(convert_imageset_ECN ${EXTRA_LIBS} -lgflags)
TARGET_LINK_LIBRARIES(compute_image_mean_ECN ${EXTRA_LIBS} -lgflags)
