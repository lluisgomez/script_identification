cmake_minimum_required(VERSION 2.8)
project(ufl-scene-text-recognition-train)

# Select a default build configuration if none was chosen
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

find_package(OpenCV REQUIRED)

ADD_EXECUTABLE(extract_filters extract_filters.cpp)
ADD_EXECUTABLE(extract_features extract_features.cpp)
ADD_EXECUTABLE(show_filters show_filters.cpp)

FIND_PACKAGE(OpenCV REQUIRED)
IF(OpenCV_FOUND)
  TARGET_LINK_LIBRARIES(extract_filters ${OpenCV_LIBS})
  TARGET_LINK_LIBRARIES(extract_features ${OpenCV_LIBS})
  TARGET_LINK_LIBRARIES(show_filters ${OpenCV_LIBS})
ENDIF()
