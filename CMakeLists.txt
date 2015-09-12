cmake_minimum_required(VERSION 2.8)
project(ufl_nbnn_script_identification)

# Select a default build configuration if none was chosen
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

find_package(OpenCV REQUIRED)

ADD_EXECUTABLE(cvsi_ClassifyAllScripts cvsi_ClassifyAllScripts.cpp)
ADD_EXECUTABLE(cvsi_ClassifyAllScriptsWeighted cvsi_ClassifyAllScriptsWeighted.cpp)
ADD_EXECUTABLE(babel_ClassifyAllScripts babel_ClassifyAllScripts.cpp)
ADD_EXECUTABLE(babel_ClassifyAllScriptsWeighted babel_ClassifyAllScriptsWeighted.cpp)

FIND_PACKAGE(OpenCV REQUIRED)
IF(OpenCV_FOUND)
  TARGET_LINK_LIBRARIES(cvsi_ClassifyAllScripts ${OpenCV_LIBS})
  TARGET_LINK_LIBRARIES(cvsi_ClassifyAllScriptsWeighted ${OpenCV_LIBS})
  TARGET_LINK_LIBRARIES(babel_ClassifyAllScripts ${OpenCV_LIBS})
  TARGET_LINK_LIBRARIES(babel_ClassifyAllScriptsWeighted ${OpenCV_LIBS})
ENDIF()
