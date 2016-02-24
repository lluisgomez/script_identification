cmake_minimum_required(VERSION 2.8)
project(ufl_nbnn_script_identification)

# Select a default build configuration if none was chosen
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

find_package(OpenCV REQUIRED)

ADD_EXECUTABLE(cvsi_ClassifyAllScripts cvsi_ClassifyAllScripts.cpp)
ADD_EXECUTABLE(cvsi_ClassifyAllScriptsWeighted cvsi_ClassifyAllScriptsWeighted.cpp)
ADD_EXECUTABLE(mle2e_ClassifyAllScripts mle2e_ClassifyAllScripts.cpp)
ADD_EXECUTABLE(mle2e_ClassifyAllScriptsWeighted mle2e_ClassifyAllScriptsWeighted.cpp)

FIND_PACKAGE(OpenCV REQUIRED)
IF(OpenCV_FOUND)
  TARGET_LINK_LIBRARIES(cvsi_ClassifyAllScripts ${OpenCV_LIBS})
  TARGET_LINK_LIBRARIES(cvsi_ClassifyAllScriptsWeighted ${OpenCV_LIBS})
  TARGET_LINK_LIBRARIES(mle2e_ClassifyAllScripts ${OpenCV_LIBS})
  TARGET_LINK_LIBRARIES(mle2e_ClassifyAllScriptsWeighted ${OpenCV_LIBS})
ENDIF()
