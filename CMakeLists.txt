#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(radialStripe4DCurvature)

set(OpenGL_GL_PREFERENCE "GLVND")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_MODULE_PATH ${radialStripe4DCurvature_SOURCE_DIR}/CMakeModules)

find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED highgui core)
find_package(glfw3 REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLU REQUIRED)
find_package(OpenGL REQUIRED)

find_package(LIBIGL REQUIRED)
find_package(libigl REQUIRED)
find_package(dlib REQUIRED)

# Python
SET(PYTHON_LIBRARY "/home/thuemmel/anaconda3/lib/libpython3.7m.so")
SET(PYTHON_INCLUDE_DIR "/home/thuemmel/anaconda3/include/python3.7m")
find_package(PythonLibs 3.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIR})

find_package(OpenMP)
set(OMP_NUM_THREADS "8" CACHE STRING "" FORCE)

# PCL
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(radialStripe4DCurvature ../ ${OPENGL_INCLUDE_DIR} ${LIBIGL_INCLUDE_DIR})
add_executable(radialStripe4DCurvature main.cpp visualization.hpp)
target_link_libraries(radialStripe4DCurvature ${realsense2_LIBRARY} opencv_core opencv_highgui opencv_imgcodecs ${PCL_LIBRARIES} glfw GL ${OPENGL_gl_LIBRARY} glut GLU ${PYTHON_LIBRARY} OpenMP::OpenMP_CXX dlib::dlib)