cmake_minimum_required(VERSION 3.0.0)
project(Extrapolation VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(Eigen3 REQUIRED)
MESSAGE( [Main] " EIGEN3_INCLUDE_DIRS = ${EIGEN3_INCLUDE_DIRS}") 
# EIGEN3_INCLUDE_DIRS = /usr/include/eigen3

include_directories(${EIGEN3_INCLUDE_DIRS})

# SET(CMAKE_CXX_FLAGS "-lgsl -lgslcblas -lm -O3")

find_package(GSL REQUIRED)
link_libraries(GSL::gsl GSL::gslcblas)

if (MSVC)
  string(REGEX REPLACE "/W[0-9]" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif (MSVC)

# list(APPEND CMAKE_PREFIX_PATH /home/pesznyak/GSL)

add_executable(Extrapolation main.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17
                                                 CXX_STANDARD_REQUIRED ON
                                                 CXX_EXTENSIONS OFF)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>
                                               $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->)
#if(BUILD_TESTING)
#  add_executable(test1 test1.cpp)
#  add_test(NAME test_op COMMAND test_op)
#endif(BUILD_TESTING)
                                               
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)