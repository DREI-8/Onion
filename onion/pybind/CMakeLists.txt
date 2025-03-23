cmake_minimum_required(VERSION 3.10)
project(onion_pybind)

# Find Python
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)

# Find or download pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1  
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# C++ src path
set(ONION_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cpp" CACHE PATH "Chemin vers le code C++ d'Onion")
set(ONION_CPP_LIB_DIR "" CACHE PATH "Chemin vers la bibliothèque C++ compilée d'Onion")

# Debug info
message(STATUS "ONION_CPP_DIR: ${ONION_CPP_DIR}")
message(STATUS "ONION_CPP_LIB_DIR: ${ONION_CPP_LIB_DIR}")
message(STATUS "CMAKE_LIBRARY_PATH: ${CMAKE_LIBRARY_PATH}")

# Fetch all cpp files
file(GLOB_RECURSE PYBIND_SOURCES "*.cpp")

# Create the module
pybind11_add_module(onion_core ${PYBIND_SOURCES})

# Include C++ headers
target_include_directories(onion_core PRIVATE ${ONION_CPP_DIR})

# Find the C++ library
find_library(ONION_CPP_LIBRARY
    NAMES onion_cpp libonion_cpp
    PATHS ${ONION_CPP_LIB_DIR} ${CMAKE_LIBRARY_PATH}
    PATH_SUFFIXES lib
    REQUIRED
)

message(STATUS "Found onion_cpp library: ${ONION_CPP_LIBRARY}")

# Link the C++ library
target_link_libraries(onion_core PRIVATE ${ONION_CPP_LIBRARY})

# Install the module
install(TARGETS onion_core DESTINATION "${Python_SITEARCH}/onion")