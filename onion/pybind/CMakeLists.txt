cmake_minimum_required(VERSION 3.4)
project(onion_pybind)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find or download pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1  # Utiliser une version stable
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# C++ src path
set(ONION_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cpp" CACHE PATH "Chemin vers le code C++ d'Onion")

# Fetch all cpp files
file(GLOB_RECURSE PYBIND_SOURCES "*.cpp")

# Create the module
pybind11_add_module(onion_core ${PYBIND_SOURCES})

# Include C++ headers
target_include_directories(onion_core PRIVATE ${ONION_CPP_DIR})

# Link the C++ library
target_link_libraries(onion_core PRIVATE onion_cpp)

# Install the module
install(TARGETS onion_core DESTINATION "${Python_SITEARCH}/onion")