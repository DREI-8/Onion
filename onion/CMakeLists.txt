cmake_minimum_required(VERSION 3.4)
project(onion_cpp)

# Configure the compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch cpp files
file(GLOB_RECURSE CPP_SOURCES "*.cpp")
file(GLOB_RECURSE CPP_HEADERS "*.h" "*.hpp")

# Filter out pybind files
list(FILTER CPP_SOURCES EXCLUDE REGEX "pybind/.*")
list(FILTER CPP_HEADERS EXCLUDE REGEX "pybind/.*")

# Create a library
add_library(onion_cpp SHARED ${CPP_SOURCES} ${CPP_HEADERS})

# Installation
install(TARGETS onion_cpp DESTINATION lib)
install(FILES ${CPP_HEADERS} DESTINATION include/onion)