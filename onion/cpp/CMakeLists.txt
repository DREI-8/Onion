cmake_minimum_required(VERSION 3.4)
project(onion_cpp)

# Configure the compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch cpp files
file(GLOB_RECURSE CPP_SOURCES "*.cpp")

# Create a library
add_library(onion_cpp SHARED ${CPP_SOURCES})

# Installation
install(TARGETS onion_cpp DESTINATION lib)
install(DIRECTORY . DESTINATION include/onion FILES_MATCHING PATTERN "*.h")