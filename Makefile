# Makefile for C++ compilation of the Onion project with CMake and gcc

# Variables
CPP_DIR = ./onion/cpp
BUILD_DIR = ./build
CMAKE = cmake
MAKE = mingw32-make
CMAKE_GENERATOR = "Unix Makefiles"
DEBUG_DIR = $(BUILD_DIR)/debug
RELEASE_DIR = $(BUILD_DIR)/release

# pybind11 variables 
PYBIND_DIR = ./onion/pybind
PYTHON_CONFIG = python3-config
PYTHON_INCLUDES = $(shell $(PYTHON_CONFIG) --includes)
PYBIND11_INCLUDES = $(shell python3 -m pybind11 --includes)

.PHONY: all clean build-cpp debug-cpp release-cpp pybind

# Default target
all: debug-cpp pybind

# Create build directories
$(DEBUG_DIR):
    mkdir -p $(DEBUG_DIR)

$(RELEASE_DIR):
    mkdir -p $(RELEASE_DIR)

# Debug mode compilation
debug-cpp: $(DEBUG_DIR)
    @echo "C++ compilation in debug mode..."
    cd $(DEBUG_DIR) && $(CMAKE) -G $(CMAKE_GENERATOR) \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        ../../$(CPP_DIR)
    cd $(DEBUG_DIR) && cmake --build .

# Release mode compilation
release-cpp: $(RELEASE_DIR)
    @echo "C++ compilation in release mode..."
    cd $(RELEASE_DIR) && $(CMAKE) -G $(CMAKE_GENERATOR) \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        ../../$(CPP_DIR)
    cd $(RELEASE_DIR) && cmake --build .

# pybind11 compilation
pybind: release-cpp
    @echo "Compiling Python bindings with pybind11..."
    cd $(RELEASE_DIR) && $(CMAKE) -G $(CMAKE_GENERATOR) \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DPYTHON_EXECUTABLE=$(shell which python3) \
        -DONION_CPP_DIR=../../$(CPP_DIR) \
        ../../$(PYBIND_DIR)
    cd $(RELEASE_DIR) && cmake --build .
    @echo "Python module successfully compiled in $(RELEASE_DIR)"

# Launch gdb
debug: debug-cpp
    @echo "To debug, use: gdb $(DEBUG_DIR)/executable_name"

# Cleaner 
clean:
    rm -rf $(BUILD_DIR)

# Install
install: release-cpp
    cd $(RELEASE_DIR) && cmake --install .

# Some help
help:
    @echo "Available targets:"
    @echo "  all         : compiles C++ code in debug mode and Python bindings"
    @echo "  debug-cpp   : compile C++ extensions in debug mode"
    @echo "  release-cpp : compiles C++ extensions in release mode"
    @echo "  pybind      : compile Python bindings (requires release-cpp)"
    @echo "  debug       : compile in debug mode and display the command to start gdb"
    @echo "  install     : installs C++ components (after compilation release)"
    @echo "  clean       : deletes build directories"
    @echo "  help        : displays this help"