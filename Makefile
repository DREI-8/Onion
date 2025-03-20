# Makefile pour la compilation C++ du projet Onion avec CMake et gcc

# Variables
CPP_DIR = ./onion/cpp
BUILD_DIR = ./build
CMAKE = cmake
MAKE = make
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
    @echo "Compilation C++ en mode debug..."
    cd $(DEBUG_DIR) && $(CMAKE) -G $(CMAKE_GENERATOR) \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        ../../$(CPP_DIR)
    cd $(DEBUG_DIR) && $(MAKE)

# Release mode compilation
release-cpp: $(RELEASE_DIR)
    @echo "Compilation C++ en mode release..."
    cd $(RELEASE_DIR) && $(CMAKE) -G $(CMAKE_GENERATOR) \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        ../../$(CPP_DIR)
    cd $(RELEASE_DIR) && $(MAKE)

# pybind11 compilation
pybind: release-cpp
    @echo "Compilation des bindings Python avec pybind11..."
    cd $(RELEASE_DIR) && $(CMAKE) -G $(CMAKE_GENERATOR) \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DPYTHON_EXECUTABLE=$(shell which python3) \
        -DONION_CPP_DIR=../../$(CPP_DIR) \
        ../../$(PYBIND_DIR)
    cd $(RELEASE_DIR) && $(MAKE)
    @echo "Module Python compilé avec succès"

# Launch gdb
debug: debug-cpp
    @echo "Pour débugger, utilisez: gdb $(DEBUG_DIR)/nom_executable"

# Cleaner 
clean:
    rm -rf $(BUILD_DIR)

# Install
install: release-cpp
    cd $(RELEASE_DIR) && $(MAKE) install

# Some help
help:
    @echo "Cibles disponibles:"
    @echo "  all         : compile le code C++ en mode debug et les bindings Python"
    @echo "  debug-cpp   : compile les extensions C++ en mode debug"
    @echo "  release-cpp : compile les extensions C++ en mode release (optimisé)"
    @echo "  pybind      : compile les bindings Python (nécessite release-cpp)"
    @echo "  debug       : compile en debug et affiche la commande pour lancer gdb"
    @echo "  install     : installe les composants C++ (après compilation release)"
    @echo "  clean       : supprime les répertoires de build"
    @echo "  help        : affiche cette aide"