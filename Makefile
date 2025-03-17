# Makefile pour la compilation C++ du projet Onion avec CMake et gcc

# Variables
CPP_DIR = ./onion/cpp
BUILD_DIR = ./build
CMAKE = cmake
MAKE = make
CMAKE_GENERATOR = "Unix Makefiles"
DEBUG_DIR = $(BUILD_DIR)/debug
RELEASE_DIR = $(BUILD_DIR)/release

.PHONY: all clean build-cpp debug-cpp release-cpp

# Cible par défaut
all: debug-cpp

# Création des répertoires de build si nécessaire
$(DEBUG_DIR):
    mkdir -p $(DEBUG_DIR)

$(RELEASE_DIR):
    mkdir -p $(RELEASE_DIR)

# Compilation en mode debug
debug-cpp: $(DEBUG_DIR)
    @echo "Compilation C++ en mode debug..."
    cd $(DEBUG_DIR) && $(CMAKE) -G $(CMAKE_GENERATOR) \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        ../../$(CPP_DIR)
    cd $(DEBUG_DIR) && $(MAKE)

# Compilation en mode release
release-cpp: $(RELEASE_DIR)
    @echo "Compilation C++ en mode release..."
    cd $(RELEASE_DIR) && $(CMAKE) -G $(CMAKE_GENERATOR) \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        ../../$(CPP_DIR)
    cd $(RELEASE_DIR) && $(MAKE)

# Lancement du débugger sur un exécutable
debug: debug-cpp
    @echo "Pour débugger, utilisez: gdb $(DEBUG_DIR)/nom_executable"

# Nettoyage
clean:
    rm -rf $(BUILD_DIR)

# Installation (après compilation release)
install: release-cpp
    cd $(RELEASE_DIR) && $(MAKE) install

# Aide
help:
    @echo "Cibles disponibles:"
    @echo "  all         : compile le code C++ en mode debug (défaut)"
    @echo "  debug-cpp   : compile les extensions C++ en mode debug"
    @echo "  release-cpp : compile les extensions C++ en mode release (optimisé)"
    @echo "  debug       : compile en debug et affiche la commande pour lancer gdb"
    @echo "  install     : installe les composants C++ (après compilation release)"
    @echo "  clean       : supprime les répertoires de build"
    @echo "  help        : affiche cette aide"