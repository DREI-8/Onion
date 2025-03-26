# Makefile for C++ compilation of the Onion project with CMake and gcc

# Variables
BUILD_DIR = ./build
PYTHON = python
CMAKE = cmake
MAKE = make
CMAKE_GENERATOR = "Unix Makefiles"

ifeq ($(OS),Windows_NT)
	PYTHON_EXECUTABLE = $(subst \,/,$(shell $(PYTHON) -c "import sys; print(sys.executable)"))
	MKDIR = mkdir
else
	PYTHON_EXECUTABLE = $(shell $(PYTHON) -c "import sys; print(sys.executable)")
	MKDIR = mkdir -p
endif

.PHONY: all clean debug release install wheel develop

all: release

$(BUILD_DIR):
	$(MKDIR) $(BUILD_DIR)

$(BUILD_DIR)/debug: $(BUILD_DIR)
	$(MKDIR) $(BUILD_DIR)/debug

$(BUILD_DIR)/release: $(BUILD_DIR)
	$(MKDIR) $(BUILD_DIR)/release

# Compile in debug mode
debug: $(BUILD_DIR)/debug
	@echo "Building in debug mode..."
	cd $(BUILD_DIR)/debug && $(CMAKE) -G $(CMAKE_GENERATOR) \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DPython_EXECUTABLE=$(PYTHON_EXECUTABLE) \
		../..
	cd $(BUILD_DIR)/debug && $(CMAKE) --build .

# Compile in release mode
release: $(BUILD_DIR)/release
	@echo "Building in release mode..."
	cd $(BUILD_DIR)/release && $(CMAKE) -G $(CMAKE_GENERATOR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DPython_EXECUTABLE=$(PYTHON_EXECUTABLE) \
		../..
	cd $(BUILD_DIR)/release && $(CMAKE) --build .

# Install the C++ and Python modules
install: release
	@echo "Installing the Onion project..."
	cd $(BUILD_DIR)/release && $(CMAKE) --install .

# Create a wheel package
wheel: release
	@echo "Creating a wheel package..."
	$(PYTHON) -m pip wheel . -w $(BUILD_DIR)/wheels

# Install in develop mode
develop: 
	@echo "Installing in develop mode..."
	$(PYTHON) -m pip install -e .

# Clean the build directory
clean:
	@echo "Cleaning the build directory..."
	rm -rf $(BUILD_DIR)

# Publish on TestPyPI
test-publish: wheel
	@echo "Publishing on TestPyPI..."
	twine check $(BUILD_DIR)/wheels/*
	twine upload --repository-url https://test.pypi.org/legacy/ $(BUILD_DIR)/wheels/*

# Publish on PyPI
publish: wheel
	@echo "Publishing on PyPI..."
	twine check $(BUILD_DIR)/wheels/*
	twine upload $(BUILD_DIR)/wheels/*

# Help
help:
	@echo "Available targets:"
	@echo "  all:     Build the project in release mode"
	@echo "  debug:   Build the project in debug mode"
	@echo "  release: Build the project in release mode"
	@echo "  install: Install the C++ and Python modules"
	@echo "  wheel:   Create a wheel package"
	@echo "  develop: Install in develop mode"
	@echo "  clean:   Clean the build directory"
	@echo "  help:    Display this help message"