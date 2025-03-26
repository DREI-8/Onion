cmake_minimum_required(VERSION 3.10)
project(onion VERSION 0.1.0)

# Configure the compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# C++ sources
file(GLOB_RECURSE CPP_SOURCES "onion/*.cpp")
file(GLOB_RECURSE CPP_HEADERS "onion/*.h" "onion/*.hpp")

# Filter out pybind files
list(FILTER CPP_SOURCES EXCLUDE REGEX "onion/pybind/.*")
list(FILTER CPP_HEADERS EXCLUDE REGEX "onion/pybind/.*")

# Create the C++ library
add_library(onion_cpp SHARED ${CPP_SOURCES} ${CPP_HEADERS})

target_include_directories(onion_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onion>
    $<INSTALL_INTERFACE:include/onion>
)

# Standard installation 
install(TARGETS onion_cpp
    EXPORT onion_cpp-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include/onion
)
install(FILES ${CPP_HEADERS} DESTINATION include/onion)

# Build Python bindings 
if(BUILD_PYTHON_BINDINGS)
    # Find Python
    find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
    
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
    
    # Collect Python bindings
    file(GLOB_RECURSE PYBIND_SOURCES "onion/pybind/*.cpp")
    
    # Create the Python module
    pybind11_add_module(onion_core ${PYBIND_SOURCES})
    
    # Link the C++ library
    target_link_libraries(onion_core PRIVATE onion_cpp)
    
    # Include C++ headers
    target_include_directories(onion_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/onion
    )
    
    # Install the Python module
    install(TARGETS onion_core 
            DESTINATION "${Python_SITEARCH}/onion")
endif()