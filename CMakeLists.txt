cmake_minimum_required(VERSION 3.10)
project(onion VERSION 0.1.0)

# Configure the compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static linking for MinGW
if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static-libgcc")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive")
    message(STATUS "Using static linking for MinGW runtime libraries")
endif()

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
if(WIN32)
    add_library(onion_cpp STATIC ${CPP_SOURCES} ${CPP_HEADERS})
else()
    add_library(onion_cpp SHARED ${CPP_SOURCES} ${CPP_HEADERS})
endif()

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
    file(GLOB_RECURSE PYBIND_HEADERS "onion/pybind/*.h" "onion/pybind/*.hpp")
    message(STATUS "PYBIND sources: ${PYBIND_SOURCES}")
    message(STATUS "PYBIND headers: ${PYBIND_HEADERS}")
    
    # Create the Python module
    pybind11_add_module(onion ${PYBIND_SOURCES} ${PYBIND_HEADERS})

    if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_link_options(onion PRIVATE 
            "-static-libgcc" 
            "-static-libstdc++" 
            "-Wl,-Bstatic,--whole-archive" 
            "-lwinpthread" 
            "-Wl,--no-whole-archive"
        )
        
        # Find and install MinGW DLLs if they are not found
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++-6.dll
            OUTPUT_VARIABLE LIBSTDCPP_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        get_filename_component(MINGW_BIN_DIR ${LIBSTDCPP_PATH} DIRECTORY)
        set(MINGW_DLLS
            "${MINGW_BIN_DIR}/libstdc++-6.dll"
            "${MINGW_BIN_DIR}/libgcc_s_seh-1.dll"
            "${MINGW_BIN_DIR}/libwinpthread-1.dll"
        )

        # Install the MinGW DLLs if they are not found
        foreach(DLL ${MINGW_DLLS})
            if(EXISTS ${DLL})
                install(FILES ${DLL} DESTINATION "${Python_SITEARCH}/onion")
                message(STATUS "Installing backup DLL: ${DLL}")
            endif()
        endforeach()
    endif()
    
    # Link the C++ library
    target_link_libraries(onion PRIVATE onion_cpp)
    
    # Include C++ headers
    target_include_directories(onion PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/onion
    )
    
    # Install the Python module
    install(TARGETS onion onion_cpp
            RUNTIME DESTINATION "${Python_SITEARCH}/onion"
            LIBRARY DESTINATION "${Python_SITEARCH}/onion")

    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/onion/__init__.py"
            DESTINATION "${Python_SITEARCH}/onion")
endif()