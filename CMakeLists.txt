cmake_minimum_required(VERSION 3.12.0)
project(project_title LANGUAGES CXX C)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

include(find_libs.cmake)

set(_GMSH_EXECUTABLE "$ENV{_GMSH_EXECUTABLE}")
if (_GMSH_EXECUTABLE)
    set(CMAKE_FIND_DEBUG_MODE TRUE)
    message(STATUS "Found explicit gmsh tool path at ${_GMSH_EXECUTABLE}")
    set(_GMSH_HINTS ${_GMSH_EXECUTABLE})
else()
    set(_GMSH_HINTS "")
endif()

find_program(
    GMSH_EXECUTABLE
    NAMES gmsh gmsh.exe
    PATHS /usr/local /opt ${_GMSH_HINTS}
)
if (NOT GMSH_EXECUTABLE)
    message(FATAL_ERROR "Required program 'gmsh' not found was not found.        \
    Add gmsh to your default INCLUDE or set the _GMSH_EXECUTABLE with the        \
    path to the folder containing the gmsh executable through                    \
    export _GMSH_EXECUTABLE=<path/to/exe/> .                                     \
    (As a side note, the search may fail if you are using a different subsystem  \
    e.g. WSL, so that the path may not be visible. Also note that a windows      \ 
    executable may not be used for a WSL subsystem)")
else()
    message(STATUS "Tool gmsh was found at directory ${GMSH_EXECUTABLE}")
endif()

string(REPLACE "\\" "/" GMSH_EXECUTABLE ${GMSH_EXECUTABLE})
execute_process(
    COMMAND "${GMSH_EXECUTABLE}" "--version"
    OUTPUT_VARIABLE GMSH_VERSION
    RESULT_VARIABLE EXEC_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (EXEC_RESULT EQUAL 0 AND NOT GMSH_VERSION STREQUAL "")
    message(STATUS "Executable gmsh correctly working with detected version: ${GMSH_VERSION}.")
else()
    message(FATAL_ERROR "Failed to run the gmsh executable. There may be a problem \ 
    with the INCLUDE path or the specified path.")
endif()

message(STATUS "Will now attempt to generate the specified meshes.")

set(_MESH_PATHS "$ENV{_MESH_PATHS}")
if (_MESH_PATHS)
    message(STATUS "External mesh list found: ${_MESH_PATHS}")
    set(_MESHES ${_MESH_PATHS})
else()
    set(_MESHES "../src/gmesh/heart_cup.geo;../src/gmesh/clean_heart_cup.geo")
endif()

file(MAKE_DIRECTORY "build/generated_meshes")

foreach(MESH_ITEM IN LISTS _MESHES)
    if (NOT (MESH_ITEM MATCHES "[ \t\r\n]+" OR NOT MESH_ITEM) )
        message(STATUS "Generating mesh path: ${MESH_ITEM}")
        get_filename_component(MESH_ITEM_MSH "${MESH_ITEM}" NAME)

        string(REGEX REPLACE "\\.[^\\.]+$" ".msh" MESH_ITEM_MSH ${MESH_ITEM_MSH})

        set(MESH_ITEM_MSH "generated_meshes/${MESH_ITEM_MSH}")
        execute_process(
            COMMAND ${GMSH_EXECUTABLE} ${MESH_ITEM} -save -o ${MESH_ITEM_MSH}
            OUTPUT_VARIABLE GMSH_MESH
            RESULT_VARIABLE EXEC_RESULT
        )
        if (EXEC_RESULT EQUAL 0)
            message(STATUS "Successfully generated mesh ${MESH_ITEM_MSH}")
        else()
            message(FATAL ERROR "Failed to generate mesh ${MESH_ITEM_MSH} when calling   \
            the gmsh executable. It's possible that the specified mesh script       \
            failed to be parsed by gmsh.")
        endif()
    endif()
endforeach()

set(CPP_PATH        "src/cpp")
set(CPP_UTILS       "src/cpp/utilities")
set(CPP_SOLVER      "src/cpp/solver")

add_library(se_is STATIC "${CPP_SOLVER}/superelastic_isotropic.cpp")
deal_ii_setup_target(se_is)

add_library(math_util STATIC "${CPP_UTILS}/math.cpp")
deal_ii_setup_target(math_util)

add_library(se_or STATIC "${CPP_SOLVER}/orthotropic_solver.cpp")
deal_ii_setup_target(se_or)
target_link_libraries(se_or math_util)


target_compile_options(se_is PRIVATE
    $<$<CONFIG:Release>:-Ofast>
    $<$<CONFIG:Release>:-march=native>
)

target_compile_options(se_or PRIVATE
    $<$<CONFIG:Release>:-Ofast>
    $<$<CONFIG:Release>:-march=native>
)

set(FIX_INCLUDE_DIR "/usr/include")
# Get the current include directories for my_target
get_target_property(current_includes se_is INTERFACE_INCLUDE_DIRECTORIES)

# Check if the directory is already included
list(FIND current_includes "${FIX_INCLUDE_DIR}" dir_index)
MESSAGE(current_includes)
if(dir_index EQUAL -1)
    message(STATUS "Adding ${FIX_INCLUDE_DIR} to include directories")
    target_include_directories(se_is PRIVATE ${MY_INCLUDE_DIR})
else()
    message(STATUS "${FIX_INCLUDE_DIR} is already in the include directories")
endif()

add_library(me_io STATIC "${CPP_UTILS}/mesh_io.cpp")
deal_ii_setup_target(me_io)

add_executable(main "${CPP_PATH}/main.cpp")
deal_ii_setup_target(main)

target_link_libraries(main se_is)
target_link_libraries(main me_io)
target_link_libraries(main se_or)