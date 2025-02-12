cmake_minimum_required(VERSION 3.15)

# ------------------------------
# 1. Project Setup
# ------------------------------

# Option to enable or disable CUDA support
option(ENABLE_CUDA "Enable CUDA support" ON)

# Enable response files for CUDA on Windows to handle long argument lists
if(ENABLE_CUDA)
    # The following lines help keep 'nvcc' command lines from overflowing
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS ON)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES ON)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES ON)
endif()
# Use vcpkg toolchain
if(WIN32)
    set(CMAKE_TOOLCHAIN_FILE
        "D:/dev/C++/Dependency_Managers/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file"
    )
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    set(CMAKE_TOOLCHAIN_FILE
        "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING ""
    )
endif()

# Conditionally include CUDA in the project languages
if(ENABLE_CUDA)
    project(CeLux LANGUAGES CXX CUDA)
else()
    project(CeLux LANGUAGES CXX)
endif()

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------
# 2. Specify CUDA Paths Directly
# ------------------------------

if(ENABLE_CUDA)
    if(WIN32)
        # Windows-specific CUDA paths
        # NOTE: If you actually want CUDA 12.2, change v12.1 to v12.2 here
        set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")
        include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
        set(NPP_LIBRARIES nppicc.lib)
    else()
        # Linux-specific CUDA paths
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
        include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
        set(NPP_LIBRARIES nppicc)
    endif()
endif()

# ------------------------------
# 3. Manually Specify FFmpeg Static Libraries
# ------------------------------

if(WIN32)
    # Windows-specific FFmpeg paths
    set(FFMPEG_VCPKG_STATIC_DIR "D:/dev/C++/Dependency_Managers/vcpkg/installed/x64-windows-static")
    set(FFMPEG_LIB_EXT ".lib")
else()
    # Linux-specific FFmpeg paths
    set(FFMPEG_VCPKG_STATIC_DIR "${CMAKE_SOURCE_DIR}/vcpkg/installed/x64-linux")
    set(FFMPEG_LIB_EXT ".a")
endif()

# Common FFmpeg include directories
set(FFMPEG_INCLUDE_DIRS "${FFMPEG_VCPKG_STATIC_DIR}/include")

# FFmpeg dependencies
set(FFMPEG_DEPENDENCIES
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/libwebp${FFMPEG_LIB_EXT}"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/libwebpdecoder${FFMPEG_LIB_EXT}"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/libwebpdemux${FFMPEG_LIB_EXT}"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/libwebpmux${FFMPEG_LIB_EXT}"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/libx264${FFMPEG_LIB_EXT}"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/x265-static${FFMPEG_LIB_EXT}"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/libsharpyuv${FFMPEG_LIB_EXT}"
)

# FFmpeg static libraries
set(FFMPEG_STATIC_LIBS
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/avutil${FFMPEG_LIB_EXT}"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/avcodec${FFMPEG_LIB_EXT}"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/avformat${FFMPEG_LIB_EXT}"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/avdevice${FFMPEG_LIB_EXT}"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/swscale${FFMPEG_LIB_EXT}"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/swresample${FFMPEG_LIB_EXT}"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/avfilter${FFMPEG_LIB_EXT}"
    ${FFMPEG_DEPENDENCIES}
)

# ------------------------------
# 4. Keep Torch/CUDA as Dynamic Libraries (B)
# ------------------------------

if(WIN32)
    if(ENABLE_CUDA)
        set(TORCH_INSTALL_PREFIX "D:/dev/C++/Libraries/libtorch-cuda/libtorch")
    else()
        set(TORCH_INSTALL_PREFIX "D:/dev/C++/Libraries/libtorch-cpu/libtorch")
    endif()
else()
    if(ENABLE_CUDA)
        set(TORCH_INSTALL_PREFIX "/opt/libtorch_cuda/libtorch")
    else()
        set(TORCH_INSTALL_PREFIX "/opt/libtorch_cpu/libtorch")
    endif()
endif()
set(CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}")

# Find Torch
find_package(Torch REQUIRED)

# ------------------------------
# 5. Find Windows Media Foundation Libraries
# ------------------------------

if(WIN32)
    # Windows-specific code for Media Foundation
    set(WINDOW_SDK_LIB_PATH "C:/Program Files (x86)/Windows Kits/10/Lib")

    find_library(MF_LIB NAMES mf PATHS "${WINDOW_SDK_LIB_PATH}/*/um/x64" NO_DEFAULT_PATH)
    if(NOT MF_LIB)
        message(FATAL_ERROR "mf.lib not found. Ensure the Windows SDK is installed.")
    endif()

    find_library(MFUUUID_LIB NAMES mfuuid PATHS "${WINDOW_SDK_LIB_PATH}/*/um/x64" NO_DEFAULT_PATH)
    if(NOT MFUUUID_LIB)
        message(FATAL_ERROR "mfuuid.lib not found. Ensure the Windows SDK is installed.")
    endif()

    find_library(OLE32_LIB NAMES ole32 PATHS "${WINDOW_SDK_LIB_PATH}/*/um/x64" NO_DEFAULT_PATH)
    if(NOT OLE32_LIB)
        message(FATAL_ERROR "ole32.lib not found. Ensure the Windows SDK is installed.")
    endif()
endif()

# ------------------------------
# 6. Manually Specify Python 3.12 Paths (C)
# ------------------------------

if(WIN32)
    set(PYTHON_EXECUTABLE "C:/Users/tjerf/AppData/Local/Programs/Python/Python312/python.exe")
    set(PYTHON_INCLUDE_DIR "C:/Users/tjerf/AppData/Local/Programs/Python/Python312/include")
    set(PYTHON_LIBRARY "C:/Users/tjerf/AppData/Local/Programs/Python/Python312/libs/python312.lib")

    if(NOT EXISTS "${PYTHON_EXECUTABLE}")
        message(FATAL_ERROR "Python executable not found at ${PYTHON_EXECUTABLE}")
    endif()

    if(NOT EXISTS "${PYTHON_INCLUDE_DIR}")
        message(FATAL_ERROR "Python include directory not found at ${PYTHON_INCLUDE_DIR}")
    endif()

    if(NOT EXISTS "${PYTHON_LIBRARY}")
        message(FATAL_ERROR "Python library not found at ${PYTHON_LIBRARY}")
    endif()
else()
    # Linux-specific code to find Python 3.12
    find_package(Python3 3.12 COMPONENTS Interpreter Development REQUIRED)
    if(NOT Python3_FOUND)
        message(FATAL_ERROR "Python 3.12 not found.")
    endif()
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
    set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
    set(PYTHON_LIBRARY ${Python3_LIBRARIES})
endif()

# ------------------------------
# 7. Find pybind11 using vcpkg
# ------------------------------
set(pybind11_DIR "D:/dev/C++/Dependency_Managers/vcpkg/installed/x64-windows-static/share/pybind11")
set(spdlog_DIR "D:/dev/C++/Dependency_Managers/vcpkg/installed/x64-windows-static/share/spdlog")
set(fmt_DIR    "D:/dev/C++/Dependency_Managers/vcpkg/installed/x64-windows-static/share/fmt")

find_package(pybind11 REQUIRED)

# ------------------------------
# 8. Find Torch Python Library
# ------------------------------

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PYTHON_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT TORCH_PYTHON_PATH)
    message(FATAL_ERROR "Could not find torch in the current Python environment.")
endif()

find_library(
    TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS "${TORCH_PYTHON_PATH}/lib" "${TORCH_PYTHON_PATH}"
    NO_DEFAULT_PATH
)

if(NOT TORCH_PYTHON_LIBRARY)
    message(FATAL_ERROR "Torch Python library not found.")
endif()

# ------------------------------
# 9. Add Source Files
# ------------------------------

file(GLOB_RECURSE SOURCES_COMMON
    "src/CeLux/*.cpp"
    "src/CeLux/conversion/*.cpp"
    "src/CeLux/backends/*.cpp"
    "src/CeLux/conversion/cpu/*.cpp"
    "src/CeLux/filters/*.cpp"
)

if(ENABLE_CUDA)
    # Gather .cu files separately
    file(GLOB_RECURSE SOURCES_CUDA
        "src/CeLux/cuda/*.cu"
    )
    set(SOURCES ${SOURCES_COMMON} ${SOURCES_CUDA})
else()
    set(SOURCES ${SOURCES_COMMON})
endif()

# Create the main CeLuxLib static library
add_library(CeLuxLib STATIC ${SOURCES})

# If CUDA is enabled, define a macro and set CUDA properties
if(ENABLE_CUDA)
    target_compile_definitions(CeLuxLib PUBLIC CUDA_ENABLED)

    set_target_properties(CeLuxLib PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# We need spdlog and fmt
find_package(spdlog CONFIG REQUIRED)
message(STATUS "spdlog found: ${spdlog_VERSION}")

find_package(fmt CONFIG REQUIRED)
message(STATUS "fmt found: ${fmt_VERSION}")

target_include_directories(CeLuxLib PUBLIC
    ${CMAKE_SOURCE_DIR}/include/CeLux
    ${CMAKE_SOURCE_DIR}/include/CeLux/conversion
    ${CMAKE_SOURCE_DIR}/include/CeLux/conversion/cpu
    ${CMAKE_SOURCE_DIR}/include/CeLux/conversion/gpu
    ${CMAKE_SOURCE_DIR}/include/CeLux/conversion/gpu/cuda
    ${CMAKE_SOURCE_DIR}/include/CeLux/error
    ${CMAKE_SOURCE_DIR}/include/CeLux/backends
    ${CMAKE_SOURCE_DIR}/include/CeLux/filters
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include
    ${FFMPEG_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
    ${spdlog_INCLUDE_DIRS}
    ${fmt_INCLUDE_DIRS}
)

if(ENABLE_CUDA)
    target_include_directories(CeLuxLib PUBLIC "${CUDA_TOOLKIT_ROOT_DIR}/include")
endif()

set_target_properties(CeLuxLib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# ------------------------------
# 10. Link Libraries
# ------------------------------

if(WIN32)
    target_link_libraries(CeLuxLib PUBLIC
        ws2_32
        bcrypt
        secur32
        mfplat
        strmiids
        ${MFUUUID_LIB}
        ${MF_LIB}
        ${OLE32_LIB}
        ${FFMPEG_STATIC_LIBS}
        ${TORCH_LIBRARIES}
        spdlog::spdlog_header_only
        fmt::fmt-header-only
    )
else()
    target_link_libraries(CeLuxLib PUBLIC
        ${FFMPEG_STATIC_LIBS}
        ${TORCH_LIBRARIES}
        spdlog::spdlog_header_only
        fmt::fmt-header_only
        pthread
        dl
    )
endif()

if(ENABLE_CUDA)
    if(WIN32)
        target_link_libraries(CeLuxLib PUBLIC
            CUDA::cudart
            "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib"
            ${NPP_LIBRARIES}
        )
    else()
        target_link_libraries(CeLuxLib PUBLIC
            CUDA::cudart
            "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcuda.so"
            ${NPP_LIBRARIES}
        )
    endif()
endif()

# ------------------------------
# 11. Build the Python Module
# ------------------------------

add_library(celux MODULE ${CMAKE_SOURCE_DIR}/src/CeLux/Python/Bindings.cpp)

# Compiler optimizations
if(WIN32)
    target_compile_options(celux PRIVATE /O2 /Ob2 /Oi /Ot /GL)
else()
    target_compile_options(celux PRIVATE -O3)
endif()

if(ENABLE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
    # Turn off object-level PDB generation on CUDA files
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/Z7\"")
endif()


# Enable verbose build output
set(CMAKE_VERBOSE_MAKEFILE ON)
# Enable Ninja verbose output
add_compile_options(-v)
# Add verbose flag to CUDA compiler
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -v")


target_include_directories(celux PUBLIC
    ${CMAKE_SOURCE_DIR}/include/CeLux
    ${CMAKE_SOURCE_DIR}/include/CeLux/conversion
    ${CMAKE_SOURCE_DIR}/include/CeLux/conversion/cpu
    ${CMAKE_SOURCE_DIR}/include/CeLux/conversion/gpu
    ${CMAKE_SOURCE_DIR}/include/CeLux/conversion/gpu/cuda
    ${CMAKE_SOURCE_DIR}/include/CeLux/error
    ${CMAKE_SOURCE_DIR}/include/CeLux/backends
    ${CMAKE_SOURCE_DIR}/include/CeLux/python
    ${CMAKE_SOURCE_DIR}/include/CeLux/filters
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include
    ${FFMPEG_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
    ${spdlog_INCLUDE_DIRS}
    ${fmt_INCLUDE_DIRS}
)

# Set module suffix for Python library
if(WIN32)
    set(MODULE_SUFFIX ".pyd")
else()
    set(MODULE_SUFFIX ".so")
endif()

set_target_properties(celux PROPERTIES
    PREFIX ""
    SUFFIX "${MODULE_SUFFIX}"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Ensure the output directory exists
if(ENABLE_CUDA)
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/celux_cuda")
    set(LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/celux_cuda")
    set(RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/celux_cuda")
    set(ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib_cuda")
else()
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/celux")
    set(LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/celux")
    set(RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/celux")
    set(ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()

set_target_properties(celux PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY}
    RUNTIME_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_DIRECTORY}
    ARCHIVE_OUTPUT_DIRECTORY ${ARCHIVE_OUTPUT_DIRECTORY}
)

# Link libraries to the Python module
if(WIN32)
    target_link_libraries(celux PUBLIC
        CeLuxLib
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARY}
        ${FFMPEG_STATIC_LIBS}
        ${PYTHON_LIBRARY}
        spdlog::spdlog_header_only
        fmt::fmt-header-only
    )
else()
    target_link_libraries(celux PUBLIC
        CeLuxLib
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARY}
        ${FFMPEG_STATIC_LIBS}
        ${PYTHON_LIBRARY}
        spdlog::spdlog_header_only
        fmt::fmt-header-only
        pthread
        dl
    )
endif()

if(ENABLE_CUDA)
    target_link_libraries(celux PUBLIC CUDA::cudart)
endif()

# ------------------------------
# 12. Optional: Clean Up Other Output Directories
# ------------------------------

# Example: unify library outputs for CeLuxLib
set_target_properties(CeLuxLib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${ARCHIVE_OUTPUT_DIRECTORY}"
)

message(STATUS "CMake configuration complete. Building with CUDA = ${ENABLE_CUDA}.")
