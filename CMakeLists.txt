cmake_minimum_required(VERSION 3.23)

# ─── Auto-detect vcpkg from local directory ───
# If CMAKE_TOOLCHAIN_FILE is not set, try to find vcpkg in the source directory
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND NOT DEFINED ENV{CMAKE_TOOLCHAIN_FILE})
  set(_local_vcpkg_toolchain "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
  if(EXISTS "${_local_vcpkg_toolchain}")
    set(CMAKE_TOOLCHAIN_FILE "${_local_vcpkg_toolchain}" CACHE FILEPATH "vcpkg toolchain file")
    message(STATUS "Auto-detected local vcpkg: ${CMAKE_TOOLCHAIN_FILE}")
    
    # Also set default triplet if not set
    if(NOT DEFINED VCPKG_TARGET_TRIPLET AND NOT DEFINED ENV{VCPKG_TARGET_TRIPLET})
      if(WIN32)
        set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg target triplet")
      else()
        set(VCPKG_TARGET_TRIPLET "x64-linux-dynamic" CACHE STRING "vcpkg target triplet")
      endif()
      message(STATUS "Using default triplet: ${VCPKG_TARGET_TRIPLET}")
    endif()
  endif()
endif()

project(CeLux VERSION 1.0 LANGUAGES CXX)

# ─── Compiler options ─── 
# Note: These apply to CXX compiler only. CUDA compiler options are set separately.
if (MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>                   # treat all .cpp/.hpp as UTF-8 (kills C4828)
    $<$<COMPILE_LANGUAGE:CXX>:/wd4267>                  # disable C4267: conversion from 'size_t' to 'int'
    $<$<COMPILE_LANGUAGE:CXX>:/wd4828>                  # disable C4828: illegal character in source
    $<$<COMPILE_LANGUAGE:CXX>:/external:anglebrackets>  # mark <> includes as "external"
    $<$<COMPILE_LANGUAGE:CXX>:/external:W0>             # suppress all warnings in external headers
  )
endif()

# Force only Release config for multi-config generators
if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
endif()
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD        17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CELUX_ENABLE_CUDA   "Enable CUDA support in CeLux" OFF)
option(CELUX_BUILD_PYTHON "Build Python bindings"     ON)

# Host-side performance options (C++ only)
# - AVX2: targets modern x86_64 CPUs; can be disabled for broader compatibility.
# - LTO: enables interprocedural optimization (IPO/LTO) in Release when supported.
option(CELUX_ENABLE_AVX2 "Enable AVX2 for host (C++) code" ON)
option(CELUX_ENABLE_LTO  "Enable IPO/LTO for Release builds" ON)

# --- Enable CUDA language if requested ---
if(CELUX_ENABLE_CUDA)
  # Try to help CMake find CUDA if not already set
  if(NOT CMAKE_CUDA_COMPILER)
    # Check CUDA_PATH environment variable (set by NVIDIA installer)
    if(DEFINED ENV{CUDA_PATH})
      set(_cuda_path "$ENV{CUDA_PATH}")
      file(TO_CMAKE_PATH "${_cuda_path}" _cuda_path)
      if(WIN32)
        set(_nvcc_candidate "${_cuda_path}/bin/nvcc.exe")
      else()
        set(_nvcc_candidate "${_cuda_path}/bin/nvcc")
      endif()
      if(EXISTS "${_nvcc_candidate}")
        set(CMAKE_CUDA_COMPILER "${_nvcc_candidate}" CACHE FILEPATH "CUDA compiler" FORCE)
        message(STATUS "Found nvcc via CUDA_PATH: ${CMAKE_CUDA_COMPILER}")
      endif()
    endif()

    # Also check CUDAToolkit_ROOT
    if(NOT CMAKE_CUDA_COMPILER AND DEFINED ENV{CUDAToolkit_ROOT})
      set(_cuda_root "$ENV{CUDAToolkit_ROOT}")
      file(TO_CMAKE_PATH "${_cuda_root}" _cuda_root)
      if(WIN32)
        set(_nvcc_candidate "${_cuda_root}/bin/nvcc.exe")
      else()
        set(_nvcc_candidate "${_cuda_root}/bin/nvcc")
      endif()
      if(EXISTS "${_nvcc_candidate}")
        set(CMAKE_CUDA_COMPILER "${_nvcc_candidate}" CACHE FILEPATH "CUDA compiler" FORCE)
        message(STATUS "Found nvcc via CUDAToolkit_ROOT: ${CMAKE_CUDA_COMPILER}")
      endif()
    endif()
  endif()

  # Newer MSVC toolsets (e.g. VS 2026) may not yet be officially supported by
  # the installed CUDA Toolkit. NVCC hard-errors during compiler identification
  # unless we opt in to the override.
  #
  # We only enable this automatically for MSVC versions newer than VS 2022
  # (MSVC 19.3x => MSVC_VERSION 1930-1939).
  if(MSVC AND MSVC_VERSION GREATER 1939)
    string(APPEND CMAKE_CUDA_FLAGS_INIT " -allow-unsupported-compiler")
    message(STATUS "MSVC ${MSVC_VERSION} detected; adding NVCC flag: -allow-unsupported-compiler")
  endif()

  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
    # Set CUDA architectures for common GPUs (can be overridden)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90" CACHE STRING "CUDA architectures")
    endif()
  else()
    message(WARNING "CELUX_ENABLE_CUDA is ON but no CUDA compiler found. CUDA features disabled.")
    message(WARNING "Set CUDA_PATH environment variable or CMAKE_CUDA_COMPILER to enable CUDA.")
    set(CELUX_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA support in CeLux" FORCE)
  endif()
endif()

# --- Compiler cache (ccache) ---
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
  set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  if(CELUX_ENABLE_CUDA)
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  endif()
endif()

# --- Dependencies ---
find_package(spdlog CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(libyuv REQUIRED)

# Support both exported names (fmt changed the hyphen/underscore in some builds)
if(TARGET fmt::fmt-header-only)
  set(FMT_HO_TARGET fmt::fmt-header-only)
elseif(TARGET fmt::fmt-header_only)
  set(FMT_HO_TARGET fmt::fmt-header_only)
else()
  message(FATAL_ERROR "fmt header-only target not found. Install a package that exports fmt::fmt-header-only.")
endif()

# --- FFmpeg ---
# Uses pre-built FFmpeg from external/ffmpeg directory
# Run: .\tools\download_ffmpeg.ps1 to download FFmpeg before building
set(CELUX_FFMPEG_DIR "${CMAKE_SOURCE_DIR}/external/ffmpeg")

if(NOT EXISTS "${CELUX_FFMPEG_DIR}/include/libavcodec/avcodec.h")
  message(FATAL_ERROR 
    "FFmpeg not found at ${CELUX_FFMPEG_DIR}\n"
    "Run: .\\tools\\download_ffmpeg.ps1 to download pre-built FFmpeg")
endif()

message(STATUS "Using external FFmpeg from: ${CELUX_FFMPEG_DIR}")

# Set include directories
set(CELUX_FFMPEG_INCS "${CELUX_FFMPEG_DIR}/include")

# Find import libraries
if(WIN32)
  set(_ffmpeg_lib_dir "${CELUX_FFMPEG_DIR}/lib")
  set(CELUX_FFMPEG_LIBS
    "${_ffmpeg_lib_dir}/avcodec.lib"
    "${_ffmpeg_lib_dir}/avformat.lib"
    "${_ffmpeg_lib_dir}/avutil.lib"
    "${_ffmpeg_lib_dir}/avfilter.lib"
    "${_ffmpeg_lib_dir}/swscale.lib"
    "${_ffmpeg_lib_dir}/swresample.lib"
    "${_ffmpeg_lib_dir}/avdevice.lib"
  )
else()
  set(_ffmpeg_lib_dir "${CELUX_FFMPEG_DIR}/lib")
  set(CELUX_FFMPEG_LIBS
    "${_ffmpeg_lib_dir}/libavcodec.so"
    "${_ffmpeg_lib_dir}/libavformat.so"
    "${_ffmpeg_lib_dir}/libavutil.so"
    "${_ffmpeg_lib_dir}/libavfilter.so"
    "${_ffmpeg_lib_dir}/libswscale.so"
    "${_ffmpeg_lib_dir}/libswresample.so"
    "${_ffmpeg_lib_dir}/libavdevice.so"
  )
endif()

# Verify libraries exist
foreach(_lib ${CELUX_FFMPEG_LIBS})
  if(NOT EXISTS "${_lib}")
    message(FATAL_ERROR "FFmpeg library not found: ${_lib}")
  endif()
endforeach()

message(STATUS "FFmpeg includes: ${CELUX_FFMPEG_INCS}")
message(STATUS "FFmpeg libraries: ${CELUX_FFMPEG_LIBS}")

if(CELUX_BUILD_PYTHON)
  find_package(Python3 3.13 REQUIRED COMPONENTS Interpreter Development.Module)
  message(STATUS "Python3: ${Python3_EXECUTABLE}")
  message(STATUS "Python3 includes: ${Python3_INCLUDE_DIRS}")

  # Determine site-packages path (needed for PyTorch and fallback pybind11)
  execute_process(
    COMMAND ${Python3_EXECUTABLE}
            -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE Python3_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message(STATUS "Python site-packages prefix: ${Python3_SITE_PACKAGES}")

  # -- Find pybind11 --
  find_package(pybind11 CONFIG)
  if(pybind11_FOUND)
    message(STATUS "pybind11 (CMake): ${pybind11_VERSION}")
    set(CELUX_PYBIND11_INCS ${pybind11_INCLUDE_DIRS})
    set(CELUX_PYBIND11_LIBS pybind11::module)
  else()
    message(STATUS "pybind11 not found via CMake; falling back to pip headers")
    
    if(WIN32)
      set(CELUX_PYBIND11_INCS
          "${Python3_SITE_PACKAGES}/Lib/site-packages/pybind11/include")
    else()
      set(CELUX_PYBIND11_INCS
          "${Python3_SITE_PACKAGES}/pybind11/include")
    endif()
    set(CELUX_PYBIND11_LIBS "")
    if(NOT EXISTS "${CELUX_PYBIND11_INCS}/pybind11/pybind11.h")
      message(FATAL_ERROR "pybind11 headers missing; run: pip install pybind11")
    endif()
  endif()

  # -- PyTorch paths (detected dynamically via Python import) --
  execute_process(
    COMMAND ${Python3_EXECUTABLE}
            -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_PACKAGE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _torch_result
  )
  
  if(_torch_result EQUAL 0 AND EXISTS "${TORCH_PACKAGE_DIR}")
    set(CELUX_PYTORCH_LIBDIR "${TORCH_PACKAGE_DIR}/lib")
    set(CELUX_PYTORCH_INCLUDEDIR "${TORCH_PACKAGE_DIR}/include")
    
    # Main headers: torch/include and torch/include/torch/csrc/api/include
    include_directories(
      ${CELUX_PYTORCH_INCLUDEDIR}
      ${CELUX_PYTORCH_INCLUDEDIR}/torch/csrc/api/include
    )
    # Add the lib directory to the linker
    link_directories(${CELUX_PYTORCH_LIBDIR})
    
    message(STATUS "Using torch headers: ${CELUX_PYTORCH_INCLUDEDIR}")
    message(STATUS "Using torch libs:    ${CELUX_PYTORCH_LIBDIR}")
  else()
    message(FATAL_ERROR "PyTorch not found. Run: pip install torch")
  endif()
endif()

file(GLOB_RECURSE CELUX_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/conversion/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/backends/*.cpp
)

# Add CUDA sources if CUDA is enabled  
if(CELUX_ENABLE_CUDA)
  # Add only the CUDA cpp files to main sources, but NOT the .cu files
  file(GLOB CELUX_CUDA_CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/backends/cuda/*.cpp
  )
  list(APPEND CELUX_SOURCES ${CELUX_CUDA_CPP_SOURCES})
  
  # Create a separate CUDA library for .cu files
  file(GLOB CELUX_CUDA_CU_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/backends/cuda/*.cu
  )
  
  if(CELUX_CUDA_CU_SOURCES)
    add_library(CeLuxCuda STATIC ${CELUX_CUDA_CU_SOURCES})
    set_target_properties(CeLuxCuda PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF
      POSITION_INDEPENDENT_CODE ON
      CUDA_STANDARD 17
      CUDA_RUNTIME_LIBRARY Shared
    )
    target_include_directories(CeLuxCuda PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux
      ${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/backends
      ${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/backends/cuda
    )
    target_compile_definitions(CeLuxCuda PUBLIC CELUX_ENABLE_CUDA)
    # Do NOT link CUDA::cudart here - let the runtime be resolved at link time
    message(STATUS "CUDA .cu sources: ${CELUX_CUDA_CU_SOURCES}")
  endif()
  
  message(STATUS "CUDA .cpp sources: ${CELUX_CUDA_CPP_SOURCES}")
endif()

add_library(CeLuxLib STATIC ${CELUX_SOURCES})

# --- Host ISA / AVX2 ---
if(CELUX_ENABLE_AVX2)
  if(MSVC)
    target_compile_options(CeLuxLib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
  else()
    target_compile_options(CeLuxLib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2>)
  endif()
endif()

# --- IPO / LTO (Release) ---
if(CELUX_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _celux_ipo_supported OUTPUT _celux_ipo_error)
  if(_celux_ipo_supported)
    set_property(TARGET CeLuxLib PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  else()
    message(WARNING "IPO/LTO requested but not supported: ${_celux_ipo_error}")
  endif()
endif()

# Link CUDA library if enabled
if(CELUX_ENABLE_CUDA)
  target_compile_definitions(CeLuxLib PUBLIC CELUX_ENABLE_CUDA)
  if(TARGET CeLuxCuda)
    target_link_libraries(CeLuxLib PUBLIC CeLuxCuda)
  endif()
endif()

target_include_directories(CeLuxLib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/backends>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/backends/cuda>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/python>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/conversion>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/conversion/cpu>
    $<INSTALL_INTERFACE:include>
    ${CELUX_FFMPEG_INCS}
    $<$<BOOL:${CELUX_BUILD_PYTHON}>:${Python3_INCLUDE_DIRS}>
    $<$<BOOL:${CELUX_ENABLE_CUDA}>:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
)

target_link_libraries(CeLuxLib PUBLIC
  ${CELUX_FFMPEG_LIBS}
  spdlog::spdlog_header_only
  ${FMT_HO_TARGET}
  yuv
  $<$<BOOL:${CELUX_BUILD_PYTHON}>:torch_cpu>
  $<$<BOOL:${CELUX_BUILD_PYTHON}>:c10>
  $<$<BOOL:${CELUX_BUILD_PYTHON}>:torch>
  $<$<BOOL:${CELUX_BUILD_PYTHON}>:torch_python>
  $<$<AND:$<BOOL:${CELUX_BUILD_PYTHON}>,$<BOOL:${CELUX_ENABLE_CUDA}>>:torch_cuda>
  $<$<AND:$<BOOL:${CELUX_BUILD_PYTHON}>,$<BOOL:${CELUX_ENABLE_CUDA}>>:c10_cuda>
)
if(WIN32)
  target_link_libraries(CeLuxLib PUBLIC ws2_32 bcrypt)
endif()
# If you want to add Linux-specific libs:
#if(UNIX AND NOT APPLE)
#  target_link_libraries(CeLuxLib PUBLIC pthread dl rt)
#endif()

if(CELUX_BUILD_PYTHON)
  add_library(celux MODULE src/CeLux/python/Bindings.cpp)

  # Match host performance flags on the Python extension module
  if(CELUX_ENABLE_AVX2)
    if(MSVC)
      target_compile_options(celux PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    else()
      target_compile_options(celux PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2>)
    endif()
  endif()

  if(CELUX_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _celux_ipo_supported_py OUTPUT _celux_ipo_error_py)
    if(_celux_ipo_supported_py)
      set_property(TARGET celux PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    else()
      message(WARNING "IPO/LTO requested but not supported for celux module: ${_celux_ipo_error_py}")
    endif()
  endif()

  target_include_directories(celux PRIVATE
    ${CELUX_PYBIND11_INCS}
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/python
  )

  target_link_libraries(celux PRIVATE
    CeLuxLib
    Python3::Module
    ${CELUX_PYBIND11_LIBS}
    $<$<BOOL:${CELUX_BUILD_PYTHON}>:torch_cpu>
    $<$<BOOL:${CELUX_BUILD_PYTHON}>:c10>
    $<$<BOOL:${CELUX_BUILD_PYTHON}>:torch>
    $<$<BOOL:${CELUX_BUILD_PYTHON}>:torch_python>
    $<$<AND:$<BOOL:${CELUX_BUILD_PYTHON}>,$<BOOL:${CELUX_ENABLE_CUDA}>>:torch_cuda>
    $<$<AND:$<BOOL:${CELUX_BUILD_PYTHON}>,$<BOOL:${CELUX_ENABLE_CUDA}>>:c10_cuda>
  )

  set_target_properties(celux PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_celux"

    # single-config fallback
    LIBRARY_OUTPUT_DIRECTORY        ${CMAKE_CURRENT_SOURCE_DIR}/celux
    RUNTIME_OUTPUT_DIRECTORY        ${CMAKE_CURRENT_SOURCE_DIR}/celux

    # multi-config overrides:
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/celux
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/celux
    LIBRARY_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/celux
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/celux
  )
  if(WIN32)
    set_target_properties(celux PROPERTIES SUFFIX ".pyd")
  else()
    set_target_properties(celux PROPERTIES SUFFIX ".so")
  endif()

  # RPATH for Linux (optional, to run .so without LD_LIBRARY_PATH)
  if(UNIX AND NOT APPLE)
    set_target_properties(celux PROPERTIES
      INSTALL_RPATH "$ORIGIN"
      BUILD_WITH_INSTALL_RPATH TRUE
    )
  endif()
endif()

if(WIN32 AND CELUX_BUILD_PYTHON)
  # DLL copy logic for Windows only
  # NOTE: FFmpeg DLLs are NOT bundled - user must have FFmpeg in their PATH
  set(CELUX_DLL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/celux")
  
  # vcpkg provides spdlog, fmt, libyuv
  if(DEFINED ENV{VCPKG_TARGET_TRIPLET})
    set(_vcpkg_triplet "$ENV{VCPKG_TARGET_TRIPLET}")
  else()
    set(_vcpkg_triplet "x64-windows")
  endif()
  
  if(DEFINED ENV{VCPKG_ROOT})
    file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" _vcpkg_root_cmake)
    set(VCPKG_BIN "${_vcpkg_root_cmake}/installed/${_vcpkg_triplet}/bin")
  else()
    set(VCPKG_BIN "${CMAKE_SOURCE_DIR}/vcpkg/installed/${_vcpkg_triplet}/bin")
  endif()
  
  file(GLOB CELUX_DLLS
    "${VCPKG_BIN}/libyuv*.dll"
    "${VCPKG_BIN}/spdlog*.dll"
    "${VCPKG_BIN}/fmt*.dll"
  )

  if(CELUX_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(CUDAToolkit_FOUND AND DEFINED CUDAToolkit_BIN_DIR)
      file(TO_CMAKE_PATH "${CUDAToolkit_BIN_DIR}" CUDA_BIN_DIR)
    elseif(DEFINED ENV{CUDA_PATH})
      file(TO_CMAKE_PATH "$ENV{CUDA_PATH}/bin" CUDA_BIN_DIR)
    endif()

    if(DEFINED CUDA_BIN_DIR)
      file(GLOB CELUX_CUDA_DLLS
        "${CUDA_BIN_DIR}/cudart64_*.dll"
        "${CUDA_BIN_DIR}/nvrtc64_*.dll"
      )
      list(APPEND CELUX_DLLS ${CELUX_CUDA_DLLS})
      message(STATUS "CUDA DLLs to bundle from ${CUDA_BIN_DIR}: ${CELUX_CUDA_DLLS}")
    else()
      message(WARNING "Could not determine CUDA BIN directory. CUDA DLLs will not be bundled!")
    endif()
  endif()

  message(STATUS "Found ${CMAKE_CURRENT_LIST_DIR} DLLs to bundle: ${CELUX_DLLS}")

  foreach(dll ${CELUX_DLLS})
    add_custom_command(TARGET celux POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "${CELUX_DLL_DIR}"
      COMMENT "Copying dependency DLL: ${dll} → ${CELUX_DLL_DIR}"
    )
  endforeach()

    # <-- install those same DLLs into the celux package!
  install(FILES ${CELUX_DLLS}
          DESTINATION celux
          OPTIONAL)   # OPTIONAL just suppresses warnings if any DLL wasn't globbed
endif()

if(UNIX AND CELUX_BUILD_PYTHON)
  # -- where vcpkg puts its .so’s for x64-linux-dynamic
  # NOTE: FFmpeg .so files are NOT bundled - user must have FFmpeg installed
  if(DEFINED ENV{VCPKG_ROOT})
    file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" _vcpkg_root_cmake)
    set(VCPKG_LIB "${_vcpkg_root_cmake}/installed/x64-linux-dynamic/lib")
  else()
    set(VCPKG_LIB "${CMAKE_SOURCE_DIR}/vcpkg/installed/x64-linux-dynamic/lib")
  endif()

  set(CELUX_SO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/celux")

  # Only bundle vcpkg-provided libs (spdlog, fmt, libyuv) - NOT FFmpeg
  file(GLOB CELUX_SOS
    "${VCPKG_LIB}/libspdlog.so*"
    "${VCPKG_LIB}/libyuv.so*"
    "${VCPKG_LIB}/libfmt.so*"
  )

  foreach(sofile ${CELUX_SOS})
    add_custom_command(TARGET celux POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${sofile}"
        "${CELUX_SO_DIR}"
      COMMENT "Copying dependency .so: ${sofile} → ${CELUX_SO_DIR}"
    )
  endforeach()
endif()

install(TARGETS celux
  LIBRARY   DESTINATION celux   # for .so on Linux
  RUNTIME   DESTINATION celux   # for .pyd on Windows
)

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/celux/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/celux/py.typed
  ${CMAKE_CURRENT_SOURCE_DIR}/celux/_celux.pyi
  DESTINATION celux
)

# debug dump of the variables we care about
message(STATUS "---- CMake Debug Dump ----")
message(STATUS "  CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "  VCPKG_ROOT (env)       = $ENV{VCPKG_ROOT}")
message(STATUS "  VCPKG_TARGET_TRIPLET   = $ENV{VCPKG_TARGET_TRIPLET}")
message(STATUS "  CMAKE_PREFIX_PATH      = ${CMAKE_PREFIX_PATH}")
message(STATUS "  CMAKE_MODULE_PATH      = ${CMAKE_MODULE_PATH}")
message(STATUS "  Python3_EXECUTABLE     = ${Python3_EXECUTABLE}")
message(STATUS "  Python3_ROOT_DIR       = ${Python3_ROOT_DIR}")
message(STATUS "----------------------------")
