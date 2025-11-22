cmake_minimum_required(VERSION 3.23)
project(CeLux VERSION 1.0 LANGUAGES CXX)
# ─── MSVC: enforce UTF-8, suppress size_t→int & illegal-char warnings, silence external headers ───
if (MSVC)
  add_compile_options(
    /utf-8                   # treat all .cpp/.hpp as UTF-8 (kills C4828)
    /wd4267                  # disable C4267: conversion from 'size_t' to 'int'
    /wd4828                  # disable C4828: illegal character in source
    /external:anglebrackets  # mark <> includes as “external”
    /external:W0             # suppress all warnings in external headers
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

if(DEFINED ENV{VCPKG_ROOT} AND DEFINED VCPKG_TARGET_TRIPLET)
  set(_vcpkg_ffmpeg_mod
      "$ENV{VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}/share/ffmpeg")
  file(TO_CMAKE_PATH "${_vcpkg_ffmpeg_mod}" _vcpkg_ffmpeg_mod)
  message(STATUS "Appending vcpkg FFmpeg module path: ${_vcpkg_ffmpeg_mod}")
  list(APPEND CMAKE_MODULE_PATH "${_vcpkg_ffmpeg_mod}")
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(FFMPEG REQUIRED COMPONENTS avcodec avformat avutil avfilter
                                  swscale swresample avdevice)

set(CELUX_FFMPEG_LIBS ${FFMPEG_LIBRARIES})
set(CELUX_FFMPEG_INCS ${FFMPEG_INCLUDE_DIRS})

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

  # -- PyTorch paths (platform-specific) --
  if(WIN32)
    set(CELUX_PYTORCH_LIBDIR "${Python3_SITE_PACKAGES}/Lib/site-packages/torch/lib")
    set(CELUX_PYTORCH_INCLUDEDIR "${Python3_SITE_PACKAGES}/Lib/site-packages/torch/include")
  else()
    set(CELUX_PYTORCH_LIBDIR "${Python3_SITE_PACKAGES}/torch/lib")
    set(CELUX_PYTORCH_INCLUDEDIR "${Python3_SITE_PACKAGES}/torch/include")
  endif()

  # Main headers: torch/include and torch/include/torch/csrc/api/include
  include_directories(
    ${CELUX_PYTORCH_INCLUDEDIR}
    ${CELUX_PYTORCH_INCLUDEDIR}/torch/csrc/api/include
    ${CELUX_PYTORCH_INCLUDEDIR}
  )
  # Add the lib directory to the linker
  link_directories(${CELUX_PYTORCH_LIBDIR})

  message(STATUS "Using Python site-packages torch headers: ${CELUX_PYTORCH_INCLUDEDIR}")
  message(STATUS "Using Python site-packages torch libs:    ${CELUX_PYTORCH_LIBDIR}")
endif()

file(GLOB_RECURSE CELUX_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/conversion/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/backends/*.cpp
)

add_library(CeLuxLib STATIC ${CELUX_SOURCES})

target_include_directories(CeLuxLib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/backends>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/python>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/conversion>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/conversion/cpu>
    $<INSTALL_INTERFACE:include>
    ${CELUX_FFMPEG_INCS}
    $<$<BOOL:${CELUX_BUILD_PYTHON}>:${Python3_INCLUDE_DIRS}>
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
  if(DEFINED ENV{VCPKG_ROOT})
    file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" _vcpkg_root_cmake)
    set(VCPKG_BIN "${_vcpkg_root_cmake}/installed/x64-windows/bin")
  else()
    set(VCPKG_BIN "${CMAKE_SOURCE_DIR}/vcpkg/installed/x64-windows/bin")
  endif()
  set(CELUX_DLL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/celux")
  file(GLOB CELUX_DLLS
    "${VCPKG_BIN}/av*.dll"
    "${VCPKG_BIN}/sw*.dll"
    "${VCPKG_BIN}/ffmpeg*.dll"
    "${VCPKG_BIN}/libx264*.dll"
    "${VCPKG_BIN}/libx265*.dll"
    "${VCPKG_BIN}/libyuv*.dll"
    "${VCPKG_BIN}/spdlog*.dll"
    "${VCPKG_BIN}/fmt*.dll"
    "${VCPKG_BIN}/*jpeg*.dll"
    "${VCPKG_BIN}/dav1d.dll"
  )
  
  foreach(dll ${CELUX_DLLS})
    add_custom_command(TARGET celux POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "${CELUX_DLL_DIR}"
      COMMENT "Copying dependency DLL: ${dll} → ${CELUX_DLL_DIR}"
    )
  endforeach()

    # <-- install those same DLLs into the celux package!
  install(FILES ${CELUX_DLLS}
          DESTINATION celux
          OPTIONAL)   # OPTIONAL just suppresses warnings if any DLL wasn’t globbed
endif()

if(UNIX AND CELUX_BUILD_PYTHON)
  # -- where vcpkg puts its .so’s for x64-linux-dynamic
  if(DEFINED ENV{VCPKG_ROOT})
    file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" _vcpkg_root_cmake)
    set(VCPKG_LIB "${_vcpkg_root_cmake}/installed/x64-linux-dynamic/lib")
  else()
    set(VCPKG_LIB "${CMAKE_SOURCE_DIR}/vcpkg/installed/x64-linux-dynamic/lib")
  endif()

  set(CELUX_SO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/celux")

  # glob all the .so files we depend on
  file(GLOB CELUX_SOS
    "${VCPKG_LIB}/libavcodec.so*"
    "${VCPKG_LIB}/libavformat.so*"
    "${VCPKG_LIB}/libavutil.so*"
    "${VCPKG_LIB}/libswscale.so*"
    "${VCPKG_LIB}/libswresample.so*"
    "${VCPKG_LIB}/libx264.so*"
    "${VCPKG_LIB}/libx265.so*"
    "${VCPKG_LIB}/libspdlog.so*"
    "${VCPKG_LIB}/libavfilter.so*"
    "${VCPKG_LIB}/libyuv.so*"
    "${VCPKG_LIB}/libfmt.so*"
    "${VCPKG_LIB}/libturbojpeg.so*"
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

