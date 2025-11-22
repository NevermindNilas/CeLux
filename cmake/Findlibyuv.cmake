# Tries to find libyuv

find_path(LIBYUV_INCLUDE_DIR NAMES libyuv.h PATH_SUFFIXES include)
find_library(LIBYUV_LIBRARY NAMES yuv libyuv)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(libyuv DEFAULT_MSG LIBYUV_LIBRARY LIBYUV_INCLUDE_DIR)

if(libyuv_FOUND)
  if(NOT TARGET yuv)
    add_library(yuv UNKNOWN IMPORTED)
    set_target_properties(yuv PROPERTIES
      IMPORTED_LOCATION "${LIBYUV_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${LIBYUV_INCLUDE_DIR}"
    )
  endif()
endif()
