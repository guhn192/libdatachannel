#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "LibJuice::LibJuice" for configuration "Release"
set_property(TARGET LibJuice::LibJuice APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LibJuice::LibJuice PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libjuice.so.1.3.4"
  IMPORTED_SONAME_RELEASE "libjuice.so.1.3.4"
  )

list(APPEND _IMPORT_CHECK_TARGETS LibJuice::LibJuice )
list(APPEND _IMPORT_CHECK_FILES_FOR_LibJuice::LibJuice "${_IMPORT_PREFIX}/lib/libjuice.so.1.3.4" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
