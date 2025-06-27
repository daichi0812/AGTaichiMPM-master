include( CMakeSourceFiles.txt )
SET( CMAKE_CXX_COMPILER /usr/bin/clang++ )
set(CMAKE_CXX_STANDARD 17)
add_definitions( -std=c++17 -Wall )
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


# Track the Git commit within the source
include( GetGitRevisionDescription )
get_git_head_revision( GIT_REFSPEC GIT_SHA1 )

# Track the build time
string( TIMESTAMP BUILD_DATE_TIME "%Y-%m-%d %H:%M:%S" )

# Create a file with build information
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/CompileDefinitions.h.in" "${CMAKE_CURRENT_SOURCE_DIR}/CompileDefinitions.h" @ONLY )

# if( CMAKE_Fortran_COMPILER )
#   set_source_files_properties( CompileDefinitions.h CompileDefinitions.cpp PROPERTIES COMPILE_DEFINITIONS FORTRAN_FOUND )
# endif()

# if( USE_HDF5 )
#   set( Headers "${Headers};HDF5File.h" )
#   set( Sources "${Sources};HDF5File.cpp" )
# endif()

add_executable( ParticleSkinner ${Headers} ${Sources} )
# if( ENABLE_IWYU )
#   set_property( TARGET scisim PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${iwyu_path} )
# endif()

# Add a reliable indicator of the current compiler
if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
  target_compile_definitions( ParticleSkinner PUBLIC CMAKE_DETECTED_GNU_COMPILER )
elseif( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" )
  target_compile_definitions( ParticleSkinner PUBLIC CMAKE_DETECTED_INTEL_COMPILER )
elseif( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
  target_compile_definitions( ParticleSkinner PUBLIC CMAKE_DETECTED_CLANG_COMPILER )
elseif( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" )
  target_compile_definitions( ParticleSkinner PUBLIC CMAKE_DETECTED_CLANG_COMPILER )
else()
  message( FATAL_ERROR "Unsupported C++ compiler ${CMAKE_CXX_COMPILER_ID}. Please email the maintainer." )
endif()

# Eigen3 is header only and required for callers
find_package( Eigen3 3.2.9 REQUIRED )
target_include_directories( ParticleSkinner SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR} )
# Disable Eigen's built in multithreading as we will use OpenMP around Eigen
target_compile_definitions( ParticleSkinner PUBLIC EIGEN_DONT_PARALLELIZE )
if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
  # EIGEN_DONT_ALIGN also enables EIGEN_DONT_VECTORIZE
  target_compile_definitions( ParticleSkinner PUBLIC EIGEN_DONT_ALIGN EIGEN_INITIALIZE_MATRICES_BY_NAN )
endif()


#find_package( HDF5 REQUIRED C )
find_package( HDF5 REQUIRED )
target_include_directories( ParticleSkinner SYSTEM PUBLIC ${HDF5_INCLUDE_DIRS} )
target_compile_definitions( ParticleSkinner PUBLIC USE_HDF5 )
# target_link_libraries( ParticleSkinner INTERFACE ${HDF5_C_LIBRARIES} )
target_link_libraries( ParticleSkinner ${HDF5_LIBRARIES} )


find_package( RapidXML REQUIRED )
target_include_directories( ParticleSkinner SYSTEM PRIVATE ${RAPIDXML_INCLUDE_DIR} )
