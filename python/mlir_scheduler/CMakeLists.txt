################################################################################
# Generate packages and shared library
################################################################################

set(_source_components
  MLIRPythonSources.Core

  DataflowSchedulerDialectsPythonSources
  DataflowSchedulerDialectsPythonExtensions
)

if(MLIR_PYTHON_STUBGEN_ENABLED)
  list(APPEND _source_components MLIRPythonExtension.Core.type_stub_gen)
endif()

add_mlir_python_common_capi_library(DataflowSchedulerDialectsPythonCAPI
  INSTALL_COMPONENT DataflowSchedulerDialectsPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${PYTHON_PACKAGE_DIR}/_mlir_libs"
  DECLARED_SOURCES ${_source_components}
)
if(MLIR_LINK_MLIR_DYLIB)
  get_property(_all_mlir_libs GLOBAL PROPERTY MLIR_ALL_LIBS)
  set_property(TARGET DataflowSchedulerDialectsPythonCAPI APPEND PROPERTY
    MLIR_AGGREGATE_EXCLUDE_LIBS "${_all_mlir_libs}"
  )
  target_link_libraries(DataflowSchedulerDialectsPythonCAPI PUBLIC MLIR)
endif()

# FIXME: It is unclear why this work-around is needed on some platforms.
set_target_properties(DataflowSchedulerDialectsPythonCAPI PROPERTIES LINKER_LANGUAGE CXX)

add_mlir_python_modules(DataflowSchedulerDialectsPythonModules
  ROOT_PREFIX "${PYTHON_PACKAGE_DIR}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    DataflowSchedulerDialectsPythonCAPI
)
