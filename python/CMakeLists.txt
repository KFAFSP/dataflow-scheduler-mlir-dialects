include(AddDataflowSchedulerDialectsPython)

# NOTE: These are for tablegen only!
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)

# Scheduler bindings contain their own copy of the MLIR bindings.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

# Where all the Python sources are.
set(PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_scheduler")
# Where all the binding library sources are.
set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib/Bindings/Python")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(DataflowSchedulerDialectsPythonSources
  ROOT_DIR "${PYTHON_ROOT_DIR}"
)
if (MLIR_PYTHON_STUBGEN_ENABLED)
  declare_mlir_python_sources(DataflowSchedulerDialectsPythonSources.type_stubs
    ADD_TO_PARENT DataflowSchedulerDialectsPythonSources
  )
endif ()
declare_mlir_python_sources(DataflowSchedulerDialectsPythonSources.Dialects
  ADD_TO_PARENT DataflowSchedulerDialectsPythonSources
)
declare_mlir_python_sources(DataflowSchedulerDialectsPythonExtensions
  ROOT_DIR "${PYTHON_ROOT_DIR}"
  SOURCES
    _mlir_libs/_dataflow_scheduler_dialects/py.typed
    _mlir_libs/_dataflow_scheduler_dialects_ktdf/py.typed
    _mlir_libs/_dataflow_scheduler_dialects_ktdf_arch/py.typed
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT DataflowSchedulerDialectsPythonSources.Dialects
  ROOT_DIR "${PYTHON_ROOT_DIR}"
  TD_FILE dialects/AgenOps.td
  SOURCES 
    dialects/agen.py
  DIALECT_NAME agen
  GEN_ENUM_BINDINGS
  GEN_ENUM_BINDINGS_TD_FILE
    dialects/AgenEnums.td
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT DataflowSchedulerDialectsPythonSources.Dialects
  ROOT_DIR "${PYTHON_ROOT_DIR}"
  TD_FILE dialects/DataflowOps.td
  SOURCES 
    dialects/dataflow.py
  DIALECT_NAME dataflow
  GEN_ENUM_BINDINGS
  GEN_ENUM_BINDINGS_TD_FILE
    dialects/DataflowEnums.td
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT DataflowSchedulerDialectsPythonSources.Dialects
  ROOT_DIR "${PYTHON_ROOT_DIR}"
  TD_FILE dialects/KTDFOps.td
  SOURCES 
    dialects/ktdf.py
  DIALECT_NAME ktdf
  GEN_ENUM_BINDINGS
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT DataflowSchedulerDialectsPythonSources.Dialects
  ROOT_DIR "${PYTHON_ROOT_DIR}"
  TD_FILE dialects/KTDFArchOps.td
  SOURCES 
    dialects/ktdf_arch.py
  DIALECT_NAME ktdf_arch
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT DataflowSchedulerDialectsPythonSources.Dialects
  ROOT_DIR "${PYTHON_ROOT_DIR}"
  TD_FILE dialects/KTDFLoweringOps.td
  SOURCES 
    dialects/ktdf_lowering.py
  DIALECT_NAME ktdf_lowering
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT DataflowSchedulerDialectsPythonSources.Dialects
  ROOT_DIR "${PYTHON_ROOT_DIR}"
  TD_FILE dialects/UniformOps.td
  SOURCES 
    dialects/uniform.py
  DIALECT_NAME uniform
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT DataflowSchedulerDialectsPythonSources.Dialects
  ROOT_DIR "${PYTHON_ROOT_DIR}"
  TD_FILE dialects/VectorChainOps.td
  SOURCES 
    dialects/vectorchain.py
  DIALECT_NAME vectorchain
  GEN_ENUM_BINDINGS
  GEN_ENUM_BINDINGS_TD_FILE
    dialects/VectorChainEnums.td
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(DataflowSchedulerDialectsPythonExtensions.Main
  MODULE_NAME _dataflow_scheduler_dialects
  ADD_TO_PARENT DataflowSchedulerDialectsPythonExtensions
  SOURCES
    DataflowSchedulerDialectsModule.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPITransforms
  PRIVATE_LINK_LIBS
    LLVMSupport
)

declare_mlir_python_extension(DataflowSchedulerDialectsPythonExtensions.Dialects.KTDF.Nanobind
  MODULE_NAME _dataflow_scheduler_dialects_ktdf
  ADD_TO_PARENT DataflowSchedulerDialectsPythonSources.Dialects.ktdf
  ROOT_DIR ${PYTHON_SOURCE_DIR}
  SOURCES
    DialectKTDF.cpp
  EMBED_CAPI_LINK_LIBS
    DataflowSchedulerDialects::KTDFDialect
  PRIVATE_LINK_LIBS
    LLVMSupport
)

declare_mlir_python_extension(DataflowSchedulerDialectsPythonExtensions.Dialects.KTDFArch.Nanobind
  MODULE_NAME _dataflow_scheduler_dialects_ktdf_arch
  ADD_TO_PARENT DataflowSchedulerDialectsPythonSources.Dialects.ktdf_arch
  ROOT_DIR ${PYTHON_SOURCE_DIR}
  SOURCES
    DialectKTDFArch.cpp
  EMBED_CAPI_LINK_LIBS
    DataflowSchedulerDialects::KTDFArchDialect
  PRIVATE_LINK_LIBS
    LLVMSupport
)

if(MLIR_PYTHON_STUBGEN_ENABLED)
  add_dataflow_scheduler_dialects_python_type_stubs(DataflowSchedulerDialectsPythonExtensions.Main
    ADD_TO_PARENT DataflowSchedulerDialectsPythonSources.type_stubs
    DEPENDS_TARGETS
      DataflowSchedulerDialectsPythonModules.extension._mlir.dso
      DataflowSchedulerDialectsPythonModules.extension._dataflow_scheduler_dialects.dso
      DataflowSchedulerDialectsPythonModules.sources.MLIRPythonSources.Core
      DataflowSchedulerDialectsPythonModules.sources.MLIRPythonSources.Core.Python
      DataflowSchedulerDialectsPythonModules.sources.MLIRPythonExtension.Core.type_stub_gen
    OUTPUTS
      _dataflow_scheduler_dialects.pyi
  )

  add_dataflow_scheduler_dialects_python_type_stubs(DataflowSchedulerDialectsPythonExtensions.Dialects.KTDF.Nanobind
    ADD_TO_PARENT DataflowSchedulerDialectsPythonSources.type_stubs
    DEPENDS_TARGETS
      DataflowSchedulerDialectsPythonModules.extension._mlir.dso
      DataflowSchedulerDialectsPythonModules.extension._dataflow_scheduler_dialects_ktdf.dso
    OUTPUTS
      _dataflow_scheduler_dialects_ktdf.pyi
  )

  add_dataflow_scheduler_dialects_python_type_stubs(DataflowSchedulerDialectsPythonExtensions.Dialects.KTDFArch.Nanobind
    ADD_TO_PARENT DataflowSchedulerDialectsPythonSources.type_stubs
    DEPENDS_TARGETS
      DataflowSchedulerDialectsPythonModules.extension._mlir.dso
      DataflowSchedulerDialectsPythonModules.extension._dataflow_scheduler_dialects_ktdf_arch.dso
    OUTPUTS
      _dataflow_scheduler_dialects_ktdf_arch.pyi
  )
endif()

set(EXCLUDE_FROM_ALL)
if (NOT PROJECT_IS_TOP_LEVEL)
  set(EXCLUDE_FROM_ALL EXCLUDE_FROM_ALL)
endif ()
add_subdirectory(mlir_scheduler ${EXCLUDE_FROM_ALL})
