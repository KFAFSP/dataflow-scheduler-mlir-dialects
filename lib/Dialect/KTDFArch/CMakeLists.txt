add_subdirectory(Analysis)
add_subdirectory(Transforms)

add_dataflow_scheduler_dialects_library(DataflowSchedulerDialectsKTDFArchDialect
    KTDFArchAttributes.cpp
    KTDFArchDialect.cpp
    KTDFArchIntrinsics.cpp
    KTDFArchOpInterfaces.cpp
    KTDFArchOps.cpp
    KTDFArchTypes.cpp

  GROUP DIALECT
  EXPORT_NAME KTDFArchDialect
  
  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/dataflow-scheduler/Dialect/KTDFArch
    ${PROJECT_BINARY_DIR}/include/dataflow-scheduler/Dialect/KTDFArch

  DEPENDS
    DataflowSchedulerDialectsKTDFArchIncGen

  LINK_MLIR_LIBS
    MLIRIR
    MLIRControlFlowInterfaces
    MLIRPDLDialect
    MLIRInferTypeOpInterface
)
