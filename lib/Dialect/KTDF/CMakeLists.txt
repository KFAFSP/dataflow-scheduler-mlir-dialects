add_dataflow_scheduler_dialects_library(DataflowSchedulerDialectsKTDFDialect
    KTDFAttributes.cpp
    KTDFDialect.cpp
    KTDFEnums.cpp
    KTDFOps.cpp
    KTDFTypes.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Dialect/KTDF
    ${DataflowScheduler_BINARY_DIR}/include/dataflow-scheduler/Dialect/KTDF

  GROUP DIALECT
  EXPORT_NAME KTDFDialect

  DEPENDS
    DataflowSchedulerDialectsKTDFIncGen

  LINK_MLIR_LIBS
    MLIRArithDialect
    MLIRControlFlowInterfaces
    MLIRSideEffectInterfaces
    MLIRDestinationStyleOpInterface
    MLIRDialectUtils
    MLIRIR
    MLIRSCFDialect
)
