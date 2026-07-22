add_dataflow_scheduler_dialects_library(DataflowSchedulerDialectsKTDFLoweringDialect
    KTDFLoweringDialect.cpp
    KTDFLoweringOps.cpp
  
  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Dialect/KTDFLowering
    ${DataflowScheduler_BINARY_DIR}/include/dataflow-scheduler/Dialect/KTDFLowering

  GROUP DIALECT
  EXPORT_NAME KTDFLoweringDialect

  DEPENDS
    DataflowSchedulerDialectsKTDFLoweringIncGen

  LINK_LIBS PUBLIC
    DataflowSchedulerDialects::KTDFDialect

  LINK_MLIR_LIBS
    MLIRIR
    MLIRControlFlowInterfaces
)
