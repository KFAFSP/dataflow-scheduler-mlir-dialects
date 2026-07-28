add_dataflow_scheduler_dialects_library(DataflowSchedulerDialectsDataflowDialect
    DataflowAttributes.cpp
    DataflowDialect.cpp
    DataflowEnums.cpp
    DataflowOpInterfaces.cpp
    DataflowOps.cpp
    DataflowTypes.cpp
    Utils.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowSchedulerDialects_SOURCE_DIR}/include/dataflow-scheduler/Dialect/Dataflow
    ${DataflowSchedulerDialects_BINARY_DIR}/include/dataflow-scheduler/Dialect/Dataflow

  GROUP DIALECT
  EXPORT_NAME DataflowDialect

  DEPENDS
    DataflowSchedulerDialectsDataflowIncGen

  LINK_MLIR_LIBS
    MLIRIR
    MLIRSCFDialect
    MLIRArithDialect
    MLIRAffineDialect
    MLIRAffineAnalysis
    MLIRSCFDialect
)
